import argparse
import hashlib
import io
import json
import os
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def client_builder(source):
    start = source.index("    let client = Client::builder()")
    end = source.index('.expect("Failed to create HTTP client");', start)
    return source[start:end + len('.expect("Failed to create HTTP client");')]


def fixture(baseline, candidate):
    clients = "\n".join(
        f"fn {name}(config: &Config) -> Client {{\n{builder}\n    client\n}}"
        for name, builder in [("baseline_client", baseline), ("candidate_client", candidate)]
    )
    return r'''
use reqwest::Client;
use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

struct Config { request_timeout_secs: u64 }
''' + clients + r'''

async fn compare_client(build: fn(&Config) -> Client, expected_connections: usize) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/health", listener.local_addr().unwrap());
    let accepted = Arc::new(AtomicUsize::new(0));
    let requests = Arc::new(AtomicUsize::new(0));
    let server_accepted = accepted.clone();
    let server_requests = requests.clone();
    let server = tokio::spawn(async move {
        let mut connections = tokio::task::JoinSet::new();
        loop {
            let (mut stream, _) = listener.accept().await.unwrap();
            server_accepted.fetch_add(1, Ordering::SeqCst);
            let requests = server_requests.clone();
            connections.spawn(async move {
                loop {
                    let mut header = Vec::new();
                    while !header.ends_with(b"\r\n\r\n") {
                        let mut byte = [0];
                        if stream.read(&mut byte).await.unwrap() == 0 { return; }
                        header.push(byte[0]);
                        assert!(header.len() <= 8192);
                    }
                    assert!(header.starts_with(b"GET /health HTTP/1.1\r\n"));
                    requests.fetch_add(1, Ordering::SeqCst);
                    stream.write_all(
                        b"HTTP/1.1 200 OK\r\nContent-Length: 1\r\nConnection: keep-alive\r\n\r\nx"
                    ).await.unwrap();
                }
            });
        }
    });
    let client = build(&Config { request_timeout_secs: 14400 });
    for _ in 0..2 {
        let response = client.get(&url).send().await.unwrap();
        assert_eq!(response.status().as_u16(), 200);
        assert_eq!(response.version(), reqwest::Version::HTTP_11);
        assert_eq!(response.bytes().await.unwrap().as_ref(), b"x");
        tokio::task::yield_now().await;
    }
    assert_eq!(requests.load(Ordering::SeqCst), 2);
    assert_eq!(accepted.load(Ordering::SeqCst), expected_connections);
    drop(client);
    server.abort();
    assert!(server.await.unwrap_err().is_cancelled());
}

#[tokio::test(flavor = "current_thread")]
async fn idle_pool_connection_counts() {
    tokio::time::timeout(Duration::from_secs(5), async {
        compare_client(baseline_client, 1).await;
        compare_client(candidate_client, 2).await;
    }).await.expect("local HTTP fixture exceeded five seconds");
}

fn main() {}
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("router_source", type=Path)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--compile-only", action="store_true")
    action.add_argument("--http-check", action="store_true")
    args = parser.parse_args()
    bundle = Path(__file__).resolve().parent
    metadata = json.loads((bundle / "router-idle-pool.metadata.json").read_text())
    for name, expected in metadata["preserved_bundle_sha256"].items():
        assert digest((bundle / name).read_bytes()) == expected, name
    patch = bundle / metadata["patch"]["path"]
    assert digest(patch.read_bytes()) == metadata["patch"]["sha256"]
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.router_source, text=True
    ).strip()
    assert head == metadata["router_base_commit"], head
    archive = subprocess.check_output(["git", "archive", head], cwd=args.router_source)
    measurements = []
    with tempfile.TemporaryDirectory(prefix="router-idle-pool-") as temp:
        root = Path(temp)
        with tarfile.open(fileobj=io.BytesIO(archive)) as archive_file:
            archive_file.extractall(root, filter="data")
        patches = [bundle / name for name in metadata["prerequisite_patches"]]
        subprocess.run(["git", "apply", "--check", *map(str, patches), str(patch)], cwd=root, check=True)
        subprocess.run(["git", "apply", *map(str, patches)], cwd=root, check=True)
        source_path = root / "src/server.rs"
        baseline = source_path.read_text()
        assert digest(source_path.read_bytes()) == metadata["patch"]["base_source_sha256"]
        subprocess.run(["git", "apply", str(patch)], cwd=root, check=True)
        candidate = source_path.read_text()
        assert baseline.count(".pool_max_idle_per_host(500)") == 1
        assert candidate == baseline.replace(".pool_max_idle_per_host(500)", ".pool_max_idle_per_host(0)")
        assert digest(source_path.read_bytes()) == metadata["patch"]["source_sha256"]
        for name, expected in metadata["preserved_source_sha256"].items():
            assert digest((root / name).read_bytes()) == expected, name
        test_source = root / "examples/router_idle_pool_check.rs"
        test_source.parent.mkdir(exist_ok=True)
        test_source.write_text(fixture(client_builder(baseline), client_builder(candidate)))
        subprocess.run(["rustfmt", "--edition", "2021", str(test_source)], check=True)
        http_status = "not_run"
        if args.compile_only or args.http_check:
            command = ["cargo", "test", "--offline", "--locked", "--example", "router_idle_pool_check"]
            if args.compile_only:
                command.append("--no-run")
            else:
                command.extend(["idle_pool_connection_counts", "--", "--test-threads=1"])
            environment = dict(os.environ)
            for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
                environment.pop(name, None)
            environment.update({"NO_PROXY": "*", "no_proxy": "*"})
            started = time.monotonic()
            result = subprocess.run(command, cwd=root, env=environment, capture_output=True, text=True)
            measurements.append({"command": command, "seconds": time.monotonic() - started,
                                 "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
            http_status = "compiled" if args.compile_only else "passed"
            if result.returncode:
                print(json.dumps({"status": "failed", "bundle_check": "passed", "measurements": measurements}))
                raise SystemExit(result.returncode)
    print(json.dumps({"status": "passed", "bundle_check": "passed", "http_check": http_status,
                      "measurements": measurements}))


if __name__ == "__main__":
    main()
