import argparse
import hashlib
import json
import subprocess
import tempfile
import time
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("router_source", type=Path)
    args = parser.parse_args()
    bundle = Path(__file__).resolve().parent
    metadata = json.loads((bundle / "router-send-diagnostics.metadata.json").read_text())
    patch = bundle / metadata["patch"]["path"]
    auth = bundle / metadata["prerequisite_auth_patch"]["path"]
    assert digest(patch.read_bytes()) == metadata["patch"]["sha256"]
    assert digest(auth.read_bytes()) == metadata["prerequisite_auth_patch"]["sha256"]
    assert digest((bundle / "metadata.json").read_bytes()) == metadata["preserved_auth_metadata_sha256"]
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.router_source, text=True
    ).strip()
    assert head == metadata["router_base_commit"], head
    measurements = []
    with tempfile.TemporaryDirectory(prefix="router-send-diagnostics-") as temp:
        root = Path(temp)
        files = ["src/routers/http/router.rs", *metadata["preserved_source_sha256"]]
        for name in files:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(subprocess.check_output(
                ["git", "show", f"{head}:{name}"], cwd=args.router_source
            ))
        source_path = root / "src/routers/http/router.rs"
        assert digest(source_path.read_bytes()) == metadata["patch"]["base_source_sha256"]
        subprocess.run(["git", "apply", "--check", str(auth), str(patch)], cwd=root, check=True)
        subprocess.run(["git", "apply", str(auth), str(patch)], cwd=root, check=True)
        assert digest(source_path.read_bytes()) == metadata["patch"]["source_sha256"]
        for name, expected in metadata["preserved_source_sha256"].items():
            assert digest((root / name).read_bytes()) == expected, name
        source = source_path.read_text()
        helper = source[source.index("fn send_error_causes("):source.index("\nfn send_error_worker_origin(")]
        tests = source[
            source.index("    mod send_error_diagnostics {"):
            source.index('    #[tokio::test(flavor = "current_thread")]')
        ]
        test_source = root / "diagnostics.rs"
        test_source.write_text(helper + "\n" + tests)
        for command in [
            ["rustc", "--edition=2021", "--test", str(test_source), "-o", str(root / "diagnostics")],
            [str(root / "diagnostics"), "--nocapture"],
        ]:
            started = time.monotonic()
            result = subprocess.run(command, capture_output=True, text=True)
            measurements.append({
                "command": command,
                "seconds": time.monotonic() - started,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            })
            if result.returncode:
                print(json.dumps({"status": "failed", "measurements": measurements}))
                result.check_returncode()
    print(json.dumps({"status": "passed", "patch_apply": "passed", "measurements": measurements}))


if __name__ == "__main__":
    main()
