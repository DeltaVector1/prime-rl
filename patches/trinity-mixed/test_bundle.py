import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def git(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(repo), *args], stderr=subprocess.STDOUT)


def commit(repo: Path, message: str) -> str:
    git(repo, "add", ".")
    git(repo, "-c", "user.name=Bundle test", "-c", "user.email=bundle@example.invalid", "commit", "-m", message)
    return git(repo, "rev-parse", "HEAD").decode().strip()


@pytest.fixture
def assembly(tmp_path):
    root = tmp_path / "native"
    root.mkdir()
    git(root, "init")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    shutil.copyfile(Path(__file__).with_name("apply.py"), bundle / "apply.py")
    dependencies = []
    for name in ["renderers", "verifiers"]:
        repo = root / "deps" / name
        repo.mkdir(parents=True)
        git(repo, "init")
        (repo / "content").write_bytes(b"base\x00data")
        base = commit(repo, "base")
        (repo / "content").write_bytes(b"target\x00data")
        (repo / "added").write_text("added\n")
        target = commit(repo, "target")
        patch = git(repo, "diff", "--binary", "--full-index", base, target)
        (bundle / f"{name}.patch").write_bytes(patch)
        dependencies.append(
            {
                "name": name,
                "base": base,
                "target": target,
                "patch": f"{name}.patch",
                "sha256": hashlib.sha256(patch).hexdigest(),
            }
        )
        git(repo, "checkout", base)
    (bundle / "manifest.json").write_text(json.dumps({"version": 1, "dependencies": dependencies}))
    return root, bundle, dependencies


def run_bundle(assembly, *args):
    root, bundle, _ = assembly
    return subprocess.run(
        [sys.executable, str(bundle / "apply.py"), "--repo-root", str(root), *args],
        text=True,
        capture_output=True,
    )


def test_fresh_apply_check_and_idempotence(assembly):
    root, _, dependencies = assembly
    result = run_bundle(assembly)
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("PASS (ready)") == 2
    for dependency in dependencies:
        assert git(root / "deps" / dependency["name"], "status", "--porcelain") == b""
    for args in [("--apply",), (), ("--apply",)]:
        result = run_bundle(assembly, *args)
        assert result.returncode == 0, result.stderr
        assert result.stdout.count("PASS (applied)") == 2
    for dependency in dependencies:
        repo = root / "deps" / dependency["name"]
        assert git(repo, "rev-parse", "HEAD").decode().strip() == dependency["base"]
        assert git(repo, "write-tree") == git(repo, "rev-parse", f"{dependency['target']}^{{tree}}")


@pytest.mark.parametrize("fault", ["wrong_base", "corrupt_patch", "unstaged", "staged", "untracked"])
def test_rejects_fault_before_changing_either_dependency(assembly, fault):
    root, bundle, dependencies = assembly
    first = root / "deps" / "renderers"
    second = root / "deps" / "verifiers"
    if fault == "wrong_base":
        git(second, "checkout", dependencies[1]["target"])
    elif fault == "corrupt_patch":
        with (bundle / "verifiers.patch").open("ab") as output:
            output.write(b"corrupt")
    elif fault == "untracked":
        (second / "unexpected").write_text("untracked")
    else:
        (second / "content").write_text("unexpected change")
        if fault == "staged":
            git(second, "add", "content")
    before = git(second, "status", "--porcelain")
    result = run_bundle(assembly, "--apply")
    assert result.returncode != 0
    assert git(first, "status", "--porcelain") == b""
    assert git(second, "status", "--porcelain") == before
