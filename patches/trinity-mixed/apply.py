"""Check the dependency bundle; --apply explicitly stages it under --repo-root/deps.

Keep native upstream gitlinks unchanged. Follow manifest.json's install contract.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def git(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(repo), *args], stderr=subprocess.STDOUT)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def check(repo_root: Path, bundle: Path, dependency: dict) -> tuple[Path, Path, str]:
    name = dependency["name"]
    if name not in {"renderers", "verifiers"} or dependency["patch"] != f"{name}.patch":
        raise ValueError(f"Invalid dependency entry: {name}")
    patch = bundle / dependency["patch"]
    if digest(patch.read_bytes()) != dependency["sha256"]:
        raise ValueError(f"{name}: patch hash mismatch")
    repo = repo_root / "deps" / name
    if repo.resolve() != repo or Path(git(repo, "rev-parse", "--show-toplevel").decode().strip()) != repo:
        raise ValueError(f"{name}: expected a repository at {repo}")
    if git(repo, "rev-parse", "HEAD").decode().strip() != dependency["base"]:
        raise ValueError(f"{name}: HEAD must equal base {dependency['base']}")
    if git(repo, "diff", "--no-ext-diff", "--name-only") or git(repo, "ls-files", "--others", "--exclude-standard"):
        raise ValueError(f"{name}: unexpected unstaged or untracked changes")
    staged = git(
        repo,
        "diff",
        "--cached",
        "--binary",
        "--full-index",
        "--no-ext-diff",
        "--no-color",
        "--no-renames",
        "--src-prefix=a/",
        "--dst-prefix=b/",
        "HEAD",
    )
    if staged:
        if digest(staged) != dependency["sha256"]:
            raise ValueError(f"{name}: unexpected staged diff")
        return repo, patch, "applied"
    git(repo, "apply", "--check", "--index", str(patch))
    return repo, patch, "ready"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="Apply and stage both verified patches")
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    if Path(git(repo_root, "rev-parse", "--show-toplevel").decode().strip()) != repo_root:
        raise ValueError(f"Expected a repository root at {repo_root}")
    bundle = Path(__file__).resolve().parent
    manifest = json.loads((bundle / "manifest.json").read_text())
    if manifest["version"] != 1:
        raise ValueError("Unsupported manifest version")
    dependencies = manifest["dependencies"]
    if sorted(item["name"] for item in dependencies) != ["renderers", "verifiers"]:
        raise ValueError("Expected exactly renderers and verifiers")
    # Check both repositories and patch hashes before modifying either repository.
    states = [check(repo_root, bundle, dependency) for dependency in dependencies]
    for dependency, (repo, patch, status) in zip(dependencies, states, strict=True):
        if args.apply and status == "ready":
            git(repo, "apply", "--index", str(patch))
            _, _, status = check(repo_root, bundle, dependency)
            if status != "applied":
                raise ValueError(f"{dependency['name']}: applied diff did not match")
        print(f"{dependency['name']}: PASS ({status})")


if __name__ == "__main__":
    main()
