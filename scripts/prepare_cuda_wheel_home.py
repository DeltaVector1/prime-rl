from argparse import ArgumentParser
from pathlib import Path
from shutil import copytree

import nvidia.cuda_cccl
import nvidia.cuda_nvcc
import nvidia.cuda_runtime


def ensure_symlink(path: Path, target: Path) -> None:
    if path.is_symlink():
        if path.resolve() != target.resolve():
            raise ValueError(f"{path} points to {path.resolve()}, expected {target.resolve()}")
        return
    if path.exists():
        raise FileExistsError(f"Cannot create CUDA header link because {path} already exists")
    path.symlink_to(target, target_is_directory=target.is_dir())


def populate_include_dir(path: Path, cuda_runtime: Path, cuda_nvcc: Path) -> None:
    runtime_include = cuda_runtime / "include"
    if path.is_symlink():
        if path.resolve() != runtime_include.resolve():
            raise ValueError(f"{path} points to {path.resolve()}, expected {runtime_include.resolve()}")
        path.unlink()
    path.mkdir(parents=True, exist_ok=True)

    for entry in runtime_include.iterdir():
        ensure_symlink(path / entry.name, entry)
    ensure_symlink(path / "crt", cuda_nvcc / "include" / "crt")


def populate_compiler_bin(path: Path, nvcc_root: Path) -> None:
    source = nvcc_root / "bin"
    if path.is_symlink():
        if path.resolve() != source.resolve():
            raise ValueError(f"{path} points to {path.resolve()}, expected {source.resolve()}")
        path.unlink()
    copytree(source, path, dirs_exist_ok=True, symlinks=True)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--nvcc-root", type=Path)
    args = parser.parse_args()

    cuda_runtime = Path(next(iter(nvidia.cuda_runtime.__path__)))
    cuda_cccl = Path(next(iter(nvidia.cuda_cccl.__path__)))
    cuda_nvcc = Path(next(iter(nvidia.cuda_nvcc.__path__)))
    cuda_home = Path("/tmp/prime-rl-cuda-wheel-home")
    target_include = cuda_home / "targets" / "x86_64-linux" / "include"
    target_include.mkdir(parents=True, exist_ok=True)

    populate_include_dir(cuda_home / "include", cuda_runtime, cuda_nvcc)
    populate_include_dir(target_include, cuda_runtime, cuda_nvcc)
    for entry in (cuda_cccl / "include").iterdir():
        if entry.name == "__init__.py":
            continue
        ensure_symlink(target_include / entry.name, entry)
    ensure_symlink(target_include / "cccl", cuda_cccl / "include")
    if args.nvcc_root is not None:
        for entry in (args.nvcc_root / "targets" / "x86_64-linux" / "include").iterdir():
            ensure_symlink(target_include / entry.name, entry)
        populate_compiler_bin(cuda_home / "bin", args.nvcc_root)
        ensure_symlink(cuda_home / "nvvm", args.nvcc_root / "nvvm")
    print(cuda_home)


if __name__ == "__main__":
    main()
