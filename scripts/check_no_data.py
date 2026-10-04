"""Block data files, model weights and large files from entering the repository.

Run by pre-commit (staged files are passed as arguments) and by CI
(no arguments: every file tracked by git is checked).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

MAX_BYTES = 1_000_000  # 1 MB
BLOCKED_EXTENSIONS = {
    ".csv", ".tsv", ".parquet", ".feather", ".arrow", ".jsonl", ".ndjson",
    ".xlsx", ".xls", ".pkl", ".pickle", ".joblib", ".npy", ".npz",
    ".pt", ".pth", ".bin", ".safetensors", ".ckpt", ".h5", ".onnx",
    ".zip", ".tar", ".gz", ".7z", ".rar", ".db", ".sqlite", ".sqlite3",
}  # fmt: skip
ALLOWED_IN_DATA = {".gitkeep", "README.md"}
ALLOWED_PREFIXES = ("tests/fixtures/",)


def tracked_files() -> list[str]:
    result = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True)
    return [line for line in result.stdout.splitlines() if line]


def find_problems(path_str: str) -> list[str]:
    path = Path(path_str)
    posix = path.as_posix()
    problems = []

    if path.parts and path.parts[0] == "data" and path.name not in ALLOWED_IN_DATA:
        problems.append("only .gitkeep and README.md are allowed under data/")

    if path.suffix.lower() in BLOCKED_EXTENSIONS and not posix.startswith(ALLOWED_PREFIXES):
        problems.append(f"'{path.suffix}' files cannot be committed")

    if path.is_file() and path.stat().st_size > MAX_BYTES:
        problems.append(f"file exceeds the {MAX_BYTES // 1000} KB limit")

    return problems


def main(argv: list[str]) -> int:
    files = argv or tracked_files()
    failed = False
    for file in files:
        for problem in find_problems(file):
            print(f"[ERROR] {file}: {problem}")
            failed = True

    if failed:
        print("\nData and model files must not be committed. See data/README.md")
        return 1

    print(f"[OK] Data guard: {len(files)} files checked, no problems found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
