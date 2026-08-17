from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "input_manifest.sha256"
INPUT_ROOTS = (
    PROJECT_ROOT / "data" / "training",
    PROJECT_ROOT / "data" / "temperature",
    PROJECT_ROOT / "data" / "validation",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create or verify the input-data SHA-256 manifest.")
    parser.add_argument("--write-manifest", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scan_inputs() -> dict[str, str]:
    files: dict[str, str] = {}
    for root in INPUT_ROOTS:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                relative = path.relative_to(PROJECT_ROOT).as_posix()
                files[relative] = sha256(path)
    return files


def read_manifest() -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in MANIFEST_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        digest, relative = line.split(maxsplit=1)
        entries[relative.strip()] = digest
    return entries


def main() -> None:
    args = parse_args()
    current = scan_inputs()
    if args.write_manifest:
        lines = [f"{digest}  {relative}" for relative, digest in sorted(current.items())]
        MANIFEST_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Wrote {MANIFEST_PATH} with {len(current)} input files.")
        return

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Missing manifest: {MANIFEST_PATH}")
    expected = read_manifest()
    missing = sorted(set(expected) - set(current))
    unexpected = sorted(set(current) - set(expected))
    changed = sorted(path for path in set(expected) & set(current) if expected[path] != current[path])
    if missing or unexpected or changed:
        raise RuntimeError(
            "Input-data verification failed. "
            f"missing={missing}, unexpected={unexpected}, changed={changed}"
        )
    print(f"Input-data verification passed: {len(current)} files.")


if __name__ == "__main__":
    main()
