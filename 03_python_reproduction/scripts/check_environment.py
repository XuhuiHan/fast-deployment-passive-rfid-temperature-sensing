from __future__ import annotations

import importlib.metadata
import sys


EXPECTED_PYTHON = (3, 12, 13)
EXPECTED_PACKAGES = {
    "numpy": "2.3.5",
    "pandas": "3.0.1",
    "scipy": "1.18.0",
    "torch": "2.13.0",
}


def main() -> None:
    actual_python = sys.version_info[:3]
    errors = []
    if actual_python != EXPECTED_PYTHON:
        errors.append(f"Python {actual_python} != {EXPECTED_PYTHON}")

    for package, expected in EXPECTED_PACKAGES.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"{package} is not installed")
            continue
        if actual != expected:
            errors.append(f"{package} {actual} != {expected}")

    if errors:
        raise RuntimeError("Environment verification failed:\n- " + "\n- ".join(errors))

    print(f"Python {'.'.join(map(str, actual_python))}")
    for package, version in EXPECTED_PACKAGES.items():
        print(f"{package}=={version}")
    print("Environment verification passed.")


if __name__ == "__main__":
    main()
