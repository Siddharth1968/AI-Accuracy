"""Stdlib-only helper for setup.bat (Python 3.10-3.12 check & NVIDIA CUDA driver check)."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys


def check_python_version() -> int:
    if (3, 10) <= sys.version_info[:2] <= (3, 12):
        print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        return 0
    return 1


def parse_cuda_driver_ok(nvidia_smi_output: str) -> bool:
    m = re.search(r"CUDA(?:\s+\w+)*\s+Version:\s*(\d+)\.(\d+)", nvidia_smi_output)
    return bool(m and (int(m.group(1)), int(m.group(2))) >= (12, 1))


def check_cuda_driver() -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi"],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=10,
        )
        if parse_cuda_driver_ok(out):
            print("CUDA_OK")
            return 0
    except Exception:
        pass
    print("NO_CUDA")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Stdlib-only environment helper for setup.bat.")
    parser.add_argument("--check-python", action="store_true")
    parser.add_argument("--check-cuda", action="store_true")
    args = parser.parse_args(argv)
    if args.check_python:
        return check_python_version()
    if args.check_cuda:
        return check_cuda_driver()
    return 0


if __name__ == "__main__":
    sys.exit(main())
