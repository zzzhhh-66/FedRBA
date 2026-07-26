"""Run an experiment matrix sequentially with fail-fast logging."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", default="scripts/experiments.txt")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--stop", type=int, default=None)
    args = parser.parse_args()
    lines = [
        line.strip()
        for line in Path(args.matrix).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    stop = args.stop or len(lines)
    for line_number in range(args.start, stop + 1):
        command = lines[line_number - 1]
        print(f"[{line_number}/{len(lines)}] {command}", flush=True)
        subprocess.run(command, shell=True, check=True)


if __name__ == "__main__":
    main()

