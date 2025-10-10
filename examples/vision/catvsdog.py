"""Cat vs. dog classification example placeholder.

This module provides the wiring and dataset utilities that were previously
embedded in `tests/catvsdog.py`. The heavy training loop has been moved out of
version control so that datasets with tens of thousands of images remain local.

Usage
-----
$ python -m examples.vision.catvsdog --data-dir /path/to/PetImages --scan-only

Set the `CATDOG_DATA` environment variable if you prefer not to pass
`--data-dir` each time.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

try:
    from PIL import Image, UnidentifiedImageError
except ImportError:  # pragma: no cover - pillow optional for quick smoke tests
    Image = None  # type: ignore
    UnidentifiedImageError = Exception  # type: ignore

VALID_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
EXPECTED_SUBDIRS = ("Cat", "Dog")


@dataclass
class ScanResult:
    items: List[Tuple[str, int]]
    skipped: int


def verify_image(path: Path) -> bool:
    """Return True if image can be opened (or Pillow missing)."""
    if Image is None:
        return True
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            im.convert("RGB").load()
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False


def scan_dataset(root: Path) -> ScanResult:
    items: List[Tuple[str, int]] = []
    skipped = 0
    for label_name, label in zip(EXPECTED_SUBDIRS, (0, 1)):
        subdir = root / label_name
        if not subdir.exists():
            print(f"[warn] missing directory: {subdir}")
            continue
        for path in subdir.iterdir():
            if not path.is_file() or path.suffix not in VALID_EXTS:
                skipped += 1
                continue
            if verify_image(path):
                items.append((str(path), label))
            else:
                skipped += 1
    print(f"[scan] usable={len(items)} skipped={skipped}")
    return ScanResult(items=items, skipped=skipped)


def summary_by_label(items: Sequence[Tuple[str, int]]) -> Iterable[str]:
    counts = {0: 0, 1: 0}
    for _, label in items:
        counts[label] = counts.get(label, 0) + 1
    yield f"Cat : {counts.get(0, 0):5d} images"
    yield f"Dog : {counts.get(1, 0):5d} images"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cat vs Dog dataset helper")
    parser.add_argument(
        "--data-dir",
        default=os.getenv("CATDOG_DATA", ""),
        help="Path to the PetImages dataset (expects Cat/ and Dog/ subfolders)",
    )
    parser.add_argument(
        "--results-dir",
        default=os.getenv("CATDOG_RESULTS", "examples/results/catvsdog"),
        help="Where to place derived artefacts when you extend this script",
    )
    parser.add_argument(
        "--scan-only",
        action="store_true",
        help="Only scan the dataset and print counts (default action).",
    )
    return parser.parse_args(argv)


def ensure_results_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    print(f"[info] results directory ready: {path}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    data_dir = Path(args.data_dir) if args.data_dir else None

    if not data_dir:
        print("[error] data directory is required (set --data-dir or CATDOG_DATA)")
        return 1
    if not data_dir.exists():
        print(f"[error] data directory not found: {data_dir}")
        return 1

    ensure_results_dir(Path(args.results_dir))
    scan = scan_dataset(data_dir)
    for line in summary_by_label(scan.items):
        print("[summary]", line)

    if args.scan_only:
        print("[info] scan complete. Extend `train()` to run full training.")
        return 0

    print(
        "[todo] Implement your training loop here. The original heavy script "
        "is kept out of git (see tests/catvsdog.py, which is ignored)."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - manual execution entrypoint
    raise SystemExit(main())
