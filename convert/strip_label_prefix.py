#!/usr/bin/env python3
"""Strip prefix from label JSON filenames and place into matching folders.

Example:
  9-2_8bit_TK09-NGB-B1-251011.json
  -> dst_root/9-2_8bit/TK09-NGB-B1-251011.json
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove prefix before first underscore and move/copy JSONs to matching folders.")
    parser.add_argument(
        "--src",
        default="/datasets/PAR/temp/label",
        help="Source folder containing prefixed JSON files (default: /datasets/PAR/temp/label)",
    )
    parser.add_argument(
        "--dst-root",
        default="/datasets/PAR/Weld/datasets/labels/1208",
        help="Destination root containing target folders (default: /datasets/PAR/Weld/datasets/labels/1208)",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing destination files",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    src_dir = Path(args.src)
    dst_root = Path(args.dst_root)

    if not src_dir.is_dir():
        raise SystemExit(f"Source directory not found: {src_dir}")

    files = sorted(src_dir.glob("*.json"))
    if not files:
        print(f"No JSON files found in {src_dir}")
        return 0

    moved = 0
    skipped = 0
    errors = 0

    for src in files:
        stem = src.stem
        if "_" not in stem:
            print(f"[skip] no underscore in filename: {src.name}")
            skipped += 1
            continue

        prefix, rest = stem.split("_", 1)
        if not rest:
            print(f"[skip] empty filename after prefix: {src.name}")
            skipped += 1
            continue

        dst_dir = dst_root / prefix
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f"{rest}{src.suffix}"

        if dst.exists() and not args.overwrite:
            print(f"[skip] exists: {dst}")
            skipped += 1
            continue

        try:
            if args.move:
                if dst.exists() and args.overwrite:
                    dst.unlink()
                shutil.move(str(src), str(dst))
            else:
                shutil.copy2(src, dst)
            moved += 1
        except OSError as exc:
            print(f"[error] {src.name} -> {dst}: {exc}")
            errors += 1

    print(f"Done. processed={len(files)} moved_or_copied={moved} skipped={skipped} errors={errors}")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
