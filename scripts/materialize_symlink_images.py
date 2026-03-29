#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable, List


DEFAULT_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace image symlinks with real files copied from their targets."
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="Directory containing image symlinks to materialize.",
    )
    parser.add_argument(
        "--extensions",
        nargs="*",
        default=list(DEFAULT_EXTENSIONS),
        help="Image suffixes to process. Default: common image extensions.",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only process files directly under image-dir.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be replaced without modifying files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each processed symlink.",
    )
    return parser.parse_args()


def iter_candidates(root: Path, recursive: bool) -> Iterable[Path]:
    return root.rglob("*") if recursive else root.iterdir()


def normalize_extensions(raw_extensions: Iterable[str]) -> List[str]:
    normalized = []
    for ext in raw_extensions:
        ext = ext.strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = f".{ext}"
        normalized.append(ext)
    return normalized


def build_temp_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.materializing")


def materialize_symlink(link_path: Path, dry_run: bool) -> Path:
    target_path = link_path.resolve(strict=True)
    if not target_path.is_file():
        raise FileNotFoundError(f"symlink target is not a file: {target_path}")

    if dry_run:
        return target_path

    temp_path = build_temp_path(link_path)
    if temp_path.exists() or temp_path.is_symlink():
        temp_path.unlink()

    shutil.copy2(target_path, temp_path)
    # Replace the symlink entry atomically after the file copy finishes.
    os.replace(temp_path, link_path)
    return target_path


def main() -> None:
    args = parse_args()
    image_dir = args.image_dir.resolve()
    if not image_dir.exists():
        raise FileNotFoundError(f"image dir not found: {image_dir}")
    if not image_dir.is_dir():
        raise NotADirectoryError(f"not a directory: {image_dir}")

    extensions = set(normalize_extensions(args.extensions))
    recursive = not args.no_recursive

    total_candidates = 0
    symlink_count = 0
    replaced_count = 0
    skipped_count = 0
    error_count = 0

    for path in sorted(iter_candidates(image_dir, recursive=recursive)):
        if not path.is_symlink() and not path.is_file():
            continue
        if path.suffix.lower() not in extensions:
            continue

        total_candidates += 1
        if not path.is_symlink():
            skipped_count += 1
            continue

        symlink_count += 1
        try:
            target_path = materialize_symlink(path, dry_run=args.dry_run)
            replaced_count += 1
            if args.verbose or args.dry_run:
                action = "would replace" if args.dry_run else "replaced"
                print(f"{action}: {path} -> {target_path}")
        except Exception as exc:  # noqa: BLE001
            error_count += 1
            print(f"error: {path}: {exc}", file=sys.stderr)

    mode = "dry-run" if args.dry_run else "done"
    print(
        f"{mode}: candidates={total_candidates}, symlinks={symlink_count}, "
        f"replaced={replaced_count}, skipped_non_symlink={skipped_count}, errors={error_count}"
    )

    if error_count:
        sys.exit(1)


if __name__ == "__main__":
    main()
