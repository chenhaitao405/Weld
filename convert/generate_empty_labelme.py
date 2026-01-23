#!/usr/bin/env python
"""
Generate empty LabelMe JSON annotations for images.

- Recursively scans an image root for supported image files.
- Writes empty LabelMe JSON files into a single output directory.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List

import cv2

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from utils import save_labelme_json  # noqa: E402
from utils.constants import IMAGE_EXTENSIONS  # noqa: E402

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def _iter_image_files(root: Path, exts: List[str]) -> List[Path]:
    files: List[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in exts:
            files.append(path)
    return sorted(files)


def _build_empty_labelme(image_path: Path, image_root: Path, version: str) -> dict:
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError("Failed to read image")
    height, width = img.shape[:2]
    rel_path = os.path.relpath(image_path, image_root)
    return {
        "version": version,
        "flags": {},
        "shapes": [],
        "imagePath": rel_path,
        "imageData": None,
        "imageHeight": int(height),
        "imageWidth": int(width),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate empty LabelMe JSON files for images."
    )
    parser.add_argument(
        "--image-dir",
        required=True,
        help="Root directory containing images (scans recursively).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for generated LabelMe JSON files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing JSON files.",
    )
    parser.add_argument(
        "--version",
        default="5.9.1",
        help="LabelMe JSON version field (default: 5.9.1).",
    )
    args = parser.parse_args()

    image_root = Path(args.image_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not image_root.exists():
        raise FileNotFoundError(f"Image dir not found: {image_root}")

    output_dir.mkdir(parents=True, exist_ok=True)

    exts = [ext.lower() for ext in IMAGE_EXTENSIONS]
    image_files = _iter_image_files(image_root, exts)
    if not image_files:
        print(f"No images found under {image_root}")
        return

    iterable = tqdm(image_files, desc="Generating", unit="file") if tqdm else image_files
    created = 0
    skipped = 0
    failed = 0

    for image_path in iterable:
        json_name = f"{image_path.stem}.json"
        json_path = output_dir / json_name
        if json_path.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            data = _build_empty_labelme(image_path, image_root, args.version)
            save_labelme_json(data, str(json_path))
            created += 1
        except Exception as exc:
            failed += 1
            print(f"[WARN] {image_path}: {exc}")

    print(f"Created: {created}, Skipped: {skipped}, Failed: {failed}")


if __name__ == "__main__":
    main()
