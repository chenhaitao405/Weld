#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare a preprocess-ready dataset layout for run_data_pipeline (labelme2yolo).

It builds:
  <output_root>/images/<dataset_name>/<group>/*.<ext>
  <output_root>/labels/<dataset_name>/<group>/*.json

Group names are derived from each JSON's relative directory under label_root,
with any "label"/"labels" path segments removed and path separators flattened
into "__" so each group is a single directory level (required by the pipeline).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

try:
    from utils.constants import IMAGE_EXTENSIONS
except Exception as exc:  # pragma: no cover - fail fast when utils missing
    raise SystemExit(f"Failed to import utils.constants: {exc}") from exc

IGNORE_JSON_NAMES = {
    "val_manifest.json",
    "train_manifest.json",
    "manifest.json",
}


def iter_jsons(label_root: Path) -> Iterable[Path]:
    for path in label_root.rglob("*.json"):
        if not path.is_file():
            continue
        if path.name in IGNORE_JSON_NAMES or path.name.endswith("_manifest.json"):
            continue
        yield path


def strip_label_parts(parts: List[str]) -> List[str]:
    cleaned: List[str] = []
    for part in parts:
        if part.lower() in {"label", "labels"}:
            continue
        cleaned.append(part)
    return cleaned


def infer_dataset_name(label_root: Path) -> str:
    name = label_root.name
    if name.lower() in {"label", "labels"}:
        return label_root.parent.name or "dataset"
    return name or "dataset"


def compute_rel_no_label(json_path: Path, label_root: Path) -> Path:
    rel = json_path.relative_to(label_root)
    parts = strip_label_parts(list(rel.parts))
    if not parts:
        return Path(json_path.name)
    return Path(*parts)


def group_name_for(rel_no_label: Path) -> str:
    rel_dir = rel_no_label.parent
    if not rel_dir.parts:
        return "root"
    return "__".join(rel_dir.parts)


def find_image_for(
    json_path: Path,
    image_root: Path,
    rel_no_label: Path,
    search_all: bool,
) -> Optional[Path]:
    stem = json_path.stem
    rel_dir = rel_no_label.parent

    for ext in IMAGE_EXTENSIONS:
        candidate = image_root / rel_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate

    if not search_all:
        return None

    for ext in IMAGE_EXTENSIONS:
        for candidate in image_root.rglob(f"{stem}{ext}"):
            if candidate.is_file():
                return candidate

    return None


def link_or_copy(src: Path, dst: Path, mode: str, overwrite: bool) -> bool:
    if dst.exists() or dst.is_symlink():
        if overwrite:
            dst.unlink()
        else:
            return False

    if mode == "symlink":
        os.symlink(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unsupported link mode: {mode}")
    return True


def ensure_dir(path: Path, dry_run: bool) -> None:
    if dry_run:
        return
    path.mkdir(parents=True, exist_ok=True)


def prepare_dataset(args: argparse.Namespace) -> int:
    label_root = Path(args.label_root).resolve()
    image_root = Path(args.image_root).resolve()

    if not label_root.exists():
        raise SystemExit(f"label_root not found: {label_root}")
    if not image_root.exists():
        raise SystemExit(f"image_root not found: {image_root}")

    dataset_name = args.dataset_name or infer_dataset_name(label_root)

    output_root = Path(args.output_root).resolve()
    output_images_root = output_root / "images" / dataset_name
    output_labels_root = output_root / "labels" / dataset_name

    ensure_dir(output_images_root, args.dry_run)
    ensure_dir(output_labels_root, args.dry_run)

    total_jsons = 0
    linked_jsons = 0
    linked_images = 0
    missing_images = 0
    skipped_existing = 0

    group_stats: Dict[str, Dict[str, int]] = {}

    for json_path in iter_jsons(label_root):
        total_jsons += 1
        rel_no_label = compute_rel_no_label(json_path, label_root)
        group_name = group_name_for(rel_no_label)

        group_stats.setdefault(group_name, {"json": 0, "images": 0, "missing": 0})

        img_path = find_image_for(
            json_path,
            image_root,
            rel_no_label,
            search_all=args.search_all,
        )

        if img_path is None:
            missing_images += 1
            group_stats[group_name]["missing"] += 1
            if not args.keep_missing:
                continue

        out_label_dir = output_labels_root / group_name
        out_image_dir = output_images_root / group_name
        ensure_dir(out_label_dir, args.dry_run)
        ensure_dir(out_image_dir, args.dry_run)

        dst_json = out_label_dir / json_path.name
        if not args.dry_run:
            if link_or_copy(json_path, dst_json, args.link_mode, args.overwrite):
                linked_jsons += 1
            else:
                skipped_existing += 1
        else:
            linked_jsons += 1

        group_stats[group_name]["json"] += 1

        if img_path is None:
            continue

        dst_img = out_image_dir / img_path.name
        if not args.dry_run:
            if link_or_copy(img_path, dst_img, args.link_mode, args.overwrite):
                linked_images += 1
            else:
                skipped_existing += 1
        else:
            linked_images += 1

        group_stats[group_name]["images"] += 1

    print("\nDone.")
    print(f"  label_root: {label_root}")
    print(f"  image_root: {image_root}")
    print(f"  output images: {output_images_root}")
    print(f"  output labels: {output_labels_root}")
    print(f"  dataset_name: {dataset_name}")
    print(f"  json files scanned: {total_jsons}")
    print(f"  json linked/copied: {linked_jsons}")
    print(f"  images linked/copied: {linked_images}")
    print(f"  missing images: {missing_images}")
    if skipped_existing:
        print(f"  skipped existing files: {skipped_existing}")

    if group_stats:
        print("\nGroup summary:")
        for group, stats in sorted(group_stats.items()):
            print(
                f"  - {group}: json={stats['json']}, "
                f"images={stats['images']}, missing={stats['missing']}"
            )

    print("\nSuggested params.yaml entry:")
    print("  preprocess:")
    print("    dataset_pairs:")
    print(f"    - image_root: {output_images_root}")
    print(f"      label_root: {output_labels_root}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare preprocess-ready LabelMe input layout.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image-root", required=True, help="Root directory for images")
    parser.add_argument("--label-root", required=True, help="Root directory for LabelMe JSONs")
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "datasets"),
        help="Output root to create images/ and labels/ folders",
    )
    parser.add_argument("--dataset-name", default=None, help="Name under output images/labels")
    parser.add_argument(
        "--link-mode",
        choices=["symlink", "hardlink", "copy"],
        default="symlink",
        help="How to place files into output (symlink is fast and space-saving)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in output dirs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and report only; do not create files",
    )
    parser.add_argument(
        "--search-all",
        action="store_true",
        help="If a matching image isn't found in the relative folder, search all image_root",
    )
    parser.add_argument(
        "--keep-missing",
        action="store_true",
        help="Keep JSONs even when image is missing (labelme2yolo may warn later)",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return prepare_dataset(args)


if __name__ == "__main__":
    raise SystemExit(main())
