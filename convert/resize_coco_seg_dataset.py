#!/usr/bin/env python3
"""Resize a COCO segmentation dataset to a fixed square resolution.

This replicates RF-DETR's preprocessing (torchvision.transforms.functional.resize
with an explicit (H, W) tuple), so bounding boxes and segmentation polygons are
scaled independently along x/y axes. Images plus annotations for train/valid/test
splits are copied into a new dataset root with updated metadata.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

from PIL import Image
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resize COCO segmentation dataset (images + annotations)")
    parser.add_argument("--input-root", required=True,
                        help="原始 COCO segmentation 根目录，需包含 train/valid/test 子目录")
    parser.add_argument("--output-root", required=True,
                        help="输出目录，不存在会自动创建")
    parser.add_argument("--resolution", type=int, default=432,
                        help="目标分辨率（宽=高），默认432，与RF-DETR Seg Preview一致")
    parser.add_argument("--subsets", nargs="*", default=["train", "valid", "test"],
                        help="需要处理的子集，默认 train/valid/test")
    return parser.parse_args()


def load_coco_annotations(json_path: Path) -> Dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_annotation_index(annotations: Iterable[Dict]) -> Dict[int, List[Dict]]:
    by_image: Dict[int, List[Dict]] = defaultdict(list)
    for ann in annotations:
        by_image[ann["image_id"]].append(ann)
    return by_image


def resize_image(image_path: Path, output_path: Path, resolution: int) -> tuple[int, int]:
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size
    resized = image.resize((resolution, resolution), Image.BILINEAR)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resized.save(output_path)
    return orig_w, orig_h


def scale_bbox(bbox: List[float], scale_x: float, scale_y: float) -> List[float]:
    x, y, w, h = bbox
    return [x * scale_x, y * scale_y, w * scale_x, h * scale_y]


def scale_segmentation(segmentation: List[List[float]], scale_x: float, scale_y: float) -> List[List[float]]:
    scaled = []
    for poly in segmentation:
        new_poly: List[float] = []
        for i in range(0, len(poly), 2):
            new_poly.append(poly[i] * scale_x)
            new_poly.append(poly[i + 1] * scale_y)
        scaled.append(new_poly)
    return scaled


def process_subset(subset: str, src_root: Path, dst_root: Path, resolution: int) -> None:
    src_dir = src_root / subset
    if not src_dir.exists():
        print(f"[SKIP] {subset} 不存在: {src_dir}")
        return
    json_path = src_dir / "_annotations.coco.json"
    if not json_path.exists():
        print(f"[SKIP] {subset} 缺少 _annotations.coco.json")
        return

    dst_dir = dst_root / subset
    dst_dir.mkdir(parents=True, exist_ok=True)

    data = load_coco_annotations(json_path)
    annotations_by_image = build_annotation_index(data.get("annotations", []))

    for image_info in tqdm(data.get("images", []), desc=f"{subset}: resizing", ncols=90):
        img_rel = Path(image_info["file_name"])
        src_img = src_dir / img_rel
        dst_img = dst_dir / img_rel
        orig_w, orig_h = resize_image(src_img, dst_img, resolution)

        scale_x = resolution / orig_w
        scale_y = resolution / orig_h

        image_info["width"] = resolution
        image_info["height"] = resolution

        for ann in annotations_by_image.get(image_info["id"], []):
            if "bbox" in ann:
                ann["bbox"] = scale_bbox(ann["bbox"], scale_x, scale_y)
            if "segmentation" in ann and isinstance(ann["segmentation"], list):
                ann["segmentation"] = scale_segmentation(ann["segmentation"], scale_x, scale_y)
            if "area" in ann:
                ann["area"] = ann["area"] * scale_x * scale_y

    output_json = dst_dir / "_annotations.coco.json"
    output_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] {subset} → {output_json}")


def main() -> None:
    args = parse_args()
    src_root = Path(args.input_root).expanduser().resolve()
    dst_root = Path(args.output_root).expanduser().resolve()
    dst_root.mkdir(parents=True, exist_ok=True)

    for subset in args.subsets:
        process_subset(subset, src_root, dst_root, args.resolution)


if __name__ == "__main__":
    main()
