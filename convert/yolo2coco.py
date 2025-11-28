#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YOLO → COCO 标注转换脚本

用法示例：
  python convert/yolo2coco.py --input_dir /path/to/yolo_dataset --output_dir /path/to/coco_dataset
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Set

from tqdm import tqdm

try:
    from PIL import Image
except ImportError:
    Image = None

# 允许脚本直接调用项目根目录下的 utils
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.dataset_management import read_dataset_yaml, find_image_files  # noqa: E402
from utils.label_processing import read_yolo_labels  # noqa: E402


SPLIT_NAME_MAP = {
    "train": "train",
    "training": "train",
    "val": "valid",
    "valid": "valid",
    "validation": "valid",
    "test": "test"
}


def _scan_class_ids(labels_dir: Path) -> List[int]:
    class_ids: Set[int] = set()
    if not labels_dir.exists():
        return []

    for label_file in labels_dir.rglob("*.txt"):
        labels = read_yolo_labels(str(label_file), mode='det')
        for label in labels:
            if label:
                class_ids.add(int(label[0]))

    return sorted(class_ids)


def _names_from_label_id_map(label_id_map: Dict) -> List[str]:
    if not isinstance(label_id_map, dict):
        return []

    filtered = [
        (int(idx), str(name))
        for name, idx in label_id_map.items()
        if isinstance(idx, int) and idx >= 0
    ]
    if not filtered:
        return []

    max_idx = max(idx for idx, _ in filtered)
    names = [f"class_{i}" for i in range(max_idx + 1)]
    for idx, name in filtered:
        if 0 <= idx < len(names):
            names[idx] = name
    return names


def _normalize_names(raw_names) -> List[str]:
    if isinstance(raw_names, list):
        return [str(name) for name in raw_names]

    if isinstance(raw_names, dict):
        normalized: Dict[int, str] = {}
        for key, value in raw_names.items():
            try:
                normalized[int(key)] = str(value)
            except (TypeError, ValueError):
                try:
                    normalized[int(value)] = str(key)
                except (TypeError, ValueError):
                    continue

        if normalized:
            max_idx = max(normalized.keys())
            names = [f"class_{i}" for i in range(max_idx + 1)]
            for idx, name in normalized.items():
                if 0 <= idx < len(names):
                    names[idx] = name
            return names

    if isinstance(raw_names, str):
        cleaned = raw_names.strip().strip("[](){}")
        if cleaned:
            return [
                segment.strip().strip("'\"")
                for segment in cleaned.split(",")
                if segment.strip()
            ]
    return []


def _load_class_names_for_coco(dataset_dir: Path, labels_dir: Path) -> List[str]:
    dataset_yaml_path = dataset_dir / "dataset.yaml"
    names: List[str] = []

    if dataset_yaml_path.exists():
        try:
            dataset_cfg = read_dataset_yaml(str(dataset_yaml_path)) or {}
        except Exception as exc:  # pragma: no cover - runtime warning only
            print(f"⚠️ 读取 dataset.yaml 失败：{exc}")
        else:
            names = _normalize_names(dataset_cfg.get("names"))
            if not names:
                names = _names_from_label_id_map(dataset_cfg.get("label_id_map"))

    if not names:
        class_ids = _scan_class_ids(labels_dir)
        if class_ids:
            max_idx = max(class_ids)
            names = [f"class_{i}" for i in range(max_idx + 1)]

    return names


def _read_image_size(image_path: Path) -> Tuple[int, int]:
    if Image is None:
        raise ImportError("Pillow 未安装，无法读取图像尺寸。请先运行 pip install pillow")

    with Image.open(image_path) as img:
        width, height = img.size
    return int(width), int(height)


def _yolo_bbox_to_coco(label: List[float], img_width: int, img_height: int) -> Tuple[List[float], float]:
    if len(label) < 5:
        return [], 0.0

    _, x_center, y_center, width, height = label[:5]
    bbox_width = max(0.0, width) * img_width
    bbox_height = max(0.0, height) * img_height

    x_min = x_center * img_width - bbox_width / 2
    y_min = y_center * img_height - bbox_height / 2

    x_min = max(0.0, min(x_min, img_width))
    y_min = max(0.0, min(y_min, img_height))
    x_max = max(x_min, min(img_width, x_min + bbox_width))
    y_max = max(y_min, min(img_height, y_min + bbox_height))

    bbox_width = max(0.0, x_max - x_min)
    bbox_height = max(0.0, y_max - y_min)

    if bbox_width <= 0 or bbox_height <= 0:
        return [], 0.0

    bbox = [
        round(x_min, 6),
        round(y_min, 6),
        round(bbox_width, 6),
        round(bbox_height, 6)
    ]
    area = round(bbox_width * bbox_height, 6)
    return bbox, area


def _convert_split_to_coco(split_name: str,
                           split_images_dir: Path,
                           split_labels_dir: Path,
                           target_split_dir: Path) -> Tuple[List[Dict], List[Dict]]:
    image_files = find_image_files(str(split_images_dir))
    coco_images: List[Dict] = []
    coco_annotations: List[Dict] = []
    image_id = 1
    annotation_id = 1

    if target_split_dir.exists():
        shutil.rmtree(target_split_dir)
    target_split_dir.mkdir(parents=True, exist_ok=True)

    labels_available = split_labels_dir.exists()
    if not labels_available:
        print(f"⚠️ 警告：未找到 {split_name} 标签目录 {split_labels_dir}，将创建空标注")

    for image_path in tqdm(image_files, desc=f"YOLO→COCO {split_name}"):
        width, height = _read_image_size(image_path)
        shutil.copy2(image_path, target_split_dir / image_path.name)

        coco_images.append({
            "id": image_id,
            "file_name": image_path.name,
            "width": width,
            "height": height
        })

        label_file = split_labels_dir / f"{image_path.stem}.txt"
        labels = read_yolo_labels(str(label_file), mode='det') if labels_available and label_file.exists() else []

        for label in labels:
            bbox, area = _yolo_bbox_to_coco(label, width, height)
            if not bbox:
                continue

            coco_annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": int(label[0]),
                "bbox": bbox,
                "area": area,
                "iscrowd": 0,
                "segmentation": []
            })
            annotation_id += 1

        image_id += 1

    return coco_images, coco_annotations


def convert_yolo_to_coco_dataset(input_dir: str, output_dir: str) -> Dict[str, Dict[str, int]]:
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"输入目录不存在：{input_dir}")

    images_root = input_path / "images"
    labels_root = input_path / "labels"

    if not images_root.exists():
        raise FileNotFoundError(f"未找到 YOLO images 目录：{images_root}")
    if not labels_root.exists():
        print(f"⚠️ 警告：未找到 YOLO labels 目录 {labels_root}，将以空标注形式导出")

    class_names = _load_class_names_for_coco(input_path, labels_root)
    if not class_names:
        raise ValueError("无法根据 dataset.yaml 或标签推断类别，请检查输入目录")

    categories = [
        {"id": idx, "name": name, "supercategory": "none"}
        for idx, name in enumerate(class_names)
    ]

    output_path = Path(output_dir)
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    split_dirs = sorted([d for d in images_root.iterdir() if d.is_dir()], key=lambda p: p.name.lower())
    if not split_dirs:
        raise ValueError(f"{images_root} 中未找到任何 split 目录")

    summary: Dict[str, Dict[str, int]] = {}

    for split_dir in split_dirs:
        raw_split = split_dir.name
        mapped_split = SPLIT_NAME_MAP.get(raw_split.lower(), raw_split.lower())
        target_split_dir = output_path / mapped_split

        coco_images, coco_annotations = _convert_split_to_coco(
            raw_split,
            split_dir,
            labels_root / raw_split,
            target_split_dir
        )

        annotation_data = {
            "info": {
                "description": "YOLO to COCO conversion",
                "version": "1.0",
                "year": datetime.now().year,
                "contributor": "convert/yolo2coco.py",
                "date_created": datetime.utcnow().isoformat() + "Z"
            },
            "licenses": [],
            "images": coco_images,
            "annotations": coco_annotations,
            "categories": categories
        }

        with open(target_split_dir / "_annotations.coco.json", 'w', encoding='utf-8') as f:
            json.dump(annotation_data, f, ensure_ascii=True, indent=2)

        summary[mapped_split] = {
            "images": len(coco_images),
            "annotations": len(coco_annotations)
        }

    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="将 YOLO 检测标注转换为 COCO 标注格式",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input_dir", required=True, help="YOLO 数据集目录（包含 images/ 与 labels/）")
    parser.add_argument("--output_dir", required=True, help="输出的 COCO 数据集目录")
    return parser.parse_args()


def main():
    args = parse_args()
    summary = convert_yolo_to_coco_dataset(args.input_dir, args.output_dir)

    print("\n✅ YOLO→COCO 转换完成!")
    for split, stats in summary.items():
        print(f"  - {split}: {stats['images']} 张图像, {stats['annotations']} 条标注")
    print(f"📁 输出目录：{args.output_dir}")


if __name__ == "__main__":
    main()
