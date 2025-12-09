#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将两个 COCO 数据集（train/valid/test 等 split）合并成一个新的数据集。

特性：
  * 针对每个 split（train/valid/test ...）分别合并 annotations；
  * 自动重排 image_id / annotation_id，保持唯一；
  * 依据类别名称合并 categories，并统一 category_id；
  * 图像默认以软链接方式指向原始数据，避免重复拷贝；
  * 若存在重名图像，可通过 --prefix-a / --prefix-b 或自动追加序号避免冲突；

用法示例：
  python convert/merge_coco.py \\
      --dataset-a /path/to/coco_a \\
      --dataset-b /path/to/coco_b \\
      --output-dir /path/to/merged_coco
"""

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm


ANNOTATION_FILENAME = "_annotations.coco.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="合并两个 COCO 数据集（逐 split 合并，图像软链接，标注写入新 JSON）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--dataset-a", required=True, help="第一个 COCO 数据集根目录")
    parser.add_argument("--dataset-b", required=True, help="第二个 COCO 数据集根目录")
    parser.add_argument("--output-dir", required=True, help="合并后的输出目录")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid", "test"],
        help="需要合并的 split 名称列表"
    )
    parser.add_argument(
        "--prefix-a",
        default=None,
        help="为数据集 A 图像添加的名称前缀（默认取 dataset-a 目录名）"
    )
    parser.add_argument(
        "--prefix-b",
        default=None,
        help="为数据集 B 图像添加的名称前缀（默认取 dataset-b 目录名）"
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="若指定则复制图像文件；默认创建软链接"
    )
    return parser.parse_args()


def _sanitize_prefix(prefix: Optional[str]) -> str:
    if prefix is None:
        return ""
    cleaned = prefix.strip()
    cleaned = cleaned.replace(" ", "_").replace(os.sep, "_")
    return cleaned


def _load_split(dataset_root: Path,
                split_name: str,
                prefix: str) -> Optional[Dict]:
    split_dir = dataset_root / split_name
    if not split_dir.exists():
        return None

    annotation_path = split_dir / ANNOTATION_FILENAME
    if annotation_path.exists():
        with open(annotation_path, "r", encoding="utf-8") as f:
            annotation_data = json.load(f)
    else:
        print(f"⚠️ 警告：{split_dir} 缺少 {ANNOTATION_FILENAME}，将创建空标注")
        annotation_data = {
            "images": [],
            "annotations": [],
            "categories": [],
            "licenses": []
        }

    return {
        "dataset_name": dataset_root.name,
        "prefix": prefix,
        "split_dir": split_dir,
        "data": annotation_data
    }


def _link_or_copy_image(source: Path, target: Path, copy_images: bool):
    if target.exists() or target.is_symlink():
        target.unlink()

    if copy_images:
        shutil.copy2(source, target)
        return

    try:
        os.symlink(source.resolve(), target)
    except OSError:
        # 回退到复制，确保流程不中断
        shutil.copy2(source, target)


def _unique_filename(original_name: str,
                     prefix: str,
                     used_names: Dict[str, int]) -> str:
    base_name = original_name
    if prefix:
        base_name = f"{prefix}_{original_name}"

    if base_name not in used_names:
        used_names[base_name] = 0
        return base_name

    used_names[base_name] += 1
    counter = used_names[base_name]
    candidate = f"{prefix}_{counter}_{original_name}" if prefix else f"{counter}_{original_name}"
    while candidate in used_names:
        counter += 1
        candidate = f"{prefix}_{counter}_{original_name}" if prefix else f"{counter}_{original_name}"
    used_names[candidate] = 0
    return candidate


def _merge_licenses(entries: List[Dict]) -> List[Dict]:
    merged: List[Dict] = []
    seen = set()
    for entry in entries:
        for license_entry in entry["data"].get("licenses") or []:
            key = (
                license_entry.get("id"),
                license_entry.get("name"),
                license_entry.get("url")
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(license_entry)
    return merged


def _build_category_mappings(entries: List[Dict]) -> Tuple[List[Dict], List[Dict[int, int]]]:
    """
    返回：
      merged_categories: List[Dict]
      dataset_category_maps: List[Dict[old_id, new_id]]
    """
    name_to_id: Dict[str, int] = {}
    merged_categories: List[Dict] = []
    dataset_maps: List[Dict[int, int]] = []

    for entry in entries:
        mapping: Dict[int, int] = {}
        for category in entry["data"].get("categories", []):
            name = str(category.get("name") or f"category_{category.get('id', 0)}")
            supercategory = category.get("supercategory") or "none"
            if name not in name_to_id:
                new_id = len(name_to_id)
                name_to_id[name] = new_id
                merged_categories.append({
                    "id": new_id,
                    "name": name,
                    "supercategory": supercategory
                })
            mapping[int(category["id"])] = name_to_id[name]
        dataset_maps.append(mapping)

    merged_categories.sort(key=lambda c: c["id"])
    return merged_categories, dataset_maps


def _update_annotation_ids(entries: List[Dict],
                           category_maps: List[Dict[int, int]],
                           target_split_dir: Path,
                           copy_images: bool) -> Tuple[List[Dict], List[Dict]]:
    merged_images: List[Dict] = []
    merged_annotations: List[Dict] = []

    used_names: Dict[str, int] = {}
    next_image_id = 1
    next_annotation_id = 1

    for entry, category_map in zip(entries, category_maps):
        data = entry["data"]
        split_dir = entry["split_dir"]
        prefix = entry["prefix"]
        image_id_map: Dict[int, int] = {}

        for image in tqdm(data.get("images", []), desc=f"链接图像 {split_dir}", leave=False):
            original_name = image["file_name"]
            src_path = split_dir / original_name
            if not src_path.exists():
                alt_path = split_dir / "images" / original_name
                if alt_path.exists():
                    src_path = alt_path

            if not src_path.exists():
                print(f"⚠️ 警告：找不到图像 {src_path}，跳过该条记录")
                continue

            new_name = _unique_filename(original_name, prefix, used_names)
            target_path = target_split_dir / new_name
            _link_or_copy_image(src_path, target_path, copy_images)

            new_image = dict(image)
            new_image["id"] = next_image_id
            new_image["file_name"] = new_name

            merged_images.append(new_image)
            image_id_map[int(image["id"])] = next_image_id
            next_image_id += 1

        for annotation in data.get("annotations", []):
            old_image_id = int(annotation["image_id"])
            if old_image_id not in image_id_map:
                continue

            category_id = int(annotation.get("category_id"))
            if category_id not in category_map:
                raise KeyError(f"在 {split_dir} 中发现未知的 category_id={category_id}")

            new_annotation = dict(annotation)
            new_annotation["id"] = next_annotation_id
            new_annotation["image_id"] = image_id_map[old_image_id]
            new_annotation["category_id"] = category_map[category_id]

            merged_annotations.append(new_annotation)
            next_annotation_id += 1

    return merged_images, merged_annotations


def merge_split(entries: List[Dict],
                output_split_dir: Path,
                copy_images: bool) -> Dict[str, int]:
    if output_split_dir.exists():
        shutil.rmtree(output_split_dir)
    output_split_dir.mkdir(parents=True, exist_ok=True)

    merged_categories, category_maps = _build_category_mappings(entries)
    merged_images, merged_annotations = _update_annotation_ids(
        entries,
        category_maps,
        output_split_dir,
        copy_images
    )
    merged_licenses = _merge_licenses(entries)
    source_names = [entry["dataset_name"] for entry in entries]

    annotation_payload = {
        "info": {
            "description": f"Merged COCO dataset from {', '.join(source_names)}",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "convert/merge_coco.py",
            "date_created": datetime.utcnow().isoformat() + "Z",
            "source_datasets": source_names
        },
        "licenses": merged_licenses,
        "images": merged_images,
        "annotations": merged_annotations,
        "categories": merged_categories
    }

    with open(output_split_dir / ANNOTATION_FILENAME, "w", encoding="utf-8") as f:
        json.dump(annotation_payload, f, ensure_ascii=False, indent=2)

    return {
        "images": len(merged_images),
        "annotations": len(merged_annotations)
    }


def main():
    args = parse_args()

    dataset_a = Path(args.dataset_a).expanduser().resolve()
    dataset_b = Path(args.dataset_b).expanduser().resolve()
    if not dataset_a.exists():
        raise FileNotFoundError(f"dataset-a 不存在：{dataset_a}")
    if not dataset_b.exists():
        raise FileNotFoundError(f"dataset-b 不存在：{dataset_b}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    default_prefix_a = _sanitize_prefix(dataset_a.name)
    default_prefix_b = _sanitize_prefix(dataset_b.name)
    prefix_a = _sanitize_prefix(args.prefix_a) if args.prefix_a is not None else default_prefix_a
    prefix_b = _sanitize_prefix(args.prefix_b) if args.prefix_b is not None else default_prefix_b

    summary: Dict[str, Dict[str, int]] = {}
    for split_name in args.splits:
        entries: List[Dict] = []
        split_a = _load_split(dataset_a, split_name, prefix_a)
        split_b = _load_split(dataset_b, split_name, prefix_b)
        if split_a:
            entries.append(split_a)
        if split_b:
            entries.append(split_b)

        if not entries:
            print(f"⚠️ 未在两个数据集中找到 split '{split_name}'，跳过")
            continue

        stats = merge_split(entries, output_dir / split_name, args.copy_images)
        summary[split_name] = stats
        print(f"✅ 合并完成：{split_name} —— {stats['images']} 张图像 / {stats['annotations']} 条标注")

    if not summary:
        print("❌ 未生成任何 split，请确认输入目录结构是否正确")
        return

    print("\n📊 合并统计：")
    for split, stats in summary.items():
        print(f"  - {split}: {stats['images']} 张图像, {stats['annotations']} 条标注")


if __name__ == "__main__":
    main()
