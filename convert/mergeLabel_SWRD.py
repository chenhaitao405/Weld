#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SWRD8bit 数据的 LabelMe 标签融合脚本。
参考 convert/mergeLabel.py 的映射关系，并新增对标签 “焊缝” 的过滤（直接删除该标注）。
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

# ========================= 路径配置 =========================
SWRD_ROOT = Path("/home/lenovo/code/CHT/datasets/Xray/opensource/SWRD8bit").resolve()
INPUT_JSON_ROOT = SWRD_ROOT / "crop_weld_jsons"
OUTPUT_JSON_ROOT = SWRD_ROOT / "crop_weld_jsons_merged"

# 过滤掉的标签
FILTER_LABELS = {"焊缝"}

# 标签映射关系（与 convert/mergeLabel.py 保持一致）
LABEL_MAPPING: Dict[str, str] = {
    # 裂纹类
    "裂纹": "裂纹",

    # 未熔合类
    "未熔合": "未熔合",
    "坡口未熔合": "未熔合",
    "层间未熔合": "未熔合",
    "接头未熔合": "未熔合",
    "根部未熔合": "未熔合",
    "未融合": "未熔合",

    # 未焊透类
    "未焊透": "未焊透",

    # 条形缺陷类
    "条形缺陷": "条形缺陷",
    "条状缺陷": "条形缺陷",

    # 圆形缺陷类
    "圆形缺陷": "圆形缺陷",
    "气孔": "圆形缺陷",
    "密孔": "圆形缺陷",

    # 咬边类
    "咬边": "咬边",
    "内咬边": "咬边",
    "外咬边": "咬边",

    # 内凹类
    "内凹": "内凹",

    # 其他类
    "条状缺陷，圆形缺陷": "其他",
    "未盖面": "其他",
    "未焊满": "其他",
    "凹陷": "其他",
    "漏焊": "其他",
    "焊瘤": "其他",
    "根部成形不良": "其他",
    "错口": "其他",
    "折口": "其他",
    "焊丝": "其他",
    "焊头": "其他",
    "伪缺线": "其他",
    "伪缺陷": "其他",
    "水渍": "其他",
    "划伤": "其他",
    "母材缺陷": "其他",
    "内腐蚀": "其他",
    "异物": "其他",
    "其他": "其他",
    "铁水": "其他"
}


def adjust_label(label: str) -> str:
    """根据映射关系调整标签，未命中映射则归为“其他”"""
    if label in LABEL_MAPPING:
        return LABEL_MAPPING[label]
    print(f"警告: 未知标签 '{label}'，将归类为'其他'")
    return "其他"


def process_json_file(input_path: Path, output_path: Path) -> Optional[Dict[str, int]]:
    """处理单个 LabelMe JSON 文件"""
    try:
        with input_path.open('r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        print(f"  ⚠️ 无法解析 {input_path}: {exc}")
        return None

    shapes = data.get('shapes', [])
    new_shapes: List[Dict] = []
    stats = {
        "original_total": 0,
        "filtered_weld": 0,
        "unknown_labels": 0
    }

    for shape in shapes:
        label = str(shape.get('label', '')).strip()
        if not label:
            continue

        stats["original_total"] += 1

        if label in FILTER_LABELS:
            stats["filtered_weld"] += 1
            continue

        adjusted_label = LABEL_MAPPING.get(label)
        if adjusted_label is None:
            stats["unknown_labels"] += 1
            adjusted_label = adjust_label(label)

        shape['label'] = adjusted_label
        new_shapes.append(shape)

    data['shapes'] = new_shapes
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return stats


def main():
    print("=" * 60)
    print("SWRD8bit LabelMe 标签融合工具 (含“焊缝”过滤)")
    print("=" * 60)

    if not INPUT_JSON_ROOT.exists():
        raise FileNotFoundError(f"未找到输入标签目录: {INPUT_JSON_ROOT}")

    json_files = list(INPUT_JSON_ROOT.rglob("*.json"))
    if not json_files:
        print("未在 crop_weld_jsons 中找到任何 JSON 文件")
        return

    print(f"发现 {len(json_files)} 个 JSON 文件，开始处理...")

    total_stats = {
        "files": 0,
        "filtered_weld": 0,
        "unknown_labels": 0,
        "original_total": 0
    }

    for json_file in json_files:
        relative = json_file.relative_to(INPUT_JSON_ROOT)
        output_file = OUTPUT_JSON_ROOT / relative
        stats = process_json_file(json_file, output_file)
        if not stats:
            continue

        total_stats["files"] += 1
        total_stats["filtered_weld"] += stats["filtered_weld"]
        total_stats["unknown_labels"] += stats["unknown_labels"]
        total_stats["original_total"] += stats["original_total"]

    print("\n处理完成!")
    print(f"  ✔ 处理文件数: {total_stats['files']}")
    print(f"  ✔ 原始标注数: {total_stats['original_total']}")
    print(f"  ✔ 过滤 '焊缝' 标注: {total_stats['filtered_weld']}")
    print(f"  ✔ 归入 '其他' 的未知标签数: {total_stats['unknown_labels']}")
    print(f"\n融合后的标签已保存至: {OUTPUT_JSON_ROOT}")


if __name__ == "__main__":
    main()
