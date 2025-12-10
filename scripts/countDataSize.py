#!/usr/bin/env python3
"""Utility to summarize image resolutions and aspect ratios in a dataset."""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

try:
    from PIL import Image  # type: ignore
except ImportError as exc:  # pragma: no cover - Pillow should be available but we keep a clear message
    raise SystemExit("Pillow is required: pip install pillow") from exc


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统计图像尺寸、长宽比分布")
    parser.add_argument(
        "image_dir",
        nargs="?",
        default="/datasets/PAR/Xray/self/1120/labeled/roi2_merge/patch_det_yolo",
        help="图像根目录（默认为焊缝 ROI patch 数据集路径）",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="打印出现频率最高的尺寸/长宽比个数",
    )
    parser.add_argument(
        "--export-csv",
        help="可选：输出 per-image 统计 CSV 路径",
    )
    return parser.parse_args()


def collect_images(root: Path) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"目录不存在: {root}")
    return [p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED_EXTS and p.is_file()]


def read_image_size(path: Path) -> Optional[Tuple[int, int]]:
    try:
        with Image.open(path) as img:
            width, height = img.size
        return int(width), int(height)
    except Exception as exc:
        print(f"无法读取 {path}: {exc}")
        return None


def summarize(values: Sequence[float]) -> Tuple[float, float, float, float, float]:
    if not values:
        return (math.nan,) * 5
    return (
        float(min(values)),
        float(max(values)),
        float(statistics.mean(values)),
        float(statistics.median(values)),
        float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
    )


def export_csv(rows: Iterable[Tuple[str, int, int, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image", "width", "height", "aspect_ratio(width/height)"])
        writer.writerows(rows)
    print(f"CSV 已保存: {output_path}")


def main() -> None:
    args = parse_args()
    root = Path(args.image_dir)
    image_paths = collect_images(root)
    if not image_paths:
        raise SystemExit(f"未在 {root} 下找到图像文件")

    dims_counter: Counter[Tuple[int, int]] = Counter()
    ratios_counter: Counter[float] = Counter()
    widths: List[int] = []
    heights: List[int] = []
    ratios: List[float] = []
    csv_rows: List[Tuple[str, int, int, float]] = []

    for path in image_paths:
        size = read_image_size(path)
        if size is None:
            continue
        width, height = size
        ratio = width / height if height else math.nan
        dims_counter[(width, height)] += 1
        ratios_counter[round(ratio, 3)] += 1
        widths.append(width)
        heights.append(height)
        ratios.append(ratio)
        if args.export_csv:
            csv_rows.append((str(path), width, height, ratio))

    if args.export_csv:
        export_csv(csv_rows, Path(args.export_csv))

    width_stats = summarize(widths)
    height_stats = summarize(heights)
    ratio_stats = summarize(ratios)

    print("\n=== 数据集规模 ===")
    print(f"总图像数: {len(image_paths)}")
    print(f"成功统计: {len(widths)}，失败: {len(image_paths) - len(widths)}")

    def _print_stats(title: str, stats: Tuple[float, float, float, float, float]) -> None:
        min_v, max_v, mean_v, median_v, std_v = stats
        print(f"\n{title}:")
        print(f"  最小: {min_v:.1f}")
        print(f"  最大: {max_v:.1f}")
        print(f"  均值: {mean_v:.1f}")
        print(f"  中位数: {median_v:.1f}")
        print(f"  标准差: {std_v:.1f}")

    _print_stats("宽度统计", width_stats)
    _print_stats("高度统计", height_stats)
    _print_stats("宽/高比统计", ratio_stats)

    def bucket_summary(values: Sequence[float], buckets: Sequence[Tuple[Optional[float], Optional[float]]], title: str) -> None:
        if not values:
            return
        counts = [0] * len(buckets)
        for value in values:
            for idx, (low, high) in enumerate(buckets):
                low_ok = True if low is None else value >= low
                high_ok = True if high is None else value < high
                if low_ok and high_ok:
                    counts[idx] += 1
                    break
        print(f"\n=== {title} ===")
        total = len(values)
        for (low, high), count in zip(buckets, counts):
            if high is None:
                label = f">= {low:g}" if low is not None else ">= 0"
            elif low is None:
                label = f"< {high:g}"
            else:
                label = f"[{low:g}, {high:g})"
            pct = count / total * 100 if total else 0
            print(f"  {label:>12}: {count:4d} ({pct:4.1f}%)")

    width_buckets = [(None, 640), (640, 1280), (1280, 2560), (2560, 5120), (5120, None)]
    height_buckets = [(None, 320), (320, 640), (640, 960), (960, 1280), (1280, None)]
    ratio_buckets = [(None, 1.2), (1.2, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, None)]

    bucket_summary(widths, width_buckets, "宽度分段统计")
    bucket_summary(heights, height_buckets, "高度分段统计")
    bucket_summary(ratios, ratio_buckets, "宽高比分段统计")

    print("\n=== 最常见的尺寸 (width x height) ===")
    for (w, h), count in dims_counter.most_common(args.top_k):
        pct = count / len(widths) * 100 if widths else 0
        print(f"  {w:5d} x {h:5d}: {count:5d} 张 ({pct:4.1f}%)")

    print("\n=== 最常见的宽高比 (四舍五入至 0.001) ===")
    for ratio_value, count in ratios_counter.most_common(args.top_k):
        pct = count / len(ratios) * 100 if ratios else 0
        print(f"  {ratio_value:7.3f}: {count:5d} 张 ({pct:4.1f}%)")

    tall = sum(1 for r in ratios if r < 1)
    squareish = sum(1 for r in ratios if 0.9 <= r <= 1.1)
    wide = sum(1 for r in ratios if r > 1)
    total = len(ratios)
    if total:
        print("\n=== 宽高方向占比 ===")
        print(f"  宽>高: {wide} ({wide / total * 100:.1f}%)")
        print(f"  近似方形(0.9-1.1): {squareish} ({squareish / total * 100:.1f}%)")
        print(f"  宽<高: {tall} ({tall / total * 100:.1f}%)")


if __name__ == "__main__":
    main()
