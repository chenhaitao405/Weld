#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
临时脚本：将 crop_weld_data 数据集中的所有图像转换为 8 位 BMP，并拷贝到 SWRD8bit 目录。
"""

import argparse
import shutil
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

# 默认路径
DEFAULT_SOURCE = Path("/home/lenovo/code/CHT/datasets/Xray/opensource/crop_weld_data")
DEFAULT_TARGET = DEFAULT_SOURCE.parent / "SWRD8bit"

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 crop_weld_data 内所有图像转为 8 位 BMP，并复制 JSON 标注。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE,
                        help="原始 crop_weld_data 目录")
    parser.add_argument("--target-root", type=Path, default=DEFAULT_TARGET,
                        help="输出 SWRD8bit 目录")
    parser.add_argument("--overwrite", action="store_true",
                        help="目标存在时先删除再重建（慎用）")
    parser.add_argument("--clip-percentiles", type=float, nargs=2, metavar=("LOW", "HIGH"),
                        default=(1.0, 99.5),
                        help="16bit 图像转换为 8bit 时使用的分位点裁剪区间，设为 0 100 则使用全范围")
    return parser.parse_args()


def ensure_clean_dir(path: Path, overwrite: bool = False) -> None:
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def find_images(root: Path) -> List[Path]:
    files: List[Path] = []
    for ext in IMAGE_EXTENSIONS:
        files.extend(root.rglob(f"*{ext}"))
        files.extend(root.rglob(f"*{ext.upper()}"))
    return sorted(set(files))


def convert_to_8bit_bmp(src: Path, dst: Path,
                        clip_percentiles: Optional[Tuple[float, float]] = None) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        mode = img.mode
        # 16bit 单/多通道图像
        if mode in {"I;16", "I;16B", "I;16L", "I"}:
            array = np.array(img, dtype=np.float32)
            if array.ndim == 3:
                array = array.mean(axis=2)

            if clip_percentiles:
                low_p, high_p = clip_percentiles
                low = np.percentile(array, low_p)
                high = np.percentile(array, high_p)
            else:
                low = float(np.min(array))
                high = float(np.max(array))

            if high <= low:
                scaled = np.zeros_like(array, dtype=np.uint8)
            else:
                clipped = np.clip(array, low, high)
                scaled = ((clipped - low) / (high - low) * 255.0).astype(np.uint8)

            out_img = Image.fromarray(scaled, mode="L")
        else:
            out_img = img.convert("L")

        out_img.save(dst, format="BMP")


def copy_json_tree(src_json_root: Path, dst_json_root: Path) -> None:
    if not src_json_root.exists():
        print(f"⚠️ 未找到标注目录：{src_json_root}，跳过复制。")
        return
    if dst_json_root.exists():
        shutil.rmtree(dst_json_root)
    shutil.copytree(src_json_root, dst_json_root)


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    target_root = args.target_root.resolve()

    images_root = source_root / "crop_weld_images"
    json_root = source_root / "crop_weld_jsons"
    if not images_root.exists():
        raise FileNotFoundError(f"未找到图像目录：{images_root}")

    ensure_clean_dir(target_root, overwrite=args.overwrite)
    dst_images_root = target_root / "crop_weld_images"
    dst_json_root = target_root / "crop_weld_jsons"

    clip_percentiles: Optional[Tuple[float, float]] = None
    if args.clip_percentiles:
        low, high = args.clip_percentiles
        if low < 0 or high > 100 or low >= high:
            raise ValueError("clip-percentiles 需满足 0 <= low < high <= 100")
        clip_percentiles = (low, high)

    images = find_images(images_root)
    if not images:
        print("⚠️ 未在 crop_weld_images 中找到图像文件。")
    else:
        print(f"共找到 {len(images)} 张图像，开始转换为 8 位 BMP ...")
        for src_path in tqdm(images, desc="转换", unit="img"):
            rel_path = src_path.relative_to(images_root)
            dst_path = dst_images_root / rel_path.with_suffix(".bmp")
            convert_to_8bit_bmp(src_path, dst_path, clip_percentiles)
        print("✅ 图像转换完成。")

    copy_json_tree(json_root, dst_json_root)
    print(f"📁 新数据集已生成：{target_root}")


if __name__ == "__main__":
    main()
