#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visualize CLAHE effect inside enhance_image_windowing."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils.image_processing import (
    enhance_image_windowing,
    auto_window_level,
    apply_window_level,
)


def _ensure_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def enhance_without_clahe(image: np.ndarray, output_bits: int) -> np.ndarray:
    gray = _ensure_gray(image)
    img_float = gray.astype(np.float32, copy=False)
    window_width, window_level = auto_window_level(img_float)
    return apply_window_level(img_float, window_width, window_level, output_bits)


def normalize_for_display(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    max_val = np.iinfo(image.dtype).max
    scaled = (image.astype(np.float32) / max_val) * 255.0
    return scaled.clip(0, 255).astype(np.uint8)


def plot_comparison(original: np.ndarray,
                    window_only: np.ndarray,
                    with_clahe: np.ndarray,
                    save_path: Path) -> None:
    orig_disp = normalize_for_display(original)
    win_disp = normalize_for_display(window_only)
    clahe_disp = normalize_for_display(with_clahe)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ["Original", "Windowing Only", "Windowing + CLAHE"]
    images = [orig_disp, win_disp, clahe_disp]

    for ax, title, img in zip(axes, titles, images):
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="检查enhance_image_windowing中的CLAHE增强效果"
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=Path(__file__).resolve().parent / "image.png",
        help="待测试的图像路径 (默认: scripts/image.png)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "windowing_clahe_comparison.png",
        help="比较图输出路径",
    )
    parser.add_argument(
        "--output-bits",
        type=int,
        default=8,
        choices=[8, 16],
        help="增强函数输出位深",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"未找到图像: {args.image}")

    image = cv2.imread(str(args.image), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"无法读取图像: {args.image}")

    window_only = enhance_without_clahe(image, args.output_bits)
    with_clahe = enhance_image_windowing(image, output_bits=args.output_bits)
    diff = cv2.absdiff(normalize_for_display(with_clahe),
                       normalize_for_display(window_only))

    plot_comparison(image, window_only, with_clahe, args.output)

    print("处理完成 ✅")
    print(f"原图: {args.image}")
    print(f"比较图已保存到: {args.output}")
    print(
        "像素差异统计 (显示映射后): min={:.2f} max={:.2f} mean={:.2f}".format(
            float(diff.min()), float(diff.max()), float(diff.mean())
        )
    )


if __name__ == "__main__":
    main()
