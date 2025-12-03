#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
随机抽取YOLO数据集样本并可视化，支持检测与分割标注。
文本渲染复用 utils.pipeline_utils 中的 FontRenderer，以解决中文乱码问题。
"""

import argparse
import os
import random
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml

from utils.label_processing import denormalize_bbox, read_yolo_labels
from utils.pipeline_utils import FontRenderer, draw_detection_instance, load_image


def load_dataset_config(data_dir: str) -> Optional[List[str]]:
    """
    从 dataset.yaml 加载类别名称
    """
    yaml_path = os.path.join(data_dir, 'dataset.yaml')
    if not os.path.exists(yaml_path):
        print(f"Warning: dataset.yaml not found at {yaml_path}")
        return None

    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        names = config.get('names')
        if isinstance(names, dict):
            # YOLO可能以dict格式存储
            names = [names[k] for k in sorted(names)]
        return list(names) if names else None
    except Exception as exc:  # pragma: no cover - 配置错误时直接提示
        print(f"Error loading dataset.yaml: {exc}")
        return None


def normalize_for_display(image: np.ndarray) -> np.ndarray:
    """
    将任意位深的BGR图像缩放到uint8，以便可视化
    """
    if image.dtype == np.uint8:
        return image

    img_float = image.astype(np.float32)
    min_val = float(img_float.min())
    max_val = float(img_float.max())
    if max_val <= min_val:
        return np.zeros_like(image, dtype=np.uint8)

    scaled = (img_float - min_val) / (max_val - min_val)
    scaled = np.clip(scaled * 255.0, 0, 255).astype(np.uint8)
    return scaled


def resolve_class_name(class_id: int, class_names: Optional[Sequence[str]]) -> str:
    if class_names and 0 <= class_id < len(class_names):
        return str(class_names[class_id])
    return f"Class {class_id}"


def yolo_segmentation_to_polygon(label: List[float],
                                 img_width: int,
                                 img_height: int) -> List[List[float]]:
    """
    将YOLO分割标签转换为像素坐标多边形
    """
    coords = label[1:]
    polygon: List[List[float]] = []
    for idx in range(0, len(coords), 2):
        if idx + 1 >= len(coords):
            break
        px = float(coords[idx]) * img_width
        py = float(coords[idx + 1]) * img_height
        polygon.append([px, py])
    return polygon


def polygon_to_bbox(polygon: List[List[float]]) -> List[float]:
    xs = [pt[0] for pt in polygon]
    ys = [pt[1] for pt in polygon]
    return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]


def render_image_with_labels(image_path: str,
                             labels: List[List[float]],
                             format_type: str,
                             class_names: Optional[Sequence[str]],
                             font_renderer: Optional[FontRenderer]) -> np.ndarray:
    """
    使用 pipeline_utils 的 draw_detection_instance 渲染检测/分割结果
    """
    canvas = load_image(Path(image_path))
    canvas = normalize_for_display(canvas)
    img_height, img_width = canvas.shape[:2]

    for label in labels:
        if not label:
            continue

        class_id = int(label[0])
        class_name = resolve_class_name(class_id, class_names)

        if format_type == 'det':
            if len(label) < 5:
                continue
            x1, y1, x2, y2 = denormalize_bbox(
                label[1], label[2], label[3], label[4],
                img_width, img_height
            )
            bbox = [float(x1), float(y1), float(x2), float(y2)]
            draw_detection_instance(
                canvas,
                bbox=bbox,
                label=class_name,
                score=None,
                class_id=class_id,
                font_renderer=font_renderer
            )
        elif format_type == 'seg':
            if len(label) < 7:
                continue
            polygon = yolo_segmentation_to_polygon(label, img_width, img_height)
            if not polygon:
                continue
            bbox = polygon_to_bbox(polygon)
            draw_detection_instance(
                canvas,
                bbox=bbox,
                label=class_name,
                score=None,
                class_id=class_id,
                font_renderer=font_renderer,
                polygon=polygon
            )
        else:
            raise ValueError(f"Unsupported format type: {format_type}")

    return canvas


def display_image(image: np.ndarray, title: str):
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def main(data_dir: str,
         format_type: str = 'seg',
         font_renderer: Optional[FontRenderer] = None):
    print(f"Visualizing dataset in {format_type.upper()} format")
    print(f"Dataset directory: {data_dir}")

    class_names = load_dataset_config(data_dir)
    if class_names:
        print(f"Loaded {len(class_names)} classes: {', '.join(map(str, class_names))}")
    else:
        print("Warning: Could not load class names from dataset.yaml")

    splits = ['train', 'valid', 'val']
    available_splits = [
        split for split in splits
        if os.path.exists(os.path.join(data_dir, 'images', split))
    ]

    if not available_splits:
        print(f"No valid splits found in {data_dir}/images/")
        print(f"Expected one of: {', '.join(splits)}")
        return

    images_root = Path(data_dir) / 'images'
    labels_root = Path(data_dir) / 'labels'
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

    while True:
        split = random.choice(available_splits)
        print(f"\nSelected split: {split}")

        images_dir = images_root / split
        labels_dir = labels_root / split
        images = [f for f in os.listdir(images_dir) if f.lower().endswith(supported_extensions)]

        if not images:
            print(f"No images found in {images_dir}")
            return

        img_file = random.choice(images)
        img_path = images_dir / img_file
        label_path = labels_dir / f"{Path(img_file).stem}.txt"

        if not label_path.exists():
            print(f"Label file for {img_file} not found. Skipping.")
            continue

        print(f"Displaying: {img_file}")

        raw_image = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if raw_image is not None:
            print(f"  Image shape: {raw_image.shape}")
            print(f"  Data type: {raw_image.dtype}")
            print(f"  Value range: [{raw_image.min()}, {raw_image.max()}]")

        label_entries = read_yolo_labels(str(label_path), mode=format_type)
        if not label_entries:
            print("  Annotations: 0 objects")
        else:
            class_counts = {}
            for entry in label_entries:
                cls = int(entry[0]) if entry else -1
                cls_name = resolve_class_name(cls, class_names)
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

            print(f"  Annotations: {len(label_entries)} objects")
            for cls_name, count in class_counts.items():
                print(f"    - {cls_name}: {count}")

        try:
            canvas = render_image_with_labels(
                str(img_path),
                label_entries,
                format_type,
                class_names,
                font_renderer
            )
            title = f"{img_file} - {format_type.upper()} ({split})"
            display_image(canvas, title)
        except Exception as exc:
            print(f"Error visualizing {img_file}: {exc}")
            continue

        user_input = input("\nPress Enter to continue (or 'q' to quit): ")
        if user_input.strip().lower() == 'q':
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Visualize YOLO dataset annotations with proper font rendering')
    parser.add_argument('--data_dir', type=str,
                        default='/home/lenovo/code/CHT/detect/dataprocess/preprocessed_data2/test/SWRDsize112',
                        help='Path to the dataset directory containing dataset.yaml')
    parser.add_argument('--format', type=str, choices=['seg', 'det'], default='seg',
                        help='Format type: seg (segmentation) or det (detection)')
    parser.add_argument('--font-path', type=str, default=None,
                        help='Optional custom font file path (for Chinese characters)')
    parser.add_argument('--font-size', type=int, default=20,
                        help='Font size for labels')

    args = parser.parse_args()
    font_renderer = FontRenderer(font_path=args.font_path, font_size=args.font_size)
    main(args.data_dir, args.format, font_renderer)
