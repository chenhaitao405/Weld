#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Validate SliceClassificationPipeline predictions against LabelMe annotations.

Steps:
1. Load datasets configured similarly to run_data_pipeline.py (BASE_PATH + JSON_BASE_PATH).
2. Run SliceClassificationPipeline on each image to obtain predicted defect patches.
3. Load the corresponding LabelMe JSON, derive bounding boxes for annotations.
4. Merge overlapping predicted defect patches mapped to the original image.
5. Compare merged predictions against ground-truth boxes by checking whether each label is sufficiently covered, then compute precision/recall.
"""

import argparse
import json
import platform
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from convert.pj.yolo_roi_extractor import WeldROIDetector
from utils.pipeline_utils import FontRenderer, load_image
from utils.slice_classification_pipeline import SliceClassificationPipeline, DEFAULT_DEFECT_CLASSES, DEFAULT_OVERLAP_RATIO, DEFAULT_WINDOW_SIZE, DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_ALPHA

# 路径配置（与 run_data_pipeline.py 保持一致）
if platform.system() == "Windows":
    DEFAULT_IMAGE_BASE = r"C:\Users\CHT\Desktop\datasets1117\labeled"
    DEFAULT_JSON_BASE = r"C:\Users\CHT\Desktop\datasets1117\adjust"
elif platform.system() == "Linux":
    DEFAULT_IMAGE_BASE = "/home/lenovo/code/CHT/datasets/Xray/self/1120/labeled"
    DEFAULT_JSON_BASE = "/home/lenovo/code/CHT/datasets/Xray/self/1120/adjust"
else:
    raise EnvironmentError(f"Unsupported OS: {platform.system()}, please update DEFAULT_IMAGE_BASE/DEFAULT_JSON_BASE.")

DEFAULT_DATASETS = [
    "D1",
    "D2",
    "D3",
    "D4",
    "img20250608",
    "img20250609",
    "val_sources"
]

SUPPORTED_IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="验证切片分类推理结果，与LabelMe标签计算覆盖率并统计精度/召回"
    )
    parser.add_argument("--cls-weights", required=True, help="切片分类模型权重路径 (SliceClassificationPipeline)")
    parser.add_argument("--roi-weights", help="可选：ROI检测模型权重路径")
    parser.add_argument("--roi-conf", type=float, default=0.25, help="ROI检测置信度阈值")
    parser.add_argument("--roi-iou", type=float, default=0.45, help="ROI检测IoU阈值")
    parser.add_argument("--roi-padding", type=float, default=0.1, help="ROI区域padding比例")

    parser.add_argument("--datasets", nargs='+', default=DEFAULT_DATASETS,
                        help="要验证的数据集列表（名称与BASE路径下目录一致）")
    parser.add_argument("--image-base", default=DEFAULT_IMAGE_BASE,
                        help="图像根目录（包含各数据集文件夹）")
    parser.add_argument("--label-base", default=DEFAULT_JSON_BASE,
                        help="LabelMe根目录（各数据集/label目录）")
    parser.add_argument("--label-overlap-threshold", "--iou-threshold",
                        dest="label_overlap_threshold", type=float, default=0.3,
                        help="标签覆盖阈值：预测结果与标签的重叠面积占标签面积的比例高于该值即视为命中（兼容旧参数 --iou-threshold）")
    parser.add_argument("--max-images", type=int,
                        help="每个数据集最多验证的图像数量")

    parser.add_argument("--window-size", type=int, nargs=2,
                        default=[DEFAULT_WINDOW_SIZE, DEFAULT_WINDOW_SIZE],
                        help="切片窗口大小 [height width]")
    parser.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP_RATIO,
                        help="滑动窗口重叠率 (0.0-1.0)")
    parser.add_argument("--enhance-mode", choices=["original", "windowing"],
                        default="windowing", help="图像增强模式")
    parser.add_argument("--confidence-threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD,
                        help="切片分类置信度阈值，用于判定缺陷")
    parser.add_argument("--defect-classes", type=int, nargs='+',
                        default=list(DEFAULT_DEFECT_CLASSES),
                        help="被视为缺陷的分类ID")
    parser.add_argument("--use-confidence-weight", action="store_true",
                        help="热力图累积使用置信度加权（对精度评估无影响）")
    parser.add_argument("--colormap", default="jet",
                        choices=['hot', 'jet', 'turbo', 'viridis', 'plasma',
                                 'coolwarm', 'RdYlBu', 'YlOrRd'],
                        help="热力图色图名称（仅用于内部pipeline，验证不输出热力图）")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                        help="热力图叠加透明度（内部pipeline参数）")
    parser.add_argument("--display-mode", default="overlay",
                        choices=['overlay', 'contour', 'sparse'],
                        help="热力图显示模式（内部pipeline参数）")
    parser.add_argument("--visualize", action="store_true",
                        help="是否保存可视化结果（标签+热力图+验证标签）")
    parser.add_argument("--visualize-dir", default="validation_visualizations",
                        help="可视化输出目录，默认写入 validation_visualizations")
    parser.add_argument("--visualize-no-colorbar", action="store_true",
                        help="生成图像时不添加颜色条")
    parser.add_argument("--font-path", help="可选：指定字体以正确显示中文标注")
    parser.add_argument("--font-size", type=int, default=20,
                        help="可视化中文字大小，默认20")

    parser.add_argument("--report-path", default="slice_cls_validation.json",
                        help="保存验证报告的路径（JSON）")
    return parser.parse_args()


def build_roi_detector(args: argparse.Namespace) -> Optional[WeldROIDetector]:
    if not args.roi_weights:
        return None
    print(f"加载ROI检测模型: {args.roi_weights}")
    return WeldROIDetector(
        model_path=args.roi_weights,
        roi_conf_threshold=args.roi_conf,
        roi_iou_threshold=args.roi_iou,
        padding_ratio=args.roi_padding
    )


def load_label_polygons_and_boxes(label_path: Path) -> Tuple[List[Dict[str, Any]], List[Tuple[float, float, float, float]]]:
    """读取LabelMe标签并返回多边形与边界框信息"""
    if not label_path.exists():
        return [], []

    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as exc:
        print(f"  - 警告：无法解析标签 {label_path}: {exc}")
        return [], []

    polygons: List[Dict[str, Any]] = []
    boxes: List[Tuple[float, float, float, float]] = []
    for shape in data.get('shapes', []):
        points = shape.get('points') or []
        if len(points) < 2:
            continue
        pts = np.array(points, dtype=np.float32)
        xs = pts[:, 0]
        ys = pts[:, 1]
        x1, x2 = float(xs.min()), float(xs.max())
        y1, y2 = float(ys.min()), float(ys.max())
        if x2 <= x1 or y2 <= y1:
            continue
        polygons.append({
            'points': pts.astype(np.int32),
            'label': shape.get('label', '')
        })
        boxes.append((x1, y1, x2, y2))
    return polygons, boxes


def predictions_to_boxes(predictions: Sequence[Dict[str, Any]]) -> List[Tuple[float, float, float, float]]:
    boxes: List[Tuple[float, float, float, float]] = []
    for pred in predictions:
        if not pred.get('is_defect'):
            continue
        px, py = pred['position']
        pw, ph = pred['size']
        boxes.append((float(px), float(py), float(px + pw), float(py + ph)))
    return boxes


def match_predictions_to_gt(pred_boxes: Sequence[Tuple[float, float, float, float]],
                            gt_boxes: Sequence[Tuple[float, float, float, float]],
                            threshold: float) -> Tuple[List[Dict[str, Any]], List[int], List[int]]:
    matches: List[Dict[str, Any]] = []
    matched_gt = set()
    matched_pred = set()

    for pred_idx, pred_box in enumerate(pred_boxes):
        best_ratio = 0.0
        best_gt_idx = None
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            inter_area = compute_intersection_area(pred_box, gt_box)
            if inter_area <= 0:
                continue
            gt_area = max(0.0, gt_box[2] - gt_box[0]) * max(0.0, gt_box[3] - gt_box[1])
            if gt_area <= 0:
                continue
            overlap_ratio = inter_area / gt_area
            if overlap_ratio > best_ratio:
                best_ratio = overlap_ratio
                best_gt_idx = gt_idx

        if best_gt_idx is not None and best_ratio >= threshold:
            matches.append({
                'pred_idx': pred_idx,
                'gt_idx': best_gt_idx,
                'overlap_ratio': best_ratio
            })
            matched_gt.add(best_gt_idx)
            matched_pred.add(pred_idx)

    false_preds = [idx for idx in range(len(pred_boxes)) if idx not in matched_pred]
    false_negatives = [idx for idx in range(len(gt_boxes)) if idx not in matched_gt]
    return matches, false_preds, false_negatives


def draw_polygons(canvas: np.ndarray, polygons: Sequence[Dict[str, Any]], color: Tuple[int, int, int]):
    if not polygons:
        return
    for poly in polygons:
        pts = poly.get('points')
        if pts is None or pts.size == 0:
            continue
        reshaped = pts.reshape((-1, 1, 2))
        cv2.polylines(canvas, [reshaped], isClosed=True, color=color, thickness=2)


def draw_box_with_label(canvas: np.ndarray,
                        box: Sequence[float],
                        text: str,
                        color: Tuple[int, int, int],
                        font_renderer: FontRenderer):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
    origin = (x1 + 2, max(y1 + 18, 18))
    font_renderer.draw(canvas, text, origin, color)


def render_verification_overlay(canvas: np.ndarray,
                                polygons: Sequence[Dict[str, Any]],
                                gt_boxes: Sequence[Tuple[float, float, float, float]],
                                merged_pred_boxes: Sequence[Tuple[float, float, float, float]],
                                matches: Sequence[Dict[str, Any]],
                                false_negatives: Sequence[int],
                                font_renderer: FontRenderer):
    draw_polygons(canvas, polygons, color=(0, 255, 255))
    matched_pred_ids = {m['pred_idx'] for m in matches}

    for idx, box in enumerate(merged_pred_boxes):
        if idx in matched_pred_ids:
            draw_box_with_label(canvas, box, "正确", (0, 200, 0), font_renderer)
        else:
            draw_box_with_label(canvas, box, "错误", (0, 0, 255), font_renderer)

    for gt_idx in false_negatives:
        if gt_idx < 0 or gt_idx >= len(gt_boxes):
            continue
        draw_box_with_label(canvas, gt_boxes[gt_idx], "漏检", (0, 215, 255), font_renderer)


def compute_intersection_area(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    return inter_w * inter_h


def compute_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    inter_area = compute_intersection_area(box_a, box_b)
    if inter_area <= 0:
        return 0.0
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def boxes_overlap(box_a: Sequence[float], box_b: Sequence[float]) -> bool:
    return compute_intersection_area(box_a, box_b) > 0.0


def merge_boxes(box_a: Sequence[float], box_b: Sequence[float]) -> Tuple[float, float, float, float]:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    return min(ax1, bx1), min(ay1, by1), max(ax2, bx2), max(ay2, by2)


def merge_overlapping_boxes(boxes: Sequence[Tuple[float, float, float, float]]) -> List[Tuple[float, float, float, float]]:
    merged: List[Tuple[float, float, float, float]] = []
    for box in boxes:
        current = box
        merged_with_existing = True
        while merged_with_existing:
            merged_with_existing = False
            for idx, existing in enumerate(merged):
                if boxes_overlap(current, existing):
                    current = merge_boxes(current, existing)
                    merged.pop(idx)
                    merged_with_existing = True
                    break
        merged.append(current)
    return merged


def gather_images(image_dir: Path, max_images: Optional[int]) -> List[Path]:
    image_files = []
    for ext in SUPPORTED_IMAGE_EXTS:
        image_files.extend(image_dir.glob(f"*{ext}"))
        image_files.extend(image_dir.glob(f"*{ext.upper()}"))
    image_files.sort()
    if max_images is not None:
        image_files = image_files[:max_images]
    return image_files


def evaluate_dataset(dataset: str,
                     image_dir: Path,
                     label_dir: Path,
                     pipeline: SliceClassificationPipeline,
                     label_overlap_threshold: float,
                     max_images: Optional[int],
                     visualize: bool,
                     visualization_dir: Optional[Path],
                     add_colorbar: bool,
                     font_renderer: Optional[FontRenderer]) -> Dict[str, Any]:
    image_files = gather_images(image_dir, max_images)
    if not image_files:
        print(f"[{dataset}] 未找到图像，跳过。")
        return {
            "dataset": dataset,
            "num_images": 0,
            "tp": 0,
            "fp": 0,
            "gt": 0,
            "precision": None,
            "recall": None
        }

    dataset_tp = 0
    dataset_fp = 0
    dataset_gt = 0
    images_processed = 0
    dataset_vis_dir = None
    renderer = font_renderer
    if visualize and visualization_dir is not None:
        dataset_vis_dir = visualization_dir / dataset
        if renderer is None:
            renderer = FontRenderer(font_size=20)

    for image_path in image_files:
        label_path = label_dir / f"{image_path.stem}.json"

        try:
            image = load_image(image_path)
            result = pipeline.detect_slice_classify(image, image_id=str(image_path))
        except Exception as exc:
            print(f"[{dataset}] 警告：处理 {image_path} 失败：{exc}")
            continue

        polygons, gt_boxes = load_label_polygons_and_boxes(label_path)
        defect_patches = result.get('patch_predictions', [])
        pred_boxes = predictions_to_boxes(defect_patches)
        merged_pred_boxes = merge_overlapping_boxes(pred_boxes)
        matches, false_preds, false_negatives = match_predictions_to_gt(
            merged_pred_boxes, gt_boxes, label_overlap_threshold
        )

        dataset_tp += len(matches)
        dataset_fp += len(false_preds)
        dataset_gt += len(gt_boxes)
        images_processed += 1

        if visualize and dataset_vis_dir is not None and renderer is not None:
            heatmap = pipeline.generate_heatmap(image.shape, result.get('patch_predictions', []))
            overlay = pipeline.create_heatmap_overlay(image, heatmap, apply_gaussian_blur=True)
            if (not add_colorbar) or heatmap.max() <= 0:
                overlay_with_color = overlay
            else:
                overlay_with_color = pipeline.add_colorbar(overlay, heatmap)

            render_verification_overlay(
                overlay_with_color,
                polygons,
                gt_boxes,
                merged_pred_boxes,
                matches,
                false_negatives,
                renderer
            )

            dataset_vis_dir.mkdir(parents=True, exist_ok=True)
            vis_path = dataset_vis_dir / f"{image_path.stem}_validation.jpg"
            cv2.imwrite(str(vis_path), overlay_with_color)

    precision = (dataset_tp / (dataset_tp + dataset_fp)) if (dataset_tp + dataset_fp) > 0 else None
    recall = (dataset_tp / dataset_gt) if dataset_gt > 0 else None

    precision_str = f"{precision:.3f}" if precision is not None else "N/A"
    recall_str = f"{recall:.3f}" if recall is not None else "N/A"
    print(f"[{dataset}] 图像: {images_processed}, TP: {dataset_tp}, FP: {dataset_fp}, "
          f"GT: {dataset_gt}, Precision: {precision_str}, Recall: {recall_str}")

    return {
        "dataset": dataset,
        "num_images": images_processed,
        "tp": dataset_tp,
        "fp": dataset_fp,
        "gt": dataset_gt,
        "precision": precision,
        "recall": recall
    }


def main():
    args = parse_args()
    image_base = Path(args.image_base)
    label_base = Path(args.label_base)
    visualization_root: Optional[Path] = None
    font_renderer: Optional[FontRenderer] = None
    if args.visualize:
        visualization_root = Path(args.visualize_dir)
        visualization_root.mkdir(parents=True, exist_ok=True)
        font_renderer = FontRenderer(font_path=args.font_path, font_size=args.font_size)

    roi_detector = build_roi_detector(args)

    pipeline = SliceClassificationPipeline(
        model_path=args.cls_weights,
        window_size=tuple(args.window_size),
        overlap_ratio=args.overlap,
        enhance_mode=args.enhance_mode,
        confidence_threshold=args.confidence_threshold,
        use_confidence_weight=args.use_confidence_weight,
        colormap=args.colormap,
        alpha=args.alpha,
        display_mode=args.display_mode,
        defect_class_ids=args.defect_classes,
        roi_detector=roi_detector,
        use_full_image_if_no_roi=False
    )

    reports: List[Dict[str, Any]] = []
    global_tp = global_fp = global_gt = 0

    for dataset in args.datasets:
        image_dir = image_base / dataset
        label_dir = label_base / dataset / "label"
        if not image_dir.exists():
            print(f"[{dataset}] 图像目录不存在: {image_dir}, 跳过。")
            continue
        if not label_dir.exists():
            print(f"[{dataset}] 标签目录不存在: {label_dir}, 跳过。")
            continue

        report = evaluate_dataset(
            dataset=dataset,
            image_dir=image_dir,
            label_dir=label_dir,
            pipeline=pipeline,
            label_overlap_threshold=args.label_overlap_threshold,
            max_images=args.max_images,
            visualize=args.visualize,
            visualization_dir=visualization_root,
            add_colorbar=not args.visualize_no_colorbar,
            font_renderer=font_renderer
        )
        reports.append(report)
        global_tp += report.get("tp", 0) or 0
        global_fp += report.get("fp", 0) or 0
        global_gt += report.get("gt", 0) or 0

    overall_precision = (global_tp / (global_tp + global_fp)) if (global_tp + global_fp) > 0 else None
    overall_recall = (global_tp / global_gt) if global_gt > 0 else None

    summary = {
        "datasets": reports,
        "overall": {
            "tp": global_tp,
            "fp": global_fp,
            "gt": global_gt,
            "precision": overall_precision,
            "recall": overall_recall
        },
        "config": {
            "cls_weights": args.cls_weights,
            "roi_weights": args.roi_weights,
            "image_base": str(image_base),
            "label_base": str(label_base),
            "datasets": args.datasets,
            "label_overlap_threshold": args.label_overlap_threshold,
            "window_size": args.window_size,
            "overlap": args.overlap,
            "confidence_threshold": args.confidence_threshold,
            "defect_classes": args.defect_classes,
            "use_full_image_if_no_roi": False,
            "visualize": args.visualize,
            "visualize_dir": str(visualization_root) if visualization_root else None,
            "visualize_no_colorbar": args.visualize_no_colorbar
        }
    }

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n验证完成。")
    if overall_precision is not None:
        print(f"总体 Precision: {overall_precision:.3f}")
    else:
        print("总体 Precision: N/A")
    if overall_recall is not None:
        print(f"总体 Recall: {overall_recall:.3f}")
    else:
        print("总体 Recall: N/A")
    print(f"报告已保存至: {report_path}")


if __name__ == "__main__":
    main()
