#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""对 run_inference_pipeline 推理结果进行验证与可视化辅助输出。"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from tqdm import tqdm

from utils.annotation_loader import AnnotationLoader, AnnotationRecord
from utils.pipeline_utils import FontRenderer, load_image

SCRIPT_DIR = Path(__file__).resolve().parent


STATUS_SUCCESS_CLASS = "成功分类"
STATUS_SUCCESS_DETECT = "成功检出"
STATUS_MISSED = "漏检"
STATUS_UNLABELED = "漏标注"
STATUS_FALSE = "误检"

STATUS_COLORS = {
    STATUS_SUCCESS_CLASS: (46, 204, 113),    # 绿色
    STATUS_SUCCESS_DETECT: (255, 165, 0),    # 橙色
    STATUS_MISSED: (0, 0, 255),              # 红色
    STATUS_UNLABELED: (0, 215, 255),         # 金色 (BGR)
    STATUS_FALSE: (142, 68, 173)             # 紫色
}

DEFAULT_CLASS_NAME = "未标注"
DEFAULT_FONT_FAMILIES = ["SimHei", "Noto Sans CJK SC", "Noto Sans CJK JP", "Microsoft YaHei", "PingFang SC", "DejaVu Sans"]

plt.rcParams["font.sans-serif"] = DEFAULT_FONT_FAMILIES
plt.rcParams["axes.unicode_minus"] = False


@dataclass
class PredictionRecord:
    prediction_id: str
    bbox: List[float]
    polygon: List[List[float]]
    class_id: Optional[int]
    class_name: str
    confidence: Optional[float]
    roi_index: Optional[int]
    source_roi_bbox: Optional[List[float]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prediction_id": self.prediction_id,
            "bbox": [float(v) for v in self.bbox],
            "polygon": [[float(px), float(py)] for px, py in self.polygon],
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "roi_index": self.roi_index,
            "source_roi_bbox": self.source_roi_bbox,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="读取推理结果JSON，结合LabelMe/COCO标注计算缺陷指标并生成辅助可视化。"
    )
    parser.add_argument("--inference-json", required=True, help="run_inference_pipeline 输出的JSON路径")
    parser.add_argument("--output-dir", default="validation_outputs",
                        help="验证结果输出目录（包含数据与可视化temp文件）")
    parser.add_argument("--image-root", required=True,
                        help="推理图像的根目录，用于解析相对路径与拷贝原图")

    parser.add_argument("--label-format", choices=["labelme", "coco"], required=True,
                        help="标注格式")
    parser.add_argument("--label-root",
                        help="LabelMe根目录（相对结构需与image-root一致，可通过dir token替换）")
    parser.add_argument("--label-extension", default=".json", help="LabelMe标签扩展名，默认.json")
    parser.add_argument("--image-dir-token", default="images",
                        help="在相对路径中需要被替换为标签目录的文件夹名（LabelMe模式）")
    parser.add_argument("--label-dir-token", default="label",
                        help="替换image-dir-token后的目标文件夹名（LabelMe模式）")
    parser.add_argument("--coco-json", help="COCO格式的annotations json路径")

    parser.add_argument("--iou-threshold", type=float, default=0.3,
                        help="判定命中的最小IoU阈值")
    parser.add_argument("--copy-images", dest="copy_images", action="store_true",
                        help="将原图拷贝到输出目录的media子目录，便于后续打包")
    parser.add_argument("--no-copy-images", dest="copy_images", action="store_false",
                        help="不拷贝原图，仅记录路径")
    parser.set_defaults(copy_images=True)
    parser.add_argument("--font-path", help="用于中文显示的字体路径，可选")
    parser.add_argument("--font-size", type=int, default=24, help="绘制文字的字号")
    parser.add_argument("--match-mode", choices=["best", "multi"], default="multi",
                        help="best=每个GT只匹配IoU最大的预测；multi=所有超过阈值的预测都视为命中")
    parser.add_argument("--verified-threshold", type=float, default=0.75,
                        help="(1-IoU)*confidence 超过该阈值即视为漏标注，默认0.75")
    parser.add_argument("--verified-image-dir", default=None,
                        help="漏标注原图复制到的子目录，默认 verified_image（位于输出目录内）")
    parser.add_argument("--verified-dir", dest="legacy_verified_dir", default=None,
                        help="(兼容参数) 同 --verified-image-dir，后续版本将移除")
    parser.add_argument("--export-verified-bundle", action="store_true",
                        help="若设置，则额外导出仅包含漏标注样本的独立验证包（含 HTML ）")
    parser.add_argument("--verified-bundle-dir", default="verified_bundle",
                        help="漏标注验证包保存子目录，默认 verified_bundle")
    parser.add_argument("--verified-bundle-title", default="漏标注核对报告",
                        help="漏标注验证包 HTML 标题")

    return parser.parse_args()


def load_inference_results(json_path: Path) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def compute_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def normalize_class_name(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    return str(name).strip().lower()


def classes_match(pred: Dict[str, Any], gt: Dict[str, Any]) -> bool:
    pred_id = pred.get("class_id")
    gt_id = gt.get("class_id")
    if pred_id is not None and gt_id is not None:
        try:
            return int(pred_id) == int(gt_id)
        except Exception:
            pass
    pred_name = normalize_class_name(pred.get("class_name"))
    gt_name = normalize_class_name(gt.get("class_name"))
    return pred_name is not None and pred_name == gt_name


def flatten_predictions(image_result: Dict[str, Any]) -> List[PredictionRecord]:
    mode = image_result.get("mode") or image_result.get("result_mode")
    if mode not in {"seg", "det"}:
        raise ValueError(f"仅支持seg/det模式，收到: {mode}")

    rois: List[Dict[str, Any]] = image_result.get("rois", [])
    predictions: List[PredictionRecord] = []
    for roi in rois:
        roi_bbox = roi.get("bbox")
        roi_index = roi.get("roi_index")
        records = roi.get("defects") if mode == "seg" else roi.get("detections")
        if not records:
            continue
        for idx, record in enumerate(records):
            bbox = record.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            polygon = record.get("polygon") if mode == "seg" else None
            predictions.append(PredictionRecord(
                prediction_id=f"{image_result.get('image_path', 'image')}_pred_{len(predictions)}",
                bbox=[float(v) for v in bbox],
                polygon=[[float(px), float(py)] for px, py in polygon] if polygon else [],
                class_id=_safe_int(record.get("class_id")),
                class_name=str(record.get("class_name", "")),
                confidence=_safe_float(record.get("confidence")),
                roi_index=_safe_int(roi_index),
                source_roi_bbox=[float(v) for v in roi_bbox] if roi_bbox else None,
            ))
    return predictions


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def evaluate_image(predictions: List[PredictionRecord],
                   annotations: List[AnnotationRecord],
                   iou_threshold: float,
                   match_mode: str,
                   verified_threshold: float) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    pred_data = [pred.to_dict() for pred in predictions]
    gt_data = [ann.to_dict() for ann in annotations]
    allow_multi = (match_mode == "multi")

    for pred in pred_data:
        pred["matches"] = []
        pred["status"] = STATUS_FALSE
        pred["matched_gt"] = []
        pred["iou"] = 0.0
        pred["best_iou"] = 0.0
        pred["best_gt_index"] = None
        pred["unlabeled_score"] = None

    for gt in gt_data:
        gt["matches"] = []
        gt["status"] = None
        gt["best_iou"] = 0.0
        gt["best_prediction"] = None
        gt["iou"] = 0.0

    status_counter = Counter()

    if allow_multi:
        for pred_idx, pred in enumerate(pred_data):
            for gt_idx, gt in enumerate(gt_data):
                iou = compute_iou(pred["bbox"], gt["bbox"])
                if iou > pred["best_iou"]:
                    pred["best_iou"] = iou
                    pred["best_gt_index"] = gt_idx
                if iou > gt["best_iou"]:
                    gt["best_iou"] = iou
                    gt["best_prediction"] = pred_idx
                if iou >= iou_threshold:
                    match_status = STATUS_SUCCESS_CLASS if classes_match(pred, gt) else STATUS_SUCCESS_DETECT
                    pred_match = {
                        "gt_index": gt_idx,
                        "iou": iou,
                        "status": match_status
                    }
                    pred["matches"].append(pred_match)
                    pred["matched_gt"].append(gt_idx)
                    pred["iou"] = max(pred["iou"], iou)
                    gt["matches"].append({
                        "pred_index": pred_idx,
                        "iou": iou,
                        "status": match_status
                    })
        for gt in gt_data:
            if gt["matches"]:
                if any(m["status"] == STATUS_SUCCESS_CLASS for m in gt["matches"]):
                    gt["status"] = STATUS_SUCCESS_CLASS
                else:
                    gt["status"] = STATUS_SUCCESS_DETECT
                gt["matched_prediction"] = [m["pred_index"] for m in gt["matches"]]
                gt["iou"] = max(m["iou"] for m in gt["matches"])
            else:
                gt["status"] = STATUS_MISSED
                gt["matched_prediction"] = None
                gt["iou"] = gt.get("best_iou", 0.0)
                status_counter[STATUS_MISSED] += 1
    else:
        for gt_idx, gt in enumerate(gt_data):
            best_pred_idx = None
            best_iou = 0.0
            for pred_idx, pred in enumerate(pred_data):
                iou = compute_iou(pred["bbox"], gt["bbox"])
                if iou > pred["best_iou"]:
                    pred["best_iou"] = iou
                    pred["best_gt_index"] = gt_idx
                if iou > gt["best_iou"]:
                    gt["best_iou"] = iou
                    gt["best_prediction"] = pred_idx
                if iou > best_iou:
                    best_iou = iou
                    best_pred_idx = pred_idx

            if best_pred_idx is None:
                gt["status"] = STATUS_MISSED
                gt["matched_prediction"] = None
                gt["iou"] = gt.get("best_iou", 0.0)
                status_counter[STATUS_MISSED] += 1
                continue

            if best_iou < iou_threshold:
                gt["status"] = STATUS_MISSED
                gt["matched_prediction"] = None
                gt["iou"] = best_iou
                status_counter[STATUS_MISSED] += 1
                continue

            pred = pred_data[best_pred_idx]
            match_status = STATUS_SUCCESS_CLASS if classes_match(pred, gt) else STATUS_SUCCESS_DETECT
            gt["status"] = match_status
            gt["matched_prediction"] = best_pred_idx
            gt["iou"] = best_iou
            pred_match = {
                "gt_index": gt_idx,
                "iou": best_iou,
                "status": match_status
            }
            pred["matches"].append(pred_match)
            pred["matched_gt"].append(gt_idx)
            pred["iou"] = max(pred["iou"], best_iou)

    for pred in pred_data:
        pred["iou"] = pred.get("iou", pred.get("best_iou", 0.0))
        if pred["matches"]:
            if any(m["status"] == STATUS_SUCCESS_CLASS for m in pred["matches"]):
                pred_status = STATUS_SUCCESS_CLASS
            else:
                pred_status = STATUS_SUCCESS_DETECT
            pred["status"] = pred_status
            status_counter[pred_status] += 1
        else:
            pred["matched_gt"] = None
            confidence = pred.get("confidence")
            best_iou = pred.get("best_iou") or 0.0
            score = None
            if confidence is not None:
                score = (1.0 - float(best_iou)) * float(confidence)
                pred["unlabeled_score"] = score
            if score is not None and score >= verified_threshold:
                pred["status"] = STATUS_UNLABELED
                status_counter[STATUS_UNLABELED] += 1
            else:
                pred["status"] = STATUS_FALSE
                status_counter[STATUS_FALSE] += 1

    total_preds = len(pred_data)
    total_gt = len(gt_data)
    detected_preds = status_counter[STATUS_SUCCESS_CLASS] + status_counter[STATUS_SUCCESS_DETECT]
    matched_gt = sum(1 for gt in gt_data if gt.get("status") in {STATUS_SUCCESS_CLASS, STATUS_SUCCESS_DETECT})

    metrics = {
        "defect_precision": (detected_preds / total_preds) if total_preds > 0 else None,
        "defect_recall": (matched_gt / total_gt) if total_gt > 0 else None,
        "classification_accuracy": (status_counter[STATUS_SUCCESS_CLASS] / detected_preds) if detected_preds > 0 else None,
        "counts": {
            "predictions": total_preds,
            "ground_truth": total_gt,
            STATUS_SUCCESS_CLASS: status_counter[STATUS_SUCCESS_CLASS],
            STATUS_SUCCESS_DETECT: status_counter[STATUS_SUCCESS_DETECT],
            STATUS_FALSE: status_counter[STATUS_FALSE],
            STATUS_UNLABELED: status_counter[STATUS_UNLABELED],
            STATUS_MISSED: status_counter[STATUS_MISSED]
        }
    }

    return {
        "predictions": pred_data,
        "annotations": gt_data
    }, metrics


def configure_matplotlib_font(font_path: Optional[str]):
    if font_path:
        try:
            font_manager.fontManager.addfont(font_path)
            font_prop = font_manager.FontProperties(fname=font_path)
            font_name = font_prop.get_name()
            plt.rcParams["font.sans-serif"] = [font_name]
            return
        except Exception as exc:
            print(f"⚠️ 无法加载指定字体 {font_path}: {exc}，改用默认字体")
    plt.rcParams["font.sans-serif"] = DEFAULT_FONT_FAMILIES


def _display_class_name(value: Any) -> str:
    if value is None:
        return DEFAULT_CLASS_NAME
    name = str(value).strip()
    return name or DEFAULT_CLASS_NAME


def _init_class_stat() -> Dict[str, int]:
    return {
        "pred_total": 0,
        "pred_detected": 0,
        "pred_success_class": 0,
        "gt_total": 0,
        "gt_detected": 0
    }


def update_class_statistics(class_stats: Dict[str, Dict[str, int]],
                            eval_data: Dict[str, Any]):
    for pred in eval_data["predictions"]:
        class_name = _display_class_name(pred.get("class_name"))
        if class_name not in class_stats:
            class_stats[class_name] = _init_class_stat()
        stats = class_stats[class_name]
        stats["pred_total"] += 1
        status = pred.get("status")
        if status in {STATUS_SUCCESS_CLASS, STATUS_SUCCESS_DETECT}:
            stats["pred_detected"] += 1
            if status == STATUS_SUCCESS_CLASS:
                stats["pred_success_class"] += 1

    for gt in eval_data["annotations"]:
        class_name = _display_class_name(gt.get("class_name"))
        if class_name not in class_stats:
            class_stats[class_name] = _init_class_stat()
        stats = class_stats[class_name]
        stats["gt_total"] += 1
        if gt.get("status") in {STATUS_SUCCESS_CLASS, STATUS_SUCCESS_DETECT}:
            stats["gt_detected"] += 1


def build_per_class_metrics(class_stats: Dict[str, Dict[str, int]]) -> List[Dict[str, Any]]:
    metrics: List[Dict[str, Any]] = []
    for class_name in sorted(class_stats.keys()):
        stats = class_stats[class_name]
        pred_total = stats["pred_total"]
        pred_detected = stats["pred_detected"]
        pred_success = stats["pred_success_class"]
        gt_total = stats["gt_total"]
        gt_detected = stats["gt_detected"]

        precision = (pred_detected / pred_total) if pred_total > 0 else None
        recall = (gt_detected / gt_total) if gt_total > 0 else None
        classification_acc = (pred_success / pred_detected) if pred_detected > 0 else None

        metrics.append({
            "class_name": class_name,
            "defect_precision": precision,
            "defect_precision_counts": [pred_detected, pred_total],
            "defect_recall": recall,
            "defect_recall_counts": [gt_detected, gt_total],
            "classification_accuracy": classification_acc,
            "classification_counts": [pred_success, pred_detected]
        })
    return metrics


def plot_per_class_metrics(per_class_metrics: List[Dict[str, Any]],
                           output_dir: Path,
                           overall_metrics: Dict[str, Optional[float]]) -> Optional[Path]:
    if not per_class_metrics:
        return None

    classes = [m["class_name"] for m in per_class_metrics]
    x_positions = np.arange(len(classes))
    bar_width = 0.28
    spacing_factor = 1.0  # 控制组内柱子间距

    specs = [
        ("defect_precision", "缺陷识别准确率", "defect_precision_counts", "#5470C6"),
        ("defect_recall", "缺陷识别召回率", "defect_recall_counts", "#91CC75"),
        ("classification_accuracy", "缺陷分类准确率", "classification_counts", "#EE6666")
    ]

    fig_width = max(12, len(classes) * 1.5)
    fig, ax = plt.subplots(figsize=(fig_width, 6.5))
    bars_collection = []

    for idx, (metric_key, title, count_key, color) in enumerate(specs):
        offset = (idx - 1) * bar_width * spacing_factor
        positions = x_positions + offset
        values = []
        labels = []
        for item in per_class_metrics:
            val = item[metric_key]
            counts = item[count_key]
            numerator, denominator = counts
            value_percent = (val * 100) if val is not None else 0.0
            values.append(value_percent)
            labels.append(f"{numerator}/{denominator}\n{value_percent:.1f}%")

        avg_val = overall_metrics.get(metric_key)
        avg_text = f"{avg_val * 100:.2f}%" if avg_val is not None else "--"
        legend_label = f"{title}: {avg_text}"
        bars = ax.bar(positions, values, width=bar_width, color=color, label=legend_label)
        bars_collection.append((bars, labels))

    ax.set_ylabel("百分比(%)")
    ax.set_ylim(0, 115)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(classes, rotation=28, ha="right")
    ax.legend()
    ax.set_title("各类别缺陷识别/召回/分类准确率")

    for bars, labels in bars_collection:
        for bar, label in zip(bars, labels):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 4,
                label,
                ha="center",
                va="bottom",
                fontsize=8,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1.0)
            )

    fig.tight_layout()
    output_path = output_dir / "per_class_metrics.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _draw_polygon(canvas: np.ndarray,
                  polygon: Sequence[Sequence[float]],
                  color: Tuple[int, int, int],
                  thickness: int = 2,
                  alpha: float = 0.25) -> bool:
    if not polygon or len(polygon) < 3:
        return False
    pts = np.array([
        (int(round(pt[0])), int(round(pt[1])))
        for pt in polygon
    ], dtype=np.int32)
    overlay = canvas.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)
    cv2.polylines(canvas, [pts], True, color, thickness)
    return True


def draw_overlay(image: np.ndarray,
                 eval_data: Dict[str, Any],
                 font_renderer: FontRenderer) -> np.ndarray:
    canvas = image.copy()
    for pred in eval_data["predictions"]:
        bbox = pred["bbox"]
        status = pred.get("status", STATUS_FALSE)
        color = STATUS_COLORS.get(status, (255, 255, 255))
        polygon = pred.get("polygon")
        drew_polygon = _draw_polygon(canvas, polygon, color)
        if not drew_polygon:
            x1, y1, x2, y2 = [int(round(v)) for v in bbox]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        else:
            x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        label_parts = [status]
        cls_name = pred.get("class_name")
        if cls_name:
            label_parts.append(str(cls_name))
        if pred.get("confidence") is not None:
            label_parts.append(f"{pred['confidence']:.2f}")
        if pred.get("iou") is not None:
            label_parts.append(f"IoU:{pred['iou']:.2f}")
        label_text = " | ".join(label_parts)
        origin = (x1, max(18, y1))
        font_renderer.draw(canvas, label_text, origin, color)

    for gt in eval_data["annotations"]:
        status = gt.get("status")
        if status == STATUS_MISSED:
            bbox = gt["bbox"]
            color = STATUS_COLORS.get(status, (0, 0, 255))
            polygon = gt.get("polygon")
            if not _draw_polygon(canvas, polygon, color, thickness=2, alpha=0.15):
                x1, y1, x2, y2 = [int(round(v)) for v in bbox]
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            label = f"标注:{gt.get('class_name', '')} {status}"
            origin = (int(round(bbox[0])), min(canvas.shape[0] - 10, int(round(bbox[3])) - 5))
            font_renderer.draw(canvas, label, origin, color)
    return canvas


def copy_image(image_path: Path, relative_path: Path, media_root: Path) -> Path:
    target_path = media_root / relative_path
    ensure_dir(target_path.parent)
    shutil.copy2(image_path, target_path)
    return target_path


def copy_relative_artifact(src_root: Path, relative_path: Optional[str], dst_root: Path) -> bool:
    if not relative_path:
        return False
    rel_path = Path(relative_path)
    if rel_path.is_absolute():
        rel_path = Path(rel_path.name)
    src_path = src_root / rel_path
    if not src_path.exists():
        print(f"[警告] 无法在 {src_path} 找到文件，跳过复制至漏标注包")
        return False
    target_path = dst_root / rel_path
    ensure_dir(target_path.parent)
    shutil.copy2(src_path, target_path)
    return True


def build_verified_summary(entries: List[Dict[str, Any]],
                           base_config: Dict[str, Any]) -> Dict[str, Any]:
    counts_total = Counter()
    for entry in entries:
        counts = entry.get("status_counts") or {}
        counts_total["predictions"] += counts.get("predictions", 0)
        counts_total["ground_truth"] += counts.get("ground_truth", 0)
        counts_total[STATUS_SUCCESS_CLASS] += counts.get(STATUS_SUCCESS_CLASS, 0)
        counts_total[STATUS_SUCCESS_DETECT] += counts.get(STATUS_SUCCESS_DETECT, 0)
        counts_total[STATUS_FALSE] += counts.get(STATUS_FALSE, 0)
        counts_total[STATUS_UNLABELED] += counts.get(STATUS_UNLABELED, 0)
        counts_total[STATUS_MISSED] += counts.get(STATUS_MISSED, 0)

    total_pred = counts_total["predictions"]
    total_gt = counts_total["ground_truth"]
    success_class = counts_total[STATUS_SUCCESS_CLASS]
    success_detect = counts_total[STATUS_SUCCESS_DETECT]
    detected_total = success_class + success_detect

    precision = (detected_total / total_pred) if total_pred > 0 else None
    recall = (detected_total / total_gt) if total_gt > 0 else None
    classification_accuracy = (success_class / detected_total) if detected_total > 0 else None

    config_copy = dict(base_config or {})
    config_copy["bundle_size"] = len(entries)

    overall = {
        "defect_precision": precision,
        "defect_recall": recall,
        "classification_accuracy": classification_accuracy,
        "counts": {
            "predictions": total_pred,
            "ground_truth": total_gt,
            STATUS_SUCCESS_CLASS: success_class,
            STATUS_SUCCESS_DETECT: success_detect,
            STATUS_FALSE: counts_total[STATUS_FALSE],
            STATUS_UNLABELED: counts_total[STATUS_UNLABELED],
            STATUS_MISSED: counts_total[STATUS_MISSED]
        },
        "verified_images": len(entries)
    }

    return {
        "config": config_copy,
        "overall": overall,
        "per_class": []
    }


def generate_verified_report(bundle_dir: Path, title: str):
    html_path = bundle_dir / "report.html"
    vis_script = SCRIPT_DIR / "visualize_validation_results.py"
    cmd = [
        sys.executable,
        str(vis_script),
        "--validation-dir", str(bundle_dir),
        "--output-html", str(html_path),
        "--title", title
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"[警告] 生成漏标注报告失败：{exc}")


def export_verified_bundle(output_dir: Path,
                           manifest: List[Dict[str, Any]],
                           summary: Dict[str, Any],
                           bundle_dir: Path,
                           title: str):
    verified_entries: List[Dict[str, Any]] = []
    for item in manifest:
        counts = item.get("status_counts") or {}
        if counts.get(STATUS_UNLABELED, 0) <= 0:
            continue
        entry_copy = json.loads(json.dumps(item, ensure_ascii=False))
        if not entry_copy.get("copied_image_path") and entry_copy.get("verified_image_path"):
            entry_copy["copied_image_path"] = entry_copy["verified_image_path"]
        verified_entries.append(entry_copy)

    if not verified_entries:
        print("ℹ️ 未检测到漏标注样本，跳过 verified bundle 导出。")
        return

    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    ensure_dir(bundle_dir)

    copied_paths = set()
    for entry in verified_entries:
        rel_paths = {
            entry.get("detail_path"),
            entry.get("overlay_path"),
            entry.get("copied_image_path"),
            entry.get("verified_image_path"),
        }
        for rel_path in rel_paths:
            if rel_path and rel_path not in copied_paths:
                if copy_relative_artifact(output_dir, rel_path, bundle_dir):
                    copied_paths.add(rel_path)

    manifest_path = bundle_dir / "data" / "manifest.json"
    ensure_dir(manifest_path.parent)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"images": verified_entries}, f, ensure_ascii=False, indent=2)

    bundle_summary = build_verified_summary(verified_entries, summary.get("config"))
    bundle_summary["config"]["bundle_dir"] = str(bundle_dir)
    bundle_summary["config"]["bundle_source"] = str(output_dir)
    summary_path = bundle_dir / "metrics_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(bundle_summary, f, ensure_ascii=False, indent=2)

    generate_verified_report(bundle_dir, title)
    print(f"✅ 已生成漏标注验证包：{bundle_dir}")


def relative_image_path(image_path: Path, image_root: Path) -> Path:
    if image_path.is_relative_to(image_root):
        return image_path.relative_to(image_root)
    return Path(image_path.name)


def main():
    args = parse_args()
    if not args.verified_image_dir:
        args.verified_image_dir = args.legacy_verified_dir or "verified_image"
    delattr(args, "legacy_verified_dir")
    configure_matplotlib_font(args.font_path)
    inference_path = Path(args.inference_json)
    output_dir = Path(args.output_dir)
    data_dir = output_dir / "data"
    details_dir = data_dir / "details"
    media_dir = output_dir / "media"
    temp_dir = output_dir / "temp"
    ensure_dir(details_dir)
    ensure_dir(temp_dir)
    verified_image_dir = output_dir / args.verified_image_dir
    ensure_dir(verified_image_dir)
    image_root = Path(args.image_root)

    if args.label_format == "labelme" and not args.label_root:
        raise ValueError("LabelMe 模式必须提供 --label-root")
    if args.label_format == "coco" and not args.coco_json:
        raise ValueError("COCO 模式必须提供 --coco-json")

    annotation_loader = AnnotationLoader(
        label_format=args.label_format,
        image_root=image_root,
        label_root=Path(args.label_root) if args.label_root else None,
        label_extension=args.label_extension,
        image_dir_token=args.image_dir_token,
        label_dir_token=args.label_dir_token,
        coco_json=Path(args.coco_json) if args.coco_json else None
    )

    font_renderer = FontRenderer(font_path=args.font_path, font_size=args.font_size)

    inference_data = load_inference_results(inference_path)
    mode = inference_data.get("mode")
    results: List[Dict[str, Any]] = inference_data.get("results", [])
    if mode not in {"seg", "det"}:
        raise ValueError(f"当前脚本仅支持seg/det模式，JSON中的模式为: {mode}")

    manifest: List[Dict[str, Any]] = []
    global_counts = Counter()
    global_success_class = 0
    global_success_detect = 0
    class_stats: Dict[str, Dict[str, int]] = {}
    verified_samples = 0

    for image_result in tqdm(results, desc="验证中"):
        image_path = Path(image_result.get("image_path", ""))
        if not image_path.exists():
            print(f"[警告] 图像不存在，跳过: {image_path}")
            continue

        rel_path = relative_image_path(image_path, image_root)
        try:
            image = load_image(image_path)
        except Exception as exc:
            print(f"[警告] 无法读取图像 {image_path}: {exc}")
            continue

        try:
            predictions = flatten_predictions(image_result)
        except ValueError as exc:
            print(f"[警告] {exc}，跳过 {image_path}")
            continue

        annotations = annotation_loader.load(image_path)
        eval_data, metrics = evaluate_image(
            predictions,
            annotations,
            args.iou_threshold,
            args.match_mode,
            args.verified_threshold
        )

        overlay = draw_overlay(image, eval_data, font_renderer)
        overlay_slug = rel_path.as_posix().replace('/', '_') or image_path.stem
        overlay_name = f"{overlay_slug}_validation.jpg"
        overlay_path = temp_dir / overlay_name
        cv2.imwrite(str(overlay_path), overlay)

        copied_image_rel = rel_path
        copied_image_path = None
        if args.copy_images:
            copied_image_path = copy_image(image_path, rel_path, media_dir)

        has_unlabeled = any(pred.get("status") == STATUS_UNLABELED for pred in eval_data["predictions"])
        verified_image_rel = None
        if has_unlabeled:
            verified_abs = copy_image(image_path, rel_path, verified_image_dir)
            verified_image_rel = verified_abs.relative_to(output_dir).as_posix()
            verified_samples += 1

        detail_payload = {
            "image_path": str(image_path),
            "relative_image_path": rel_path.as_posix(),
            "copied_image_path": copied_image_path.relative_to(output_dir).as_posix() if copied_image_path else None,
            "overlay_path": overlay_path.relative_to(output_dir).as_posix(),
            "mode": image_result.get("mode"),
            "width": image_result.get("width"),
            "height": image_result.get("height"),
            "num_rois": image_result.get("num_rois"),
            "metrics": metrics,
            "predictions": eval_data["predictions"],
            "annotations": eval_data["annotations"],
            "verified_image_path": verified_image_rel,
        }

        detail_file = details_dir / f"{overlay_slug}.json"
        with open(detail_file, "w", encoding="utf-8") as f:
            json.dump(detail_payload, f, ensure_ascii=False, indent=2)

        manifest.append({
            "image_path": str(image_path),
            "relative_image_path": rel_path.as_posix(),
            "detail_path": detail_file.relative_to(output_dir).as_posix(),
            "overlay_path": overlay_path.relative_to(output_dir).as_posix(),
            "copied_image_path": detail_payload["copied_image_path"],
            "verified_image_path": verified_image_rel,
            "metrics": metrics,
            "status_counts": metrics.get("counts", {})
        })

        counts = metrics["counts"]
        global_counts.update(counts)
        global_success_class += counts[STATUS_SUCCESS_CLASS]
        global_success_detect += counts[STATUS_SUCCESS_DETECT]
        update_class_statistics(class_stats, eval_data)

    per_class_metrics = build_per_class_metrics(class_stats)

    total_pred = global_counts.get("predictions", 0)
    total_gt = global_counts.get("ground_truth", 0)
    detected_total = global_success_class + global_success_detect
    summary = {
        "config": {
            "inference_json": str(inference_path),
            "image_root": str(image_root),
            "label_format": args.label_format,
            "label_root": str(args.label_root) if args.label_root else None,
            "coco_json": str(args.coco_json) if args.coco_json else None,
            "iou_threshold": args.iou_threshold,
            "copy_images": args.copy_images,
            "verified_threshold": args.verified_threshold,
            "verified_image_dir": str(verified_image_dir),
        },
        "overall": {
            "defect_precision": (detected_total / total_pred) if total_pred > 0 else None,
            "defect_recall": (detected_total / total_gt) if total_gt > 0 else None,
            "classification_accuracy": (global_success_class / detected_total) if detected_total > 0 else None,
            "counts": {
                "predictions": total_pred,
                "ground_truth": total_gt,
                STATUS_SUCCESS_CLASS: global_success_class,
                STATUS_SUCCESS_DETECT: global_success_detect,
                STATUS_FALSE: global_counts.get(STATUS_FALSE, 0),
                STATUS_UNLABELED: global_counts.get(STATUS_UNLABELED, 0),
                STATUS_MISSED: global_counts.get(STATUS_MISSED, 0)
            }
        },
        "per_class": per_class_metrics
    }
    summary["overall"]["verified_images"] = verified_samples
    plot_path = plot_per_class_metrics(per_class_metrics, output_dir, summary["overall"])
    if plot_path:
        summary["per_class_plot"] = Path(plot_path).relative_to(output_dir).as_posix()

    manifest_path = data_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"images": manifest}, f, ensure_ascii=False, indent=2)

    summary_path = output_dir / "metrics_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if args.export_verified_bundle:
        bundle_dir = output_dir / args.verified_bundle_dir
        export_verified_bundle(
            output_dir,
            manifest,
            summary,
            bundle_dir,
            args.verified_bundle_title
        )

    print("\n验证完成。")
    if summary["overall"]["defect_precision"] is not None:
        print(f"缺陷识别准确率: {summary['overall']['defect_precision']:.4f}")
    if summary["overall"]["defect_recall"] is not None:
        print(f"缺陷识别召回率: {summary['overall']['defect_recall']:.4f}")
    if summary["overall"]["classification_accuracy"] is not None:
        print(f"缺陷分类准确率: {summary['overall']['classification_accuracy']:.4f}")
    if plot_path:
        print(f"每类指标图: {plot_path}")
    print(f"Manifest: {manifest_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
