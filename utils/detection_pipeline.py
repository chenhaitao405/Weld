from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

from convert.pj.yolo_roi_extractor import WeldROIDetector
from rfdetr import RFDETRMedium, RFDETRLarge
from utils import enhance_image, calculate_stride, sliding_window_crop
from utils.pipeline_utils import (
    FontRenderer,
    align_roi_orientation,
    ensure_color,
    restore_bbox_from_rotation,
    draw_detection_instance
)


class RFDetrDetectionModel:
    """RF-DETR推理包装器，负责加载模型并执行单张图像或ROI的检测"""

    def __init__(self,
                 model_path: str,
                 confidence: float = 0.25,
                 device: Optional[str] = None,
                 optimize: bool = False,
                 optimize_batch: int = 1,
                 use_half: bool = False,
                 class_names: Optional[Sequence[str]] = None,
                 model_variant: str = "large"):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"未找到RF-DETR权重: {self.model_path}")
        self.confidence = confidence
        self.model_variant = model_variant
        self.model = self._load_model(device)
        if optimize:
            self._optimize_model(optimize_batch, use_half)
        self.class_map = self._build_class_map(class_names)

    def _load_model(self, device: Optional[str]):
        model_kwargs: Dict[str, Any] = {"pretrain_weights": str(self.model_path)}
        if device:
            model_kwargs["device"] = device
        if self.model_variant == "medium":
            return RFDETRMedium(**model_kwargs)
        return RFDETRLarge(**model_kwargs)

    def _optimize_model(self, batch_size: int, use_half: bool):
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - torch 应预装
            raise RuntimeError("需要安装torch以使用RF-DETR优化推理") from exc
        dtype = torch.float16 if use_half else torch.float32
        self.model.optimize_for_inference(batch_size=batch_size, dtype=dtype)

    def _build_class_map(self, class_names: Optional[Sequence[str]]) -> Dict[int, str]:
        if class_names:
            return {idx: str(name) for idx, name in enumerate(class_names)}
        raw_names = getattr(self.model, "class_names", None)
        if isinstance(raw_names, dict):
            return {int(k): str(v) for k, v in raw_names.items()}
        if isinstance(raw_names, (list, tuple)):
            return {idx: str(name) for idx, name in enumerate(raw_names)}
        return {}

    def predict_patch(self, patch_bgr: np.ndarray) -> List[Dict[str, Any]]:
        patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(patch_rgb)
        detections = self.model.predict(pil_image, threshold=self.confidence)
        detections = self._ensure_single_output(detections)
        if detections is None or len(getattr(detections, "xyxy", [])) == 0:
            return []

        results: List[Dict[str, Any]] = []
        for bbox, score, cls_id in zip(
                detections.xyxy, detections.confidence, detections.class_id):
            cls_id_int = int(cls_id)
            result = {
                "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                "confidence": float(score),
                "class_id": cls_id_int,
                "class_name": resolve_label(cls_id_int, self.class_map)
            }
            results.append(result)
        return results

    @staticmethod
    def _ensure_single_output(detections: Any):
        if detections is None:
            return None
        if isinstance(detections, list):
            if len(detections) == 0:
                return None
            if len(detections) == 1:
                return detections[0]
            raise RuntimeError("RF-DETR返回了批量结果，请逐个调用predict")
        return detections


def process_roi_and_detection(
        image: np.ndarray,
        image_path: Path,
        roi_detector: Optional[WeldROIDetector],
        detection_model: RFDetrDetectionModel,
        secondary_model: Optional[RFDetrDetectionModel],
        enhance_mode: str,
        patch_window: Optional[Tuple[int, int]],
        patch_overlap: float,
        fusion_iou: float,
        visualize: bool,
        visualization_dir: Optional[Path],
        font_renderer: Optional[FontRenderer],
        debug_dir: Optional[Path]) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    img_h, img_w = image.shape[:2]
    roi_boxes = roi_detector.detect_with_padding(image) if roi_detector else []
    if not roi_boxes:
        roi_boxes = [(0, 0, img_w, img_h)]

    vis_image = ensure_color(image.copy()) if visualize else None
    roi_results: List[Dict[str, Any]] = []

    for roi_idx, (x1, y1, x2, y2) in enumerate(roi_boxes):
        x1_i, y1_i, x2_i, y2_i = map(int, [x1, y1, x2, y2])
        roi_patch = image[y1_i:y2_i, x1_i:x2_i]
        if roi_patch.size == 0:
            continue

        roi_debug_dir = (debug_dir / f"roi_{roi_idx:02d}") if debug_dir else None

        aligned_roi, rotation_meta = align_roi_orientation(roi_patch)
        enhanced_roi = enhance_image(aligned_roi, mode=enhance_mode, output_bits=8)
        prepared_roi = ensure_color(enhanced_roi)
        detections_raw = detection_model.predict_patch(prepared_roi)
        if roi_debug_dir is not None:
            _save_debug_image(prepared_roi, detections_raw, roi_debug_dir / "primary_input.jpg", font_renderer)
        detections = _restore_detections_from_alignment(detections_raw, rotation_meta)
        mapped_detections = _map_to_image(detections, x1_i, y1_i, img_w, img_h, source="primary")

        secondary_mapped: List[Dict[str, Any]] = []
        if secondary_model is not None and patch_window is not None:
            patch_debug_dir = (roi_debug_dir / "patches") if roi_debug_dir else None
            patch_detections = _run_patch_detection(
                aligned_roi=aligned_roi,
                rotation_meta=rotation_meta,
                secondary_model=secondary_model,
                enhance_mode=enhance_mode,
                window_size=patch_window,
                overlap=patch_overlap,
                debug_dir=patch_debug_dir,
                font_renderer=font_renderer
            )
            secondary_mapped = _map_to_image(patch_detections, x1_i, y1_i, img_w, img_h, source="patch")

        merged_detections = mapped_detections + secondary_mapped
        if secondary_mapped:
            merged_detections = _apply_classwise_nms(merged_detections, fusion_iou)

        if vis_image is not None:
            for det in merged_detections:
                _draw_detection(vis_image, det["bbox"], det["class_name"],
                                det["confidence"], det["class_id"], font_renderer)

        roi_payload = {
            "roi_index": roi_idx,
            "bbox": [x1_i, y1_i, x2_i, y2_i],
            "num_detections": len(merged_detections),
            "detections": merged_detections
        }
        if secondary_mapped:
            roi_payload["primary_detections"] = len(mapped_detections)
            roi_payload["secondary_detections"] = len(secondary_mapped)
        roi_results.append(roi_payload)

    vis_path = None
    if visualize and vis_image is not None and visualization_dir is not None:
        visualization_dir.mkdir(parents=True, exist_ok=True)
        vis_file = visualization_dir / f"{image_path.stem}_det_viz.jpg"
        cv2.imwrite(str(vis_file), vis_image)
        vis_path = str(vis_file)

    return roi_results, vis_path


def _restore_detections_from_alignment(detections: List[Dict[str, Any]],
                                       rotation_meta: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not rotation_meta:
        return detections

    restored: List[Dict[str, Any]] = []
    for det in detections:
        new_det = det.copy()
        new_det["bbox"] = restore_bbox_from_rotation(det["bbox"], rotation_meta)
        restored.append(new_det)
    return restored


def resolve_label(class_id: int, label_map: Dict[int, str]) -> str:
    if class_id in label_map:
        return label_map[class_id]
    if (class_id + 1) in label_map:
        return label_map[class_id + 1]
    if (class_id - 1) in label_map:
        return label_map[class_id - 1]
    return f"class_{class_id}"


def _draw_detection(canvas: np.ndarray,
                    bbox: Sequence[float],
                    label: str,
                    score: float,
                    class_id: int,
                    font_renderer: Optional[FontRenderer]):
    draw_detection_instance(
        canvas,
        bbox=bbox,
        label=label,
        score=score,
        class_id=class_id,
        font_renderer=font_renderer
    )


def _run_patch_detection(aligned_roi: np.ndarray,
                         rotation_meta: Optional[Dict[str, Any]],
                         secondary_model: RFDetrDetectionModel,
                         enhance_mode: str,
                         window_size: Tuple[int, int],
                         overlap: float,
                         debug_dir: Optional[Path],
                         font_renderer: Optional[FontRenderer]) -> List[Dict[str, Any]]:
    if secondary_model is None or window_size is None:
        return []

    stride = calculate_stride(window_size, overlap)
    patches = sliding_window_crop(aligned_roi, window_size, stride)
    roi_h, roi_w = aligned_roi.shape[:2]
    detections: List[Dict[str, Any]] = []

    for patch_idx, patch_info in enumerate(patches):
        patch = patch_info['patch']
        px, py = patch_info['position']
        enhanced_patch = enhance_image(patch, mode=enhance_mode, output_bits=8)
        prepared_patch = ensure_color(enhanced_patch)
        patch_dets = secondary_model.predict_patch(prepared_patch)
        if debug_dir is not None:
            debug_dir.mkdir(parents=True, exist_ok=True)
            out_path = debug_dir / f"patch_{patch_idx:03d}.jpg"
            _save_debug_image(prepared_patch, patch_dets, out_path, font_renderer)
        for det in patch_dets:
            bbox = det["bbox"]
            offset_bbox = [
                float(np.clip(bbox[0] + px, 0, roi_w)),
                float(np.clip(bbox[1] + py, 0, roi_h)),
                float(np.clip(bbox[2] + px, 0, roi_w)),
                float(np.clip(bbox[3] + py, 0, roi_h))
            ]
            new_det = det.copy()
            new_det["bbox"] = offset_bbox
            detections.append(new_det)

    return _restore_detections_from_alignment(detections, rotation_meta)


def _map_to_image(detections: List[Dict[str, Any]],
                  offset_x: int,
                  offset_y: int,
                  img_w: int,
                  img_h: int,
                  source: str) -> List[Dict[str, Any]]:
    mapped: List[Dict[str, Any]] = []
    for det in detections:
        bbox = det["bbox"]
        mapped_bbox = [
            float(np.clip(bbox[0] + offset_x, 0, img_w)),
            float(np.clip(bbox[1] + offset_y, 0, img_h)),
            float(np.clip(bbox[2] + offset_x, 0, img_w)),
            float(np.clip(bbox[3] + offset_y, 0, img_h))
        ]
        mapped_det = {
            "class_id": det["class_id"],
            "class_name": det["class_name"],
            "confidence": det["confidence"],
            "bbox": mapped_bbox,
            "source": det.get("source", source)
        }
        mapped.append(mapped_det)
    return mapped


def _apply_classwise_nms(detections: List[Dict[str, Any]], iou_threshold: float) -> List[Dict[str, Any]]:
    if not detections:
        return []
    detections = sorted(detections, key=lambda d: d.get("confidence", 0.0), reverse=True)
    kept: List[Dict[str, Any]] = []
    for det in detections:
        suppressed = False
        for kept_det in kept:
            if det["class_id"] != kept_det["class_id"]:
                continue
            if _bbox_iou(det["bbox"], kept_det["bbox"]) >= iou_threshold:
                suppressed = True
                break
        if not suppressed:
            kept.append(det)
    return kept


def _bbox_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
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


def _save_debug_image(image: np.ndarray,
                      detections: List[Dict[str, Any]],
                      out_path: Path,
                      font_renderer: Optional[FontRenderer]):
    if not detections:
        debug_canvas = ensure_color(image.copy())
    else:
        debug_canvas = ensure_color(image.copy())
        for det in detections:
            bbox = det["bbox"]
            _draw_detection(debug_canvas, bbox, det.get("class_name", ""),
                            det.get("confidence", 0.0), det.get("class_id", 0), font_renderer)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), debug_canvas)
