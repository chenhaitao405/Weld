from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

from convert.pj.yolo_roi_extractor import WeldROIDetector
from rfdetr import RFDETRMedium
from utils import enhance_image
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
                 class_names: Optional[Sequence[str]] = None):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"未找到RF-DETR权重: {self.model_path}")
        self.confidence = confidence
        self.model = self._load_model(device)
        if optimize:
            self._optimize_model(optimize_batch, use_half)
        self.class_map = self._build_class_map(class_names)

    def _load_model(self, device: Optional[str]) -> RFDETRMedium:
        model_kwargs: Dict[str, Any] = {"pretrain_weights": str(self.model_path)}
        if device:
            model_kwargs["device"] = device
        return RFDETRMedium(**model_kwargs)

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
        enhance_mode: str,
        visualize: bool,
        visualization_dir: Optional[Path],
        font_renderer: Optional[FontRenderer]) -> Tuple[List[Dict[str, Any]], Optional[str]]:
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

        aligned_roi, rotation_meta = align_roi_orientation(roi_patch)
        enhanced_roi = enhance_image(aligned_roi, mode=enhance_mode, output_bits=8)
        prepared_roi = ensure_color(enhanced_roi)
        detections = detection_model.predict_patch(prepared_roi)
        detections = _restore_detections_from_alignment(detections, rotation_meta)

        mapped_detections: List[Dict[str, Any]] = []
        for det in detections:
            bbox = det["bbox"]
            mapped_bbox = [
                float(bbox[0] + x1_i),
                float(bbox[1] + y1_i),
                float(bbox[2] + x1_i),
                float(bbox[3] + y1_i)
            ]
            mapped_det = {
                "class_id": det["class_id"],
                "class_name": det["class_name"],
                "confidence": det["confidence"],
                "bbox": mapped_bbox
            }
            mapped_detections.append(mapped_det)

            if vis_image is not None:
                _draw_detection(vis_image, mapped_bbox, det["class_name"],
                                det["confidence"], det["class_id"], font_renderer)

        roi_results.append({
            "roi_index": roi_idx,
            "bbox": [x1_i, y1_i, x2_i, y2_i],
            "num_detections": len(mapped_detections),
            "detections": mapped_detections
        })

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
