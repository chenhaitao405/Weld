from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

from convert.pj.yolo_roi_extractor import WeldROIDetector
from rfdetr import RFDETRMedium, RFDETRLarge, RFDETRSegPreview
from utils import enhance_image, calculate_stride, sliding_window_crop
from utils.pipeline_utils import (
    FontRenderer,
    align_roi_orientation,
    ensure_color,
    restore_bbox_from_rotation,
    restore_polygon_from_rotation,
    draw_detection_instance
)
from utils.wide_slice_utils import (
    WideSliceParams,
    WideSlicePlan,
    WideSlicePatch,
    build_wide_slice_plan,
    resize_slice_to_square,
    stack_wide_slice_pair
)


def _load_rfdet_model_kwargs(model_path: Path) -> Dict[str, Any]:
    """Extract inference-relevant kwargs stored in the RF-DETR checkpoint."""
    try:
        import torch
    except ImportError:
        return {}

    if not model_path.exists():
        return {}
    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    except Exception:
        return {}

    args = checkpoint.get("args")
    if args is None:
        return {}

    allowed_fields = {
        "encoder",
        "out_feature_indexes",
        "dec_layers",
        "two_stage",
        "projector_scale",
        "hidden_dim",
        "patch_size",
        "num_windows",
        "sa_nheads",
        "ca_nheads",
        "dec_n_points",
        "bbox_reparam",
        "lite_refpoint_refine",
        "layer_norm",
        "amp",
        "num_classes",
        "resolution",
        "group_detr",
        "gradient_checkpointing",
        "positional_encoding_size",
        "ia_bce_loss",
        "cls_loss_coef",
        "segmentation_head",
        "mask_downsample_ratio",
        "num_queries",
        "num_select"
    }
    extracted: Dict[str, Any] = {}
    for field in allowed_fields:
        if hasattr(args, field):
            value = getattr(args, field)
            if value is not None:
                extracted[field] = value
    return extracted


def _ensure_single_prediction(detections: Any):
    if detections is None:
        return None
    if isinstance(detections, list):
        if len(detections) == 0:
            return None
        if len(detections) == 1:
            return detections[0]
        raise RuntimeError("RF-DETR返回了批量结果，请逐个调用predict")
    return detections


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
        checkpoint_kwargs = _load_rfdet_model_kwargs(self.model_path)
        if checkpoint_kwargs:
            checkpoint_kwargs.pop("pretrain_weights", None)
            checkpoint_kwargs.pop("device", None)
            model_kwargs.update(checkpoint_kwargs)
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
        detections = _ensure_single_prediction(detections)
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


class RFDetrSegmentationModel:
    """RF-DETR SegPreview 分割推理包装器"""

    def __init__(self,
                 model_path: str,
                 confidence: float = 0.25,
                 device: Optional[str] = None,
                 class_names: Optional[Sequence[str]] = None):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"未找到RF-DETR分割权重: {self.model_path}")
        self.confidence = confidence
        kwargs: Dict[str, Any] = {"pretrain_weights": str(self.model_path)}
        if device:
            kwargs["device"] = device
        self.model = RFDETRSegPreview(**kwargs)
        self.class_map = self._build_class_map(class_names)

    def _build_class_map(self, class_names: Optional[Sequence[str]]) -> Dict[int, str]:
        if class_names:
            return {idx: str(name) for idx, name in enumerate(class_names)}
        raw_names = getattr(self.model, "class_names", None)
        if isinstance(raw_names, dict):
            return {int(k): str(v) for k, v in raw_names.items()}
        if isinstance(raw_names, (list, tuple)):
            return {idx: str(name) for idx, name in enumerate(raw_names)}
        return {}

    def infer(self,
              image_bgr: np.ndarray,
              rotation_meta: Optional[Dict[str, Any]],
              visualize: bool,
              font_renderer: Optional[FontRenderer]) -> Tuple[List[Dict[str, Any]], Optional[np.ndarray]]:
        pil_image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        detections = self.model.predict(pil_image, threshold=self.confidence)
        detections = _ensure_single_prediction(detections)
        if detections is None or len(getattr(detections, "xyxy", [])) == 0:
            canvas = ensure_color(image_bgr.copy()) if visualize else None
            return [], _restore_canvas_from_rotation(canvas, rotation_meta)

        vis_canvas = ensure_color(image_bgr.copy()) if visualize else None
        masks = getattr(detections, "mask", None)
        defects: List[Dict[str, Any]] = []

        for idx, (bbox, score, cls_id) in enumerate(zip(
            detections.xyxy, detections.confidence, detections.class_id
        )):
            bbox_list = [float(v) for v in bbox.tolist()]
            cls_int = int(cls_id)
            polygon: List[List[float]] = []
            if masks is not None and idx < len(masks):
                polygon = _mask_to_polygon(masks[idx])
            if not polygon:
                polygon = _bbox_to_polygon(bbox_list)
            defect = {
                "class_id": cls_int,
                "class_name": resolve_label(cls_int, self.class_map),
                "confidence": float(score),
                "bbox": bbox_list,
                "polygon": polygon
            }
            defects.append(defect)

            if vis_canvas is not None:
                draw_detection_instance(
                    vis_canvas,
                    bbox=bbox_list,
                    label=defect["class_name"],
                    score=float(score),
                    class_id=cls_int,
                    font_renderer=font_renderer,
                    polygon=polygon
                )

        return defects, _restore_canvas_from_rotation(vis_canvas, rotation_meta)


def run_rfdet_segmentation(image: np.ndarray,
                           image_path: Path,
                           seg_model: RFDetrSegmentationModel,
                           visualize: bool,
                           visualization_dir: Optional[Path],
                           font_renderer: Optional[FontRenderer],
                           roi_bbox: Optional[Sequence[int]] = None,
                           rotation_meta: Optional[Dict[str, Any]] = None) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    defects, vis_canvas = seg_model.infer(image, rotation_meta=rotation_meta, visualize=visualize, font_renderer=font_renderer)
    h, w = image.shape[:2]
    bbox = roi_bbox if roi_bbox is not None else [0, 0, w, h]
    rois = [{
        "roi_index": 0,
        "bbox": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
        "num_defects": len(defects),
        "defects": defects
    }]

    vis_path = None
    if vis_canvas is not None and visualize and visualization_dir is not None:
        visualization_dir.mkdir(parents=True, exist_ok=True)
        vis_file = visualization_dir / f"{image_path.stem}_seg.jpg"
        cv2.imwrite(str(vis_file), vis_canvas)
        vis_path = str(vis_file)

    return rois, vis_path


def process_roi_and_segmentation(
        image: np.ndarray,
        image_path: Path,
        roi_detector: WeldROIDetector,
        seg_model: RFDetrSegmentationModel,
        enhance_mode: str,
        visualize: bool,
        visualization_dir: Optional[Path],
        font_renderer: Optional[FontRenderer],
        secondary_model: Optional[RFDetrSegmentationModel] = None,
        patch_window: Optional[Tuple[int, int]] = None,
        patch_overlap: float = 0.2,
        fusion_iou: float = 0.5,
        debug_dir: Optional[Path] = None) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    img_h, img_w = image.shape[:2]
    roi_boxes = roi_detector.detect_with_padding(image)
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
        primary_defects, _ = seg_model.infer(enhanced_roi, rotation_meta=rotation_meta,
                                             visualize=False, font_renderer=None)
        if roi_debug_dir is not None:
            _save_debug_image(ensure_color(enhanced_roi), primary_defects,
                              roi_debug_dir / "primary_input.jpg", font_renderer)
        primary_defects = _restore_segments_from_alignment(primary_defects, rotation_meta)
        mapped_primary = _map_to_image(primary_defects, x1_i, y1_i, img_w, img_h, source="primary")

        secondary_mapped: List[Dict[str, Any]] = []
        if secondary_model is not None and patch_window is not None:
            patch_debug_dir = (roi_debug_dir / "patches") if roi_debug_dir else None
            patch_defects = _run_patch_segmentation(
                aligned_roi=aligned_roi,
                rotation_meta=rotation_meta,
                secondary_model=secondary_model,
                enhance_mode=enhance_mode,
                window_size=patch_window,
                overlap=patch_overlap,
                debug_dir=patch_debug_dir,
                font_renderer=font_renderer
            )
            secondary_mapped = _map_to_image(patch_defects, x1_i, y1_i, img_w, img_h, source="patch")

        merged_defects = mapped_primary + secondary_mapped
        if secondary_mapped:
            merged_defects = _apply_classwise_nms(merged_defects, fusion_iou)

        if vis_image is not None:
            for det in merged_defects:
                _draw_detection(vis_image, det, font_renderer)

        roi_payload: Dict[str, Any] = {
            "roi_index": roi_idx,
            "bbox": [x1_i, y1_i, x2_i, y2_i],
            "num_defects": len(merged_defects),
            "defects": merged_defects
        }
        if secondary_mapped:
            roi_payload["primary_defects"] = len(mapped_primary)
            roi_payload["secondary_defects"] = len(secondary_mapped)

        roi_results.append(roi_payload)

    vis_path = None
    if visualize and vis_image is not None and visualization_dir is not None:
        visualization_dir.mkdir(parents=True, exist_ok=True)
        vis_file = visualization_dir / f"{image_path.stem}_seg.jpg"
        cv2.imwrite(str(vis_file), vis_image)
        vis_path = str(vis_file)

    return roi_results, vis_path


@dataclass
class RFDetrDetectionConfig:
    roi_detector: Optional[WeldROIDetector]
    detection_model: RFDetrDetectionModel
    secondary_model: Optional[RFDetrDetectionModel] = None
    enhance_mode: str = "windowing"
    patch_window: Optional[Tuple[int, int]] = None
    patch_overlap: float = 0.2
    fusion_iou: float = 0.5
    visualize: bool = False
    visualization_dir: Optional[Path] = None
    font_renderer: Optional[FontRenderer] = None
    debug_root: Optional[Path] = None
    wide_slice: Optional["WideSliceConfig"] = None


@dataclass
class WideSliceConfig:
    enabled: bool = False
    aspect_ratio_threshold: float = 3.0
    window_ratio: float = 2.0
    overlap: float = 0.3
    target_size: int = 1120

    def to_params(self) -> WideSliceParams:
        return WideSliceParams(
            aspect_ratio_threshold=self.aspect_ratio_threshold,
            window_ratio=self.window_ratio,
            overlap=self.overlap,
            target_size=self.target_size
        )


def run_rfdet_detection(image: np.ndarray,
                        image_path: Path,
                        config: RFDetrDetectionConfig) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    return _process_roi_and_detection(
        image=image,
        image_path=image_path,
        roi_detector=config.roi_detector,
        detection_model=config.detection_model,
        secondary_model=config.secondary_model,
        enhance_mode=config.enhance_mode,
        patch_window=config.patch_window,
        patch_overlap=config.patch_overlap,
        fusion_iou=config.fusion_iou,
        visualize=config.visualize,
        visualization_dir=config.visualization_dir,
        font_renderer=config.font_renderer,
        debug_dir=config.debug_root,
        wide_slice=config.wide_slice
    )


def _process_roi_and_detection(
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
        debug_dir: Optional[Path],
        wide_slice: Optional[WideSliceConfig] = None) -> Tuple[List[Dict[str, Any]], Optional[str]]:
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

        wide_slice_mapped: List[Dict[str, Any]] = []
        if wide_slice and wide_slice.enabled:
            wide_debug_dir = (roi_debug_dir / "wideslice") if roi_debug_dir else None
            extra_detections = _run_wide_slice_detection(
                aligned_roi=aligned_roi,
                detection_model=detection_model,
                slice_cfg=wide_slice,
                enhance_mode=enhance_mode,
                debug_dir=wide_debug_dir,
                font_renderer=font_renderer
            )
            if extra_detections:
                restored_extra = _restore_detections_from_alignment(extra_detections, rotation_meta)
                wide_slice_mapped = _map_to_image(restored_extra, x1_i, y1_i, img_w, img_h, source="wide_slice")

        merged_detections = mapped_detections + secondary_mapped + wide_slice_mapped
        if secondary_mapped or wide_slice_mapped:
            merged_detections = _apply_classwise_nms(merged_detections, fusion_iou)

        if vis_image is not None:
            for det in merged_detections:
                _draw_detection(vis_image, det, font_renderer)

        roi_payload = {
            "roi_index": roi_idx,
            "bbox": [x1_i, y1_i, x2_i, y2_i],
            "num_detections": len(merged_detections),
            "detections": merged_detections
        }
        if secondary_mapped or wide_slice_mapped:
            roi_payload["primary_detections"] = len(mapped_detections)
        if secondary_mapped:
            roi_payload["secondary_detections"] = len(secondary_mapped)
        if wide_slice_mapped:
            roi_payload["wide_slice_detections"] = len(wide_slice_mapped)
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


def _restore_segments_from_alignment(defects: List[Dict[str, Any]],
                                     rotation_meta: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not rotation_meta:
        return defects
    restored: List[Dict[str, Any]] = []
    for defect in defects:
        new_det = defect.copy()
        if "bbox" in defect:
            new_det["bbox"] = restore_bbox_from_rotation(defect["bbox"], rotation_meta)
        polygon = defect.get("polygon")
        if polygon:
            normalized = _normalize_polygon(polygon)
            if normalized:
                new_det["polygon"] = restore_polygon_from_rotation(normalized, rotation_meta)
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
                    detection: Dict[str, Any],
                    font_renderer: Optional[FontRenderer]):
    draw_detection_instance(
        canvas,
        bbox=detection["bbox"],
        label=detection["class_name"],
        score=detection.get("confidence"),
        class_id=detection.get("class_id", 0),
        font_renderer=font_renderer,
        polygon=detection.get("polygon")
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


def _run_wide_slice_detection(aligned_roi: np.ndarray,
                              detection_model: RFDetrDetectionModel,
                              slice_cfg: WideSliceConfig,
                              enhance_mode: str,
                              debug_dir: Optional[Path],
                              font_renderer: Optional[FontRenderer]) -> List[Dict[str, Any]]:
    if not slice_cfg.enabled:
        return []

    params = slice_cfg.to_params()
    target_size = max(1, int(params.target_size))
    plan: WideSlicePlan = build_wide_slice_plan(aligned_roi, params)
    if not plan.patches:
        return []

    for patch in plan.patches:
        patch.image = enhance_image(patch.image, mode=enhance_mode, output_bits=8)

    detections: List[Dict[str, Any]] = []
    slice_idx = 0

    if not plan.is_wide:
        solo_patch = plan.patches[0]
        detections.extend(
            _detect_resized_slice(
                patch=solo_patch,
                detection_model=detection_model,
                target_size=target_size,
                slice_index=slice_idx,
                debug_dir=debug_dir,
                font_renderer=font_renderer
            )
        )
        return detections

    pending_full: Optional[WideSlicePatch] = None
    for patch in plan.patches:
        if patch.is_full_window:
            if pending_full is None:
                pending_full = patch
            else:
                detections.extend(
                    _detect_wide_pair_slice(
                        top_patch=pending_full,
                        bottom_patch=patch,
                        detection_model=detection_model,
                        target_size=target_size,
                        roi_height=plan.roi_height,
                        roi_width=plan.roi_width,
                        slice_index=slice_idx,
                        debug_dir=debug_dir,
                        font_renderer=font_renderer
                    )
                )
                slice_idx += 1
                pending_full = None
        else:
            detections.extend(
                _detect_resized_slice(
                    patch=patch,
                    detection_model=detection_model,
                    target_size=target_size,
                    slice_index=slice_idx,
                    debug_dir=debug_dir,
                    font_renderer=font_renderer
                )
            )
            slice_idx += 1

    if pending_full is not None:
        detections.extend(
            _detect_resized_slice(
                patch=pending_full,
                detection_model=detection_model,
                target_size=target_size,
                slice_index=slice_idx,
                debug_dir=debug_dir,
                font_renderer=font_renderer
            )
        )
        slice_idx += 1

    return detections


def _detect_resized_slice(patch: WideSlicePatch,
                          detection_model: RFDetrDetectionModel,
                          target_size: int,
                          slice_index: int,
                          debug_dir: Optional[Path],
                          font_renderer: Optional[FontRenderer]) -> List[Dict[str, Any]]:
    if patch.width <= 0 or patch.height <= 0:
        return []
    resized = resize_slice_to_square(patch.image, target_size)
    prepared = ensure_color(resized)
    raw = detection_model.predict_patch(prepared)
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        det_path = debug_dir / f"wideslice_{slice_index:04d}_det.jpg"
        _save_debug_image(prepared, raw or [], det_path, font_renderer)
    if not raw:
        return []
    scale_x = patch.width / target_size
    scale_y = patch.height / target_size
    detections: List[Dict[str, Any]] = []

    for det in raw:
        bbox = det["bbox"]
        new_det = det.copy()
        new_det["bbox"] = [
            float(bbox[0] * scale_x + patch.x_offset),
            float(bbox[1] * scale_y),
            float(bbox[2] * scale_x + patch.x_offset),
            float(bbox[3] * scale_y)
        ]
        new_det["source"] = "wide_slice"
        detections.append(new_det)
    return detections


def _detect_wide_pair_slice(top_patch: WideSlicePatch,
                            bottom_patch: WideSlicePatch,
                            detection_model: RFDetrDetectionModel,
                            target_size: int,
                            roi_height: int,
                            roi_width: int,
                            slice_index: int,
                            debug_dir: Optional[Path],
                            font_renderer: Optional[FontRenderer]) -> List[Dict[str, Any]]:
    stacked = stack_wide_slice_pair(top_patch, bottom_patch)
    resized = resize_slice_to_square(stacked, target_size)
    prepared = ensure_color(resized)
    raw = detection_model.predict_patch(prepared)
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        det_path = debug_dir / f"wideslice_{slice_index:04d}_det.jpg"
        _save_debug_image(prepared, raw or [], det_path, font_renderer)
    if not raw:
        return []

    window_w = top_patch.width
    scale_x = window_w / target_size
    scale_y = (2 * roi_height) / target_size
    detections: List[Dict[str, Any]] = []
    for det in raw:
        bbox = det["bbox"]
        x1 = bbox[0] * scale_x
        y1 = bbox[1] * scale_y
        x2 = bbox[2] * scale_x
        y2 = bbox[3] * scale_y
        center_y = (y1 + y2) / 2.0
        if center_y < roi_height:
            x_offset = top_patch.x_offset
            local_y1 = y1
            local_y2 = y2
        else:
            x_offset = bottom_patch.x_offset
            local_y1 = y1 - roi_height
            local_y2 = y2 - roi_height
        mapped_bbox = [
            float(np.clip(x1 + x_offset, 0, roi_width)),
            float(np.clip(local_y1, 0, roi_height)),
            float(np.clip(x2 + x_offset, 0, roi_width)),
            float(np.clip(local_y2, 0, roi_height))
        ]
        new_det = det.copy()
        new_det["bbox"] = mapped_bbox
        new_det["source"] = "wide_slice"
        detections.append(new_det)
    return detections


def _run_patch_segmentation(aligned_roi: np.ndarray,
                            rotation_meta: Optional[Dict[str, Any]],
                            secondary_model: RFDetrSegmentationModel,
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
        defects, _ = secondary_model.infer(prepared_patch, rotation_meta=None,
                                           visualize=False, font_renderer=None)
        if debug_dir is not None:
            debug_dir.mkdir(parents=True, exist_ok=True)
            out_path = debug_dir / f"patch_{patch_idx:03d}.jpg"
            _save_debug_image(prepared_patch, defects, out_path, font_renderer)

        for defect in defects:
            bbox = defect.get("bbox")
            if bbox is None or len(bbox) != 4:
                continue
            offset_bbox = [
                float(np.clip(bbox[0] + px, 0, roi_w)),
                float(np.clip(bbox[1] + py, 0, roi_h)),
                float(np.clip(bbox[2] + px, 0, roi_w)),
                float(np.clip(bbox[3] + py, 0, roi_h))
            ]
            new_defect = defect.copy()
            new_defect["bbox"] = offset_bbox
            polygon = defect.get("polygon")
            if polygon:
                new_defect["polygon"] = _offset_polygon(polygon, px, py, roi_w, roi_h)
            new_defect["source"] = "patch"
            detections.append(new_defect)

    return _restore_segments_from_alignment(detections, rotation_meta)


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
        polygon = det.get("polygon")
        mapped_polygon = _offset_polygon(polygon, offset_x, offset_y, img_w, img_h) if polygon else None
        mapped_det = {
            "class_id": det["class_id"],
            "class_name": det["class_name"],
            "confidence": det["confidence"],
            "bbox": mapped_bbox,
            "source": det.get("source", source)
        }
        if mapped_polygon:
            mapped_det["polygon"] = mapped_polygon
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


def _bbox_to_polygon(bbox: Sequence[float]) -> List[List[float]]:
    if not bbox or len(bbox) != 4:
        return []
    x1, y1, x2, y2 = bbox
    return [
        [float(x1), float(y1)],
        [float(x2), float(y1)],
        [float(x2), float(y2)],
        [float(x1), float(y2)]
    ]


def _mask_to_polygon(mask: np.ndarray, min_area: float = 10.0) -> List[List[float]]:
    if mask is None:
        return []
    binary = (mask.astype(np.float32) > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < min_area:
        return []
    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    pts = approx.reshape(-1, 2) if approx.size else np.zeros((0, 2), dtype=np.float32)
    return _normalize_polygon(pts)


def _restore_canvas_from_rotation(canvas: Optional[np.ndarray],
                                  rotation_meta: Optional[Dict[str, Any]]) -> Optional[np.ndarray]:
    if canvas is None or not rotation_meta or rotation_meta.get("rotation") != "ccw90":
        return canvas
    return cv2.rotate(canvas, cv2.ROTATE_90_CLOCKWISE)


def _save_debug_image(image: np.ndarray,
                      detections: List[Dict[str, Any]],
                      out_path: Path,
                      font_renderer: Optional[FontRenderer]):
    if not detections:
        debug_canvas = ensure_color(image.copy())
    else:
        debug_canvas = ensure_color(image.copy())
        for det in detections:
            _draw_detection(debug_canvas, det, font_renderer)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), debug_canvas)


def _offset_polygon(polygon: Sequence[Sequence[float]],
                    offset_x: int,
                    offset_y: int,
                    img_w: int,
                    img_h: int) -> List[List[float]]:
    mapped: List[List[float]] = []
    for pt in polygon:
        if not isinstance(pt, (list, tuple, np.ndarray)) or len(pt) < 2:
            continue
        mapped.append([
            float(np.clip(pt[0] + offset_x, 0, img_w)),
            float(np.clip(pt[1] + offset_y, 0, img_h))
        ])
    return mapped


def _normalize_polygon(points: Sequence[Sequence[float]]) -> List[List[float]]:
    normalized: List[List[float]] = []
    for pt in points:
        if not isinstance(pt, (list, tuple, np.ndarray)) or len(pt) < 2:
            continue
        normalized.append([float(pt[0]), float(pt[1])])
    return normalized
