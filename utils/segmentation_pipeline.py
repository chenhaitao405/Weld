from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from convert.pj.yolo_roi_extractor import WeldROIDetector
from utils import enhance_image
from utils.pipeline_utils import (
    FontRenderer,
    align_roi_orientation,
    ensure_color,
    prepare_seg_input,
    restore_bbox_from_rotation,
    restore_polygon_from_rotation,
    draw_detection_instance,
)


def process_roi_and_segmentation(
        image: np.ndarray,
        image_path: Path,
        roi_detector: WeldROIDetector,
        seg_model: YOLO,
        class_names: Sequence[str],
        enhance_mode: str,
        seg_conf: float,
        imgsz: int,
        device: Optional[str],
        visualize: bool,
        visualization_dir: Optional[Path],
        font_renderer: Optional[FontRenderer]) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    运行ROI检测 + YOLO分割推理，并可选保存可视化结果
    """
    img_height, img_width = image.shape[:2]
    roi_boxes = roi_detector.detect_with_padding(image)
    if not roi_boxes:
        roi_boxes = [(0, 0, img_width, img_height)]

    vis_image = ensure_color(image.copy()) if visualize else None
    roi_results: List[Dict[str, Any]] = []

    for roi_idx, (x1, y1, x2, y2) in enumerate(roi_boxes):
        roi_patch = image[y1:y2, x1:x2]
        if roi_patch.size == 0:
            continue

        aligned_roi, rotation_meta = align_roi_orientation(roi_patch)
        enhanced_roi = enhance_image(aligned_roi, mode=enhance_mode)
        seg_input = prepare_seg_input(enhanced_roi)

        seg_outputs = seg_model.predict(
            source=seg_input,
            conf=seg_conf,
            imgsz=imgsz,
            device=device,
            verbose=False
        )

        defects = _extract_defects(
            seg_outputs=seg_outputs,
            offset_x=x1,
            offset_y=y1,
            class_names=class_names,
            vis_image=vis_image,
            font_renderer=font_renderer,
            rotation_meta=rotation_meta
        )

        roi_results.append({
            "roi_index": roi_idx,
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "num_defects": len(defects),
            "defects": defects
        })

    vis_path = None
    if visualize and vis_image is not None and visualization_dir is not None:
        visualization_dir.mkdir(parents=True, exist_ok=True)
        vis_file = visualization_dir / f"{image_path.stem}_viz.jpg"
        cv2.imwrite(str(vis_file), vis_image)
        vis_path = str(vis_file)

    return roi_results, vis_path


def _extract_defects(seg_outputs: Sequence[Any],
                     offset_x: int,
                     offset_y: int,
                     class_names: Sequence[str],
                     vis_image: Optional[np.ndarray],
                     font_renderer: Optional[FontRenderer],
                     rotation_meta: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """解析分割输出，并可选绘制到可视化画布"""
    defects: List[Dict[str, Any]] = []

    for result in seg_outputs:
        if result.boxes is None or len(result.boxes) == 0:
            continue

        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        polygons = result.masks.xy if result.masks is not None else [None] * len(boxes)

        for idx, box in enumerate(boxes):
            class_id = int(classes[idx])
            confidence = float(confidences[idx])
            bbox_local = [float(box[0]), float(box[1]), float(box[2]), float(box[3])]
            bbox_local = restore_bbox_from_rotation(bbox_local, rotation_meta)
            bbox = [
                float(bbox_local[0] + offset_x),
                float(bbox_local[1] + offset_y),
                float(bbox_local[2] + offset_x),
                float(bbox_local[3] + offset_y)
            ]

            polygon_points: List[List[float]] = []
            if polygons and idx < len(polygons) and polygons[idx] is not None:
                restored_polygon = restore_polygon_from_rotation(polygons[idx], rotation_meta)
                polygon_points = [
                    [float(pt[0] + offset_x), float(pt[1] + offset_y)]
                    for pt in restored_polygon
                ]

            class_name = _class_name(class_id, class_names)
            defect = {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "bbox": bbox,
                "polygon": polygon_points
            }
            defects.append(defect)

            if vis_image is not None:
                _draw_visualization(
                    vis_image,
                    bbox=bbox,
                    polygon=polygon_points,
                    label=class_name,
                    score=confidence,
                    class_id=class_id,
                    font_renderer=font_renderer
                )

    return defects


def _draw_visualization(canvas: np.ndarray,
                        bbox: List[float],
                        polygon: List[List[float]],
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
        font_renderer=font_renderer,
        polygon=polygon
    )


def _class_name(class_id: int, class_names: Sequence[str]) -> str:
    if class_names and 0 <= class_id < len(class_names):
        return str(class_names[class_id])
    return str(class_id)
