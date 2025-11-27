import cv2
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
from ultralytics import YOLO
import matplotlib.cm as cm

from convert.pj.yolo_roi_extractor import WeldROIDetector
from utils.image_processing import (
    enhance_image, sliding_window_crop, calculate_stride
)
from utils.pipeline_utils import ensure_color

DEFAULT_OVERLAP_RATIO = 0.5
DEFAULT_WINDOW_SIZE = 320
DEFAULT_CONFIDENCE_THRESHOLD = 0.9
DEFAULT_ALPHA = 0.25
DEFAULT_DEFECT_CLASSES = (0,)


class SliceClassificationPipeline:
    """基于滑动窗口的切片分类与热力图生成管线"""

    def __init__(self,
                 model_path: str,
                 window_size: Tuple[int, int] = (DEFAULT_WINDOW_SIZE, DEFAULT_WINDOW_SIZE),
                 overlap_ratio: float = DEFAULT_OVERLAP_RATIO,
                 enhance_mode: str = 'original',
                 confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
                 use_confidence_weight: bool = False,
                 colormap: str = 'hot',
                 alpha: float = DEFAULT_ALPHA,
                 display_mode: str = 'overlay',
                 defect_class_ids: Optional[List[int]] = None,
                 roi_detector: Optional[WeldROIDetector] = None):
        self.model = YOLO(model_path)
        self.window_size = window_size
        self.overlap_ratio = overlap_ratio
        self.enhance_mode = enhance_mode
        self.confidence_threshold = confidence_threshold
        self.use_confidence_weight = use_confidence_weight
        self.colormap = colormap
        self.alpha = alpha
        self.display_mode = display_mode
        self.defect_class_ids = set(
            defect_class_ids if defect_class_ids is not None else DEFAULT_DEFECT_CLASSES
        )
        self.stride = calculate_stride(window_size, overlap_ratio)
        self.cmap = cm.get_cmap(colormap)
        self.roi_detector = roi_detector

    def detect_roi_boxes(self, image: np.ndarray,
                         image_id: Optional[str] = None) -> List[Tuple[int, int, int, int]]:
        """检测焊缝ROI，若失败则回退到整图"""
        height, width = image.shape[:2]
        fallback_box = [(0, 0, width, height)]

        if self.roi_detector is None:
            return fallback_box

        image_for_roi = ensure_color(image)
        boxes = self.roi_detector.detect_with_padding(image_for_roi, (height, width))
        if not boxes:
            label = image_id if image_id else "当前图像"
            print(f"  - ROI检测未发现区域（{label}），使用整图作为ROI")
            return fallback_box

        return boxes

    def prepare_patch_for_classification(self, patch: np.ndarray) -> np.ndarray:
        enhanced_patch = enhance_image(patch, mode=self.enhance_mode, output_bits=8)
        if len(enhanced_patch.shape) == 2:
            enhanced_patch = cv2.cvtColor(enhanced_patch, cv2.COLOR_GRAY2BGR)
        return np.ascontiguousarray(enhanced_patch)

    def predict_single_patch(self, patch: np.ndarray) -> Tuple[int, float]:
        results = self.model.predict(patch, verbose=False)
        if results and len(results) > 0:
            res = results[0]
            if hasattr(res, 'probs') and res.probs is not None:
                top1 = res.probs.top1
                top1conf = res.probs.top1conf.cpu().numpy()
                return int(top1), float(top1conf)
        return -1, 0.0

    def detect_slice_classify(self,
                              image: np.ndarray,
                              image_id: Optional[str] = None) -> Dict[str, Any]:
        """封装流程：ROI检测 -> 切片增强 -> 分类 -> 映射回原图"""
        roi_boxes = self.detect_roi_boxes(image, image_id=image_id)

        pipeline_result: Dict[str, Any] = {
            'image_id': image_id,
            'image_shape': image.shape[:2],
            'rois': [],
            'patch_predictions': [],
            'total_patches': 0,
            'defect_patches': 0
        }

        progress_bar = None
        try:
            for roi_index, (x1, y1, x2, y2) in enumerate(roi_boxes):
                roi_width = max(0, x2 - x1)
                roi_height = max(0, y2 - y1)
                if roi_width <= 0 or roi_height <= 0:
                    continue

                roi_info: Dict[str, Any] = {
                    'roi_index': roi_index,
                    'box': {
                        'x1': int(x1), 'y1': int(y1),
                        'x2': int(x2), 'y2': int(y2),
                        'width': int(roi_width), 'height': int(roi_height)
                    },
                    'num_patches': 0,
                    'defect_patches': 0,
                    'patches': []
                }
                pipeline_result['rois'].append(roi_info)

                roi_image = image[y1:y2, x1:x2]
                if roi_image.size == 0:
                    continue

                patches = sliding_window_crop(roi_image, self.window_size, self.stride)
                if not patches:
                    continue

                for patch_info in patches:
                    if progress_bar is None:
                        progress_bar = tqdm(desc="分类切片", unit="patch", leave=False)

                    prepared_patch = self.prepare_patch_for_classification(patch_info['patch'])
                    pred_class, confidence = self.predict_single_patch(prepared_patch)

                    is_defect = (
                        pred_class in self.defect_class_ids and
                        confidence >= self.confidence_threshold
                    )

                    rel_x, rel_y = patch_info['position']
                    patch_width = patch_info['size'][1]
                    patch_height = patch_info['size'][0]
                    abs_x = int(x1 + rel_x)
                    abs_y = int(y1 + rel_y)

                    patch_result = {
                        'roi_index': roi_index,
                        'position': (abs_x, abs_y),
                        'size': (int(patch_width), int(patch_height)),
                        'relative_position': (int(rel_x), int(rel_y)),
                        'confidence': float(confidence),
                        'class': int(pred_class),
                        'is_defect': bool(is_defect)
                    }

                    pipeline_result['patch_predictions'].append(patch_result)
                    roi_info['patches'].append(patch_result)
                    roi_info['num_patches'] += 1
                    pipeline_result['total_patches'] += 1

                    if is_defect:
                        roi_info['defect_patches'] += 1
                        pipeline_result['defect_patches'] += 1

                    progress_bar.update(1)
        finally:
            if progress_bar is not None:
                progress_bar.close()

        return pipeline_result

    def generate_heatmap(self,
                         image_shape: Tuple[int, int],
                         predictions: List[Dict]) -> np.ndarray:
        height, width = image_shape[:2]
        heatmap = np.zeros((height, width), dtype=np.float32)

        for pred in predictions:
            is_defect = pred.get('is_defect')
            if is_defect is None:
                pred_class = pred.get('class', -1)
                pred_conf = pred.get('confidence', 0.0)
                is_defect = (
                    pred_class in self.defect_class_ids and
                    pred_conf >= self.confidence_threshold
                )

            if not is_defect:
                continue

            x, y = pred['position']
            w, h = pred['size']
            x_end = min(x + w, width)
            y_end = min(y + h, height)

            increment = pred.get('confidence', 0.0) if self.use_confidence_weight else 1.0
            heatmap[y:y_end, x:x_end] += increment

        return heatmap

    def apply_colormap(self, heatmap: np.ndarray) -> np.ndarray:
        if heatmap.max() > 0:
            normalized = heatmap / heatmap.max()
        else:
            normalized = heatmap
        colored = self.cmap(normalized)
        colored_bgr = (colored[:, :, [2, 1, 0]] * 255).astype(np.uint8)
        return colored_bgr

    def create_heatmap_overlay(self,
                               original_image: np.ndarray,
                               heatmap: np.ndarray,
                               apply_gaussian_blur: bool = True) -> np.ndarray:
        original_3ch = ensure_color(original_image)

        if self.display_mode == 'contour':
            return self._create_contour_overlay(original_3ch, heatmap)
        if self.display_mode == 'sparse':
            return self._create_sparse_overlay(original_3ch, heatmap, apply_gaussian_blur)
        return self._create_standard_overlay(original_3ch, heatmap, apply_gaussian_blur)

    def _create_standard_overlay(self, original_3ch, heatmap, apply_gaussian_blur):
        if apply_gaussian_blur and heatmap.max() > 0:
            kernel_size = max(5, min(31, int(min(heatmap.shape) * 0.02) | 1))
            heatmap_smooth = cv2.GaussianBlur(heatmap, (kernel_size, kernel_size), 0)
        else:
            heatmap_smooth = heatmap

        colored_heatmap = self.apply_colormap(heatmap_smooth)

        if heatmap_smooth.max() > 0:
            normalized_heat = heatmap_smooth / heatmap_smooth.max()
            alpha_mask = np.power(normalized_heat, 1.5) * self.alpha
        else:
            alpha_mask = np.zeros_like(heatmap_smooth, dtype=np.float32)

        alpha_mask = np.stack([alpha_mask] * 3, axis=-1)
        overlay = original_3ch.astype(np.float32) * (1 - alpha_mask) + \
            colored_heatmap.astype(np.float32) * alpha_mask
        return overlay.astype(np.uint8)

    def _create_contour_overlay(self, original_3ch, heatmap):
        overlay = original_3ch.copy()

        if heatmap.max() > 0:
            normalized = (heatmap / heatmap.max() * 255).astype(np.uint8)
            contour_levels = [30, 60, 90, 120, 150, 180, 210]
            colors = self.cmap(np.array(contour_levels) / 255.0)[:, :3]
            colors = (colors[:, [2, 1, 0]] * 255).astype(np.uint8)

            for i, level in enumerate(contour_levels):
                _, binary = cv2.threshold(normalized, level, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                color = colors[i].tolist()
                cv2.drawContours(overlay, contours, -1, color, 2)

        return overlay

    def _create_sparse_overlay(self, original_3ch, heatmap, apply_gaussian_blur):
        overlay = original_3ch.copy()

        if heatmap.max() > 0:
            if apply_gaussian_blur:
                kernel_size = max(5, min(31, int(min(heatmap.shape) * 0.02) | 1))
                heatmap_smooth = cv2.GaussianBlur(heatmap, (kernel_size, kernel_size), 0)
            else:
                heatmap_smooth = heatmap

            threshold = heatmap_smooth.max() * 0.3
            high_heat_mask = (heatmap_smooth > threshold).astype(np.float32)
            colored_heatmap = self.apply_colormap(heatmap_smooth)

            if heatmap_smooth.max() > threshold:
                alpha_values = (heatmap_smooth - threshold) / (heatmap_smooth.max() - threshold)
                alpha_mask = alpha_values * high_heat_mask * self.alpha * 1.5
                alpha_mask = np.clip(alpha_mask, 0, self.alpha)
            else:
                alpha_mask = high_heat_mask * self.alpha

            alpha_mask = np.stack([alpha_mask] * 3, axis=-1)
            overlay = original_3ch.astype(np.float32) * (1 - alpha_mask) + \
                colored_heatmap.astype(np.float32) * alpha_mask
            overlay = overlay.astype(np.uint8)

        return overlay

    def add_colorbar(self, image: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        colorbar_width = max(30, int(w * 0.03))
        colorbar_height = int(h * 0.6)
        colorbar_x = w + 20

        extended_width = w + colorbar_width + 40
        canvas = np.ones((h, extended_width, 3), dtype=np.uint8) * 255
        canvas[:, :w] = image

        colorbar_values = np.linspace(0, 1, colorbar_height).reshape(-1, 1)
        colorbar_values = np.repeat(colorbar_values, colorbar_width, axis=1)
        colorbar_colored = self.cmap(colorbar_values)
        colorbar_bgr = (colorbar_colored[:, :, [2, 1, 0]] * 255).astype(np.uint8)

        bar_y = (h - colorbar_height) // 2
        canvas[bar_y:bar_y+colorbar_height, colorbar_x:colorbar_x+colorbar_width] = colorbar_bgr

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1

        max_val = heatmap.max()
        cv2.putText(canvas, f"{max_val:.1f}",
                    (colorbar_x + colorbar_width + 5, bar_y + 10),
                    font, font_scale, (0, 0, 0), font_thickness)
        cv2.putText(canvas, "0.0",
                    (colorbar_x + colorbar_width + 5, bar_y + colorbar_height - 5),
                    font, font_scale, (0, 0, 0), font_thickness)
        cv2.putText(canvas, f"{max_val/2:.1f}",
                    (colorbar_x + colorbar_width + 5, bar_y + colorbar_height // 2),
                    font, font_scale, (0, 0, 0), font_thickness)

        return canvas

    @staticmethod
    def generate_statistics(heatmap: np.ndarray) -> Dict[str, float]:
        stats = {
            'max_value': float(heatmap.max()),
            'mean_value': float(heatmap.mean()),
            'std_value': float(heatmap.std()),
            'non_zero_pixels': int(np.sum(heatmap > 0)),
            'total_pixels': int(heatmap.size),
            'coverage_ratio': float(np.sum(heatmap > 0) / heatmap.size),
            'high_confidence_area': int(np.sum(heatmap > heatmap.max() * 0.7)) if heatmap.max() > 0 else 0
        }
        return stats
