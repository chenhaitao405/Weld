#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Weld defect inference pipeline.

功能:
    1. 批量读取指定文件夹中的原始焊缝图像
    2. 调用 convert/pj/yolo_roi_extractor.py 中的 ROI 检测模型定位焊缝区域
    3. 对ROI裁剪后的区域调用 convert/pj/patchandenhance.py 依赖的 enhance_image 函数做增强
    4. 使用YOLO分割模型进行缺陷识别，输出缺陷ID与坐标
    5. 将分割结果映射回原图坐标系
    6. 根据参数选择是否在原图上可视化绘制缺陷

注意:
    - 步骤2~5封装在 WeldInferencePipeline._process_roi_and_segmentation 中，便于后续拓展
    - 依赖 ultralytics >= 8.x
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

try:
    import matplotlib
    from matplotlib import font_manager
    MATPLOTLIB_AVAILABLE = True
    PREFERRED_FONT_FAMILIES = ['AR PL UKai CN', 'Noto Sans CJK JP', 'DejaVu Sans']
    matplotlib.rcParams['font.sans-serif'] = PREFERRED_FONT_FAMILIES
    matplotlib.rcParams['axes.unicode_minus'] = False
except ImportError:  # pragma: no cover
    matplotlib = None
    font_manager = None
    MATPLOTLIB_AVAILABLE = False
    PREFERRED_FONT_FAMILIES = []

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:  # pragma: no cover - Pillow optional
    Image = ImageDraw = ImageFont = None
    PIL_AVAILABLE = False

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from convert.pj.yolo_roi_extractor import WeldROIDetector  # noqa: E402
from utils import enhance_image, read_dataset_yaml  # noqa: E402


SUPPORTED_IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
COLOR_PALETTE = [
    (255, 99, 71),
    (102, 204, 255),
    (144, 238, 144),
    (255, 215, 0),
    (226, 110, 168),
    (0, 191, 255),
    (255, 140, 0),
    (138, 43, 226),
    (60, 179, 113),
    (255, 20, 147),
]


def _load_class_names(data_config: Optional[str], seg_model: YOLO) -> List[str]:
    """优先从dataset.yaml读取类别名称，否则回退到模型自带names"""
    names: List[str] = []
    if data_config:
        yaml_path = Path(data_config)
        if yaml_path.exists():
            cfg = read_dataset_yaml(str(yaml_path))
            yaml_names = cfg.get('names')
            if isinstance(yaml_names, dict):
                sorted_items = sorted(yaml_names.items(), key=lambda kv: int(kv[0]))
                names = [item[1] for item in sorted_items]
            elif isinstance(yaml_names, list):
                names = yaml_names
    if not names and hasattr(seg_model, 'names'):
        # YOLO的names通常为dict
        model_names = seg_model.names
        if isinstance(model_names, dict):
            names = [model_names[i] for i in sorted(model_names.keys())]
        elif isinstance(model_names, (list, tuple)):
            names = list(model_names)
    return names


class WeldInferencePipeline:
    """焊缝缺陷推理流水线"""

    def __init__(self,
                 roi_weights: str,
                 seg_weights: str,
                 enhance_mode: str = 'windowing',
                 roi_conf: float = 0.25,
                 roi_iou: float = 0.45,
                 roi_padding: float = 0.1,
                 seg_conf: float = 0.25,
                 imgsz: int = 640,
                 device: Optional[str] = None,
                 data_config: Optional[str] = None,
                 font_path: Optional[str] = None,
                 font_size: int = 20):
        self.enhance_mode = enhance_mode
        self.seg_conf = seg_conf
        self.imgsz = imgsz
        self.device = device
        self.font_path = font_path
        self.font_size = font_size

        self.roi_detector = WeldROIDetector(
            model_path=roi_weights,
            roi_conf_threshold=roi_conf,
            roi_iou_threshold=roi_iou,
            padding_ratio=roi_padding
        )

        print(f"加载分割模型: {seg_weights}")
        self.seg_model = YOLO(seg_weights)

        self.class_names = _load_class_names(data_config, self.seg_model)
        auto_font_path = font_path or self._auto_detect_font()
        self._pil_font = self._load_font(auto_font_path, font_size)

    def process_directory(self,
                          image_dir: Path,
                          output_dir: Path,
                          visualize: bool = False,
                          visualization_dir: Optional[Path] = None,
                          max_images: Optional[int] = None) -> List[Dict[str, Any]]:
        """批量处理目录，返回结果列表"""
        image_paths = sorted([
            p for p in image_dir.rglob('*')
            if p.suffix.lower() in SUPPORTED_IMAGE_EXTS and p.is_file()
        ])

        if max_images is not None:
            image_paths = image_paths[:max_images]

        if not image_paths:
            raise FileNotFoundError(f"未在 {image_dir} 中找到支持的图像文件")

        results: List[Dict[str, Any]] = []
        visualization_dir = visualization_dir or (output_dir / "visualizations")

        for image_path in tqdm(image_paths, desc="推理中"):
            try:
                result = self.process_image(
                    image_path=image_path,
                    visualize=visualize,
                    visualization_dir=visualization_dir if visualize else None
                )
                results.append(result)
            except Exception as exc:
                print(f"[警告] 处理 {image_path} 时出错: {exc}")

        return results

    def process_image(self,
                      image_path: Path,
                      visualize: bool = False,
                      visualization_dir: Optional[Path] = None) -> Dict[str, Any]:
        """处理单张图像"""
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        image = self._normalize_input_image(image)
        image = self._ensure_color(image)

        rois, vis_path = self._process_roi_and_segmentation(
            image=image,
            image_path=image_path,
            visualize=visualize,
            visualization_dir=visualization_dir
        )

        h, w = image.shape[:2]
        return {
            "image_path": str(image_path),
            "width": w,
            "height": h,
            "num_rois": len(rois),
            "rois": rois,
            "visualization": vis_path
        }

    def _process_roi_and_segmentation(self,
                                      image: np.ndarray,
                                      image_path: Path,
                                      visualize: bool,
                                      visualization_dir: Optional[Path]) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """
        封装步骤2~5:
            2. ROI检测
            3. ROI区域增强
            4. YOLO分割检测
            5. 分割结果映射回原图，并可视化
        """
        img_height, img_width = image.shape[:2]

        roi_boxes = self.roi_detector.detect_with_padding(image)
        if not roi_boxes:
            roi_boxes = [(0, 0, img_width, img_height)]

        vis_image = None
        if visualize:
            vis_image = self._ensure_color(image.copy())

        roi_results: List[Dict[str, Any]] = []
        for roi_idx, (x1, y1, x2, y2) in enumerate(roi_boxes):
            roi_patch = image[y1:y2, x1:x2]
            if roi_patch.size == 0:
                continue

            enhanced_roi = enhance_image(roi_patch, mode=self.enhance_mode)
            seg_input = self._prepare_seg_input(enhanced_roi)

            seg_outputs = self.seg_model.predict(
                source=seg_input,
                conf=self.seg_conf,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False
            )

            defects = self._extract_defects(
                seg_outputs,
                offset_x=x1,
                offset_y=y1,
                vis_image=vis_image
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

    def _prepare_seg_input(self, enhanced_roi: np.ndarray) -> np.ndarray:
        """确保分割输入为三通道uint8"""
        if enhanced_roi.dtype != np.uint8:
            enhanced_roi = cv2.normalize(enhanced_roi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        seg_input = enhanced_roi
        if seg_input.ndim == 2:
            seg_input = np.repeat(seg_input[:, :, None], 3, axis=2)
        elif seg_input.ndim == 3:
            channels = seg_input.shape[2]
            if channels == 1:
                seg_input = np.repeat(seg_input, 3, axis=2)
            elif channels == 4:
                seg_input = cv2.cvtColor(seg_input, cv2.COLOR_BGRA2BGR)
            elif channels > 3:
                seg_input = seg_input[:, :, :3]
        return np.ascontiguousarray(seg_input)

    def _extract_defects(self,
                         seg_outputs: Sequence[Any],
                         offset_x: int,
                         offset_y: int,
                         vis_image: Optional[np.ndarray]) -> List[Dict[str, Any]]:
        """解析YOLO分割输出，并回映射至原图坐标"""
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
                bbox = [
                    float(box[0] + offset_x),
                    float(box[1] + offset_y),
                    float(box[2] + offset_x),
                    float(box[3] + offset_y)
                ]

                polygon_points: List[List[float]] = []
                if polygons and idx < len(polygons) and polygons[idx] is not None:
                    polygon_points = [
                        [float(pt[0] + offset_x), float(pt[1] + offset_y)]
                        for pt in polygons[idx]
                    ]

                class_name = self._class_name(class_id)
                defects.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox": bbox,
                    "polygon": polygon_points
                })

                if vis_image is not None:
                    self._draw_visualization(
                        vis_image,
                        bbox=bbox,
                        polygon=polygon_points,
                        label=class_name,
                        score=confidence,
                        class_id=class_id
                    )

        return defects

    def _draw_visualization(self,
                            canvas: np.ndarray,
                            bbox: List[float],
                            polygon: List[List[float]],
                            label: str,
                            score: float,
                            class_id: int):
        """在原图上绘制可视化信息"""
        color = self._color_for_class(class_id)
        pt1 = (int(bbox[0]), int(bbox[1]))
        pt2 = (int(bbox[2]), int(bbox[3]))

        if polygon and len(polygon) >= 3:
            pts = np.array(polygon, dtype=np.int32)
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.3, canvas, 0.7, 0, canvas)
            cv2.polylines(canvas, [pts], True, color, 2)
        else:
            cv2.rectangle(canvas, pt1, pt2, color, 2)

        caption = f"{label}:{score:.2f}"
        text_org = (pt1[0], max(0, pt1[1] - 5))
        if self._should_use_pil_font():
            self._draw_text_with_pil(canvas, caption, text_org, color)
        else:
            cv2.putText(canvas, caption, text_org,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, lineType=cv2.LINE_AA)

    def _class_name(self, class_id: int) -> str:
        if self.class_names and 0 <= class_id < len(self.class_names):
            return str(self.class_names[class_id])
        return str(class_id)

    def _color_for_class(self, class_id: int) -> Tuple[int, int, int]:
        return COLOR_PALETTE[class_id % len(COLOR_PALETTE)]

    @staticmethod
    def _ensure_color(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.shape[2] == 1:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        if image.shape[2] > 3:
            return image[:, :, :3]
        return image

    def _load_font(self, font_path: Optional[str], font_size: int) -> Optional["ImageFont.FreeTypeFont"]:
        if not font_path:
            return None
        if not PIL_AVAILABLE:
            print("⚠️ Pillow未安装，无法加载字体。将使用OpenCV默认字体（不支持中文）。")
            return None
        try:
            return ImageFont.truetype(font_path, font_size)
        except OSError as exc:
            print(f"⚠️ 无法加载字体 {font_path}: {exc}")
            return None

    def _should_use_pil_font(self) -> bool:
        return self._pil_font is not None and PIL_AVAILABLE

    def _draw_text_with_pil(self, canvas: np.ndarray, text: str,
                            origin: Tuple[int, int], bgr_color: Tuple[int, int, int]):
        # Convert to PIL (RGB), draw, then convert back to BGR.
        canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(canvas_rgb)
        draw = ImageDraw.Draw(pil_img)
        rgb_color = (bgr_color[2], bgr_color[1], bgr_color[0])
        draw.text(origin, text, font=self._pil_font, fill=rgb_color)
        updated = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        canvas[:, :, :] = updated

    def _auto_detect_font(self) -> Optional[str]:
        if not (PIL_AVAILABLE and MATPLOTLIB_AVAILABLE and font_manager):
            return None
        for family in PREFERRED_FONT_FAMILIES or []:
            try:
                font_prop = font_manager.FontProperties(family=family)
                font_path = font_manager.findfont(font_prop, fallback_to_default=False)
            except Exception:
                continue
            if font_path and os.path.exists(font_path):
                return font_path
        return None

    @staticmethod
    def _normalize_input_image(image: np.ndarray) -> np.ndarray:
        """确保输入图像没有多余通道（如alpha），便于后续处理"""
        if image.ndim == 2:
            return image
        channels = image.shape[2]
        if channels == 1:
            return image[:, :, 0]
        if channels == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        if channels > 4:
            return image[:, :, :3]
        return image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="焊缝缺陷推理脚本（ROI + 增强 + 分割 + 映射 + 可视化）"
    )
    parser.add_argument("--image-dir", required=True, help="待推理图像目录")
    parser.add_argument("--roi-weights", required=True, help="ROI检测模型权重路径")
    parser.add_argument("--seg-weights", required=True, help="缺陷分割模型权重路径")
    parser.add_argument("--output-dir", default="inference_outputs", help="输出目录")
    parser.add_argument("--data-config", help="dataset.yaml路径，用于加载类别名称")
    parser.add_argument("--enhance-mode", choices=["original", "windowing"],
                        default="windowing", help="ROI增强模式")
    parser.add_argument("--roi-conf", type=float, default=0.25, help="ROI检测置信度阈值")
    parser.add_argument("--roi-iou", type=float, default=0.45, help="ROI检测IoU阈值")
    parser.add_argument("--roi-padding", type=float, default=0.1, help="ROI外扩比例")
    parser.add_argument("--seg-conf", type=float, default=0.25, help="分割模型置信度阈值")
    parser.add_argument("--imgsz", type=int, default=640, help="分割模型输入尺寸")
    parser.add_argument("--device", help="推理设备（如 0, cuda:0, cpu），默认自动")
    parser.add_argument("--visualize", action="store_true", help="是否保存可视化结果")
    parser.add_argument("--visualize-dir", help="可视化输出目录（默认在output_dir下）")
    parser.add_argument("--max-images", type=int, help="最多处理的图像数")
    parser.add_argument("--results-json", default="inference_results.json",
                        help="结果JSON文件名（路径相对output_dir）")
    parser.add_argument("--font-path", help="可选，指定TTF/OTF字体以正确显示中文标签")
    parser.add_argument("--font-size", type=int, default=20, help="字体像素大小（默认20）")
    return parser.parse_args()


def main():
    args = parse_args()
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    visualization_dir = Path(args.visualize_dir) if args.visualize_dir else None

    pipeline = WeldInferencePipeline(
        roi_weights=args.roi_weights,
        seg_weights=args.seg_weights,
        enhance_mode=args.enhance_mode,
        roi_conf=args.roi_conf,
        roi_iou=args.roi_iou,
        roi_padding=args.roi_padding,
        seg_conf=args.seg_conf,
        imgsz=args.imgsz,
        device=args.device,
        data_config=args.data_config,
        font_path=args.font_path,
        font_size=args.font_size
    )

    results = pipeline.process_directory(
        image_dir=image_dir,
        output_dir=output_dir,
        visualize=args.visualize,
        visualization_dir=visualization_dir,
        max_images=args.max_images
    )

    total_defects = sum(len(roi["defects"]) for item in results for roi in item["rois"])

    results_path = Path(args.results_json)
    if not results_path.is_absolute():
        results_path = output_dir / results_path

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2, ensure_ascii=False)

    print(f"\n推理完成: 共处理 {len(results)} 张图像，检测到 {total_defects} 个缺陷实例。")
    print(f"结果JSON: {results_path}")
    if args.visualize:
        vis_dir = visualization_dir or (output_dir / "visualizations")
        print(f"可视化输出目录: {vis_dir}")


if __name__ == "__main__":
    main()
