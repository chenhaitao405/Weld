#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Weld inference entry point.

Features:
    1. Batch load weld inspection images from a directory
    2. Run ROI detection to focus on weld seams
    3. Apply segmentation (YOLO mask), RF-DETR detection, or slice classification (sliding window heatmap)
    4. Map results back to the original image and optionally create visualizations
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

DEBUG_STEP = True

from convert.pj.yolo_roi_extractor import WeldROIDetector  # noqa: E402
from utils import read_dataset_yaml  # noqa: E402
from utils.pipeline_utils import FontRenderer, load_image  # noqa: E402
from utils import detection_pipeline as rfdet_pipeline  # noqa: E402
from utils import yolo_pipeline  # noqa: E402


SUPPORTED_IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


def _load_names_from_yaml(data_config: Optional[str]) -> List[str]:
    names: List[str] = []
    if not data_config:
        return names

    yaml_path = Path(data_config)
    if not yaml_path.exists():
        return names

    cfg = read_dataset_yaml(str(yaml_path))
    yaml_names = cfg.get('names')
    if isinstance(yaml_names, dict):
        sorted_items = sorted(yaml_names.items(), key=lambda kv: int(kv[0]))
        names = [item[1] for item in sorted_items]
    elif isinstance(yaml_names, list):
        names = yaml_names
    return names


def _load_class_names(explicit_names: Optional[Sequence[str]],
                      data_config: Optional[str],
                      model_with_names: Optional[Any] = None) -> List[str]:
    """按照优先级解析类别名称：命令行 > dataset.yaml > 模型自带"""
    names: List[str] = []

    if explicit_names:
        names = [str(name) for name in explicit_names]
        return names

    names = _load_names_from_yaml(data_config)
    if names:
        return names

    candidate = None
    if model_with_names is not None:
        if hasattr(model_with_names, 'names'):
            candidate = getattr(model_with_names, 'names')
        elif hasattr(model_with_names, 'class_names'):
            candidate = getattr(model_with_names, 'class_names')
        elif hasattr(model_with_names, 'model') and hasattr(model_with_names.model, 'class_names'):
            candidate = getattr(model_with_names.model, 'class_names')

    if isinstance(candidate, dict):
        sorted_items = sorted(candidate.items(), key=lambda kv: int(kv[0]))
        names = [str(item[1]) for item in sorted_items]
    elif isinstance(candidate, (list, tuple)):
        names = [str(item) for item in candidate]
    return names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="焊缝缺陷推理脚本（支持分割/检测/切片分类模式）"
    )
    parser.add_argument("--image-dir", required=True, help="待推理图像目录")
    parser.add_argument("--output-dir", default="inference_outputs", help="输出目录")
    parser.add_argument("--results-json", default="inference_results.json",
                        help="结果JSON文件名（相对output_dir）")
    parser.add_argument("--mode", choices=["seg", "det"], default="seg",
                        help="推理模式：seg=分割，det=RF-DETR检测")
    parser.add_argument("--visualize", action="store_true", help="是否保存可视化结果")
    parser.add_argument("--visualize-dir", help="可视化输出目录（默认在output_dir下）")
    parser.add_argument("--max-images", type=int, help="最多处理的图像数")

    # ROI配置
    parser.add_argument("--roi-weights", help="ROI检测权重（seg/det模式必填）")
    parser.add_argument("--roi-conf", type=float, default=0.25, help="ROI检测置信度阈值")
    parser.add_argument("--roi-iou", type=float, default=0.45, help="ROI检测IoU阈值")
    parser.add_argument("--roi-padding", type=float, default=0.1, help="ROI外扩比例")

    # 通用
    parser.add_argument("--enhance-mode", choices=["original", "windowing"],
                        default="windowing", help="图像增强模式")
    parser.add_argument("--font-path", help="可选，指定字体文件以正确显示中文标签")
    parser.add_argument("--font-size", type=int, default=20, help="可视化字体大小")
    parser.add_argument("--engine", choices=["rfdet", "yolo"], default="rfdet",
                        help="推理后端：rfdet=RF-DETR，yolo=YOLOv11")

    # 模型通用参数
    parser.add_argument("--primary-weights",
                        help="主模型权重（seg/det通用，优先于mode专用权重）")
    parser.add_argument("--secondary-weights",
                        help="可选：ROI切片增强模型权重（seg/det通用）")
    parser.add_argument("--secondary-confidence", type=float,
                        help="切片模型置信度阈值（默认与主模型一致）")
    parser.add_argument("--secondary-variant", choices=["medium", "large"],
                        help="切片检测模型规模（仅det模式有效，覆盖det-secondary-variant）")
    parser.add_argument("--patch-size", type=int, nargs=2,
                        help="切片推理窗口尺寸 [height width]，覆盖det-patch-size")
    parser.add_argument("--patch-overlap", type=float,
                        help="切片窗口重叠率 (0-1)，覆盖det-patch-overlap")
    parser.add_argument("--fusion-iou", type=float,
                        help="主干/切片NMS融合IoU阈值，覆盖det-fusion-iou")

    # 分割模式参数
    parser.add_argument("--seg-weights", help="分割模型权重（mode=seg时必填）")
    parser.add_argument("--data-config", help="dataset.yaml路径，用于加载类别名称")
    parser.add_argument("--seg-conf", type=float, default=0.25, help="分割置信度阈值")
    parser.add_argument("--imgsz", type=int, default=640, help="分割模型输入尺寸")
    parser.add_argument("--device", help="推理设备（如0, cuda:0, cpu）")

    # 检测模式参数（RF-DETR）
    parser.add_argument("--det-weights", help="RF-DETR检测权重（mode=det时必填）")
    parser.add_argument("--det-confidence", type=float, default=0.25,
                        help="RF-DETR检测置信度阈值")
    parser.add_argument("--det-device", help="RF-DETR推理设备（如cuda:0, cpu）")
    parser.add_argument("--det-optimize", action="store_true",
                        help="是否对RF-DETR执行推理优化")
    parser.add_argument("--det-optimize-batch", type=int, default=1,
                        help="RF-DETR优化推理时的batch size")
    parser.add_argument("--det-use-half", action="store_true",
                        help="RF-DETR优化时使用float16")
    parser.add_argument("--class-names", nargs='+',
                        help="自定义类别名称列表（按类别ID顺序，seg/det通用）")
    parser.add_argument("--det-secondary-weights",
                        help="可选：RF-DETR Medium权重（切片增强分支）")
    parser.add_argument("--det-secondary-confidence", type=float,
                        help="切片分支单独的置信度阈值（默认与主干一致）")
    parser.add_argument("--det-secondary-variant", choices=["medium", "large"], default="medium",
                        help="切片分支使用的RF-DETR模型规模，默认medium")
    parser.add_argument("--det-patch-size", type=int, nargs=2, default=[640, 640],
                        help="切片检测窗口大小 [height width]")
    parser.add_argument("--det-patch-overlap", type=float, default=0.2,
                        help="切片检测窗口重叠率 (0-1)")
    parser.add_argument("--det-fusion-iou", type=float, default=0.5,
                        help="主干/切片分支NMS融合IoU阈值")

    return parser.parse_args()


def collect_images(image_dir: Path, max_images: Optional[int]) -> List[Path]:
    image_paths = sorted([
        p for p in image_dir.rglob('*')
        if p.suffix.lower() in SUPPORTED_IMAGE_EXTS and p.is_file()
    ])
    if max_images is not None:
        image_paths = image_paths[:max_images]
    if not image_paths:
        raise FileNotFoundError(f"未在 {image_dir} 中找到支持的图像文件")
    return image_paths


def build_roi_detector(args: argparse.Namespace) -> Optional[WeldROIDetector]:
    if not args.roi_weights:
        return None
    print(f"加载ROI模型: {args.roi_weights}")
    return WeldROIDetector(
        model_path=args.roi_weights,
        roi_conf_threshold=args.roi_conf,
        roi_iou_threshold=args.roi_iou,
        padding_ratio=args.roi_padding
    )


def _resolve_primary_weights(args: argparse.Namespace) -> str:
    if args.primary_weights:
        return args.primary_weights
    if args.mode == "seg" and args.seg_weights:
        return args.seg_weights
    if args.mode == "det" and args.det_weights:
        return args.det_weights
    missing_flag = "--seg-weights" if args.mode == "seg" else "--det-weights"
    raise ValueError(f"{args.mode} 模式需要提供 {missing_flag} 或 --primary-weights")


def _resolve_secondary_weights(args: argparse.Namespace) -> Optional[str]:
    if args.secondary_weights:
        return args.secondary_weights
    return args.det_secondary_weights


def _resolve_secondary_confidence(args: argparse.Namespace, primary_conf: float) -> float:
    if args.secondary_confidence is not None:
        return args.secondary_confidence
    if args.det_secondary_confidence is not None:
        return args.det_secondary_confidence
    return primary_conf


def _resolve_secondary_variant(args: argparse.Namespace) -> str:
    if args.secondary_variant:
        return args.secondary_variant
    if args.det_secondary_variant:
        return args.det_secondary_variant
    return "medium"


def _coerce_patch_size(value: Optional[Sequence[int]]) -> Optional[Tuple[int, int]]:
    if not value:
        return None
    if len(value) != 2:
        raise ValueError("切片窗口尺寸需要2个整数 [height width]")
    return int(value[0]), int(value[1])


def _resolve_patch_window(args: argparse.Namespace) -> Optional[Tuple[int, int]]:
    if args.patch_size:
        return _coerce_patch_size(args.patch_size)
    return _coerce_patch_size(args.det_patch_size)


def _resolve_patch_overlap(args: argparse.Namespace) -> float:
    if args.patch_overlap is not None:
        return args.patch_overlap
    return args.det_patch_overlap


def _resolve_fusion_iou(args: argparse.Namespace) -> float:
    if args.fusion_iou is not None:
        return args.fusion_iou
    return args.det_fusion_iou


class InferencePipelineRunner:
    def __init__(self,
                 args: argparse.Namespace,
                 roi_detector: WeldROIDetector,
                 visualization_dir: Optional[Path],
                 font_renderer: FontRenderer,
                 debug_root: Optional[Path]):
        self.args = args
        self.mode = args.mode
        self.engine = args.engine
        self.roi_detector = roi_detector
        self.visualization_dir = visualization_dir
        self.visualize = args.visualize
        self.font_renderer = font_renderer
        self.debug_root = debug_root

        self._init_engine_components()

        self.primary_weights = _resolve_primary_weights(args)
        self.primary_confidence = args.det_confidence if self.mode == "det" else args.seg_conf
        self.secondary_weights = _resolve_secondary_weights(args)
        self.secondary_confidence = _resolve_secondary_confidence(args, self.primary_confidence)
        self.patch_window = _resolve_patch_window(args)
        self.patch_overlap = _resolve_patch_overlap(args)
        self.fusion_iou = _resolve_fusion_iou(args)

        self.primary_model = self._build_primary_model()
        self.secondary_model = self._build_secondary_model()
        self.class_names = self._resolve_class_names()
        self._apply_class_names()

    def _init_engine_components(self):
        if self.engine == "rfdet":
            self.det_model_cls = rfdet_pipeline.RFDetrDetectionModel
            self.seg_model_cls = rfdet_pipeline.RFDetrSegmentationModel
            self.detection_config_cls = rfdet_pipeline.RFDetrDetectionConfig
            self.run_detection_func = rfdet_pipeline.run_rfdet_detection
            self.process_segmentation_func = rfdet_pipeline.process_roi_and_segmentation
        elif self.engine == "yolo":
            self.det_model_cls = yolo_pipeline.YoloDetectionModel
            self.seg_model_cls = yolo_pipeline.YoloSegmentationModel
            self.detection_config_cls = yolo_pipeline.YoloDetectionConfig
            self.run_detection_func = yolo_pipeline.run_yolo_detection
            self.process_segmentation_func = yolo_pipeline.process_roi_and_segmentation
        else:
            raise ValueError(f"不支持的engine: {self.engine}")

    def run(self, image_paths: List[Path]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for image_path in tqdm(image_paths, desc="推理中"):
            try:
                rois, vis_path, width, height = self._process_image(image_path)
                results.append({
                    "mode": self.mode,
                    "image_path": str(image_path),
                    "width": width,
                    "height": height,
                    "num_rois": len(rois),
                    "rois": rois,
                    "visualization": vis_path
                })
            except Exception as exc:
                print(f"[警告] 处理 {image_path} 时出错: {exc}")
        return results

    def _build_primary_model(self):
        if self.mode == "det":
            if self.engine == "rfdet":
                return self.det_model_cls(
                    model_path=self.primary_weights,
                    confidence=self.primary_confidence,
                    device=self.args.det_device,
                    optimize=self.args.det_optimize,
                    optimize_batch=self.args.det_optimize_batch,
                    use_half=self.args.det_use_half,
                    class_names=None,
                    model_variant="large"
                )
            return self.det_model_cls(
                model_path=self.primary_weights,
                confidence=self.primary_confidence,
                device=self.args.device or self.args.det_device,
                class_names=None
            )
        if self.engine == "rfdet":
            return self.seg_model_cls(
                model_path=self.primary_weights,
                confidence=self.primary_confidence,
                device=self.args.device,
                class_names=None
            )
        return self.seg_model_cls(
            model_path=self.primary_weights,
            confidence=self.primary_confidence,
            device=self.args.device,
            class_names=None
        )

    def _build_secondary_model(self):
        if not self.secondary_weights:
            return None
        if self.mode == "det":
            if self.engine == "rfdet":
                variant = _resolve_secondary_variant(self.args)
                return self.det_model_cls(
                    model_path=self.secondary_weights,
                    confidence=self.secondary_confidence,
                    device=self.args.det_device,
                    optimize=self.args.det_optimize,
                    optimize_batch=self.args.det_optimize_batch,
                    use_half=self.args.det_use_half,
                    class_names=None,
                    model_variant=variant
                )
            return self.det_model_cls(
                model_path=self.secondary_weights,
                confidence=self.secondary_confidence,
                device=self.args.device or self.args.det_device,
                class_names=None
            )
        if self.engine == "rfdet":
            return self.seg_model_cls(
                model_path=self.secondary_weights,
                confidence=self.secondary_confidence,
                device=self.args.device,
                class_names=None
            )
        return self.seg_model_cls(
            model_path=self.secondary_weights,
            confidence=self.secondary_confidence,
            device=self.args.device,
            class_names=None
        )

    def _resolve_class_names(self) -> List[str]:
        base_model = getattr(self.primary_model, "model", None)
        return _load_class_names(self.args.class_names, self.args.data_config, base_model)

    def _apply_class_names(self):
        if not self.class_names:
            return
        if self.mode == "det":
            self.primary_model.class_map = self.primary_model._build_class_map(self.class_names)
            if self.secondary_model is not None:
                self.secondary_model.class_map = self.secondary_model._build_class_map(self.class_names)
        else:
            self.primary_model.class_map = self.primary_model._build_class_map(self.class_names)
            if self.secondary_model is not None:
                self.secondary_model.class_map = self.secondary_model._build_class_map(self.class_names)

    def _image_debug_dir(self, image_path: Path) -> Optional[Path]:
        if self.debug_root is None:
            return None
        return self.debug_root / image_path.stem

    def _process_image(self, image_path: Path) -> Tuple[List[Dict[str, Any]], Optional[str], int, int]:
        image = load_image(image_path)
        h, w = image.shape[:2]
        debug_dir = self._image_debug_dir(image_path)

        if self.mode == "det":
            config = self.detection_config_cls(
                roi_detector=self.roi_detector,
                detection_model=self.primary_model,
                secondary_model=self.secondary_model,
                enhance_mode=self.args.enhance_mode,
                patch_window=self.patch_window,
                patch_overlap=self.patch_overlap,
                fusion_iou=self.fusion_iou,
                visualize=self.visualize,
                visualization_dir=self.visualization_dir if self.visualize else None,
                font_renderer=self.font_renderer,
                debug_root=debug_dir
            )
            rois, vis_path = self.run_detection_func(image, image_path, config)
        else:
            if self.roi_detector is None:
                raise ValueError("seg 模式必须提供 ROI 检测器")
            rois, vis_path = self.process_segmentation_func(
                image=image,
                image_path=image_path,
                roi_detector=self.roi_detector,
                seg_model=self.primary_model,
                enhance_mode=self.args.enhance_mode,
                visualize=self.visualize,
                visualization_dir=self.visualization_dir if self.visualize else None,
                font_renderer=self.font_renderer,
                secondary_model=self.secondary_model,
                patch_window=self.patch_window,
                patch_overlap=self.patch_overlap,
                fusion_iou=self.fusion_iou,
                debug_dir=debug_dir
            )

        return rois, vis_path, w, h


def main():
    args = parse_args()
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir = Path(args.visualize_dir) if args.visualize_dir else (output_dir / "visualizations")

    image_paths = collect_images(image_dir, args.max_images)
    roi_detector = build_roi_detector(args)

    if args.mode in {"seg", "det"} and roi_detector is None:
        raise ValueError(f"{args.mode} 模式必须提供 --roi-weights 以执行ROI检测")

    font_renderer = FontRenderer(font_path=args.font_path, font_size=args.font_size)

    debug_root = (output_dir / "temp") if DEBUG_STEP else None
    if debug_root is not None:
        debug_root.mkdir(parents=True, exist_ok=True)

    runner = InferencePipelineRunner(
        args=args,
        roi_detector=roi_detector,
        visualization_dir=visualization_dir,
        font_renderer=font_renderer,
        debug_root=debug_root
    )
    results = runner.run(image_paths)

    results_path = Path(args.results_json)
    if not results_path.is_absolute():
        results_path = output_dir / results_path

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({"mode": args.mode, "results": results}, f, indent=2, ensure_ascii=False)

    print(f"\n推理完成: 模式={args.mode}，共处理 {len(results)} 张图像。")
    print(f"结果JSON: {results_path}")
    if args.visualize:
        print(f"可视化输出目录: {visualization_dir}")


if __name__ == "__main__":
    main()
