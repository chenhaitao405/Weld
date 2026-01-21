#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Weld inference entry point.

Features:
    1. Batch load weld inspection images from a directory
    2. Run ROI detection to focus on weld seams
    3. Apply RF-DETR segmentation or detection
    4. Map results back to the original image
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

DEBUG_STEP = True

from convert.pj.yolo_roi_extractor import WeldROIDetector  # noqa: E402
from utils.pipeline_utils import FontRenderer, load_image  # noqa: E402
from utils import detection_pipeline as rfdet_pipeline  # noqa: E402


SUPPORTED_IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="焊缝缺陷推理脚本（支持RF-DETR分割/检测模式）"
    )
    parser.add_argument("--image-dir", required=True, help="待推理图像目录")
    parser.add_argument("--output-dir", default="inference_outputs", help="输出目录")
    parser.add_argument("--results-json", default="inference_results.json",
                        help="结果JSON文件名（相对output_dir）")
    parser.add_argument("--mode", choices=["seg", "det"], default="seg",
                        help="推理模式：seg=分割，det=RF-DETR检测")
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

    # 模型通用参数
    parser.add_argument("--primary-weights", help="主模型权重（seg/det通用）")
    parser.add_argument("--fusion-iou", type=float, default=0.5,
                        help="主干/宽切片NMS融合IoU阈值")

    # 分割模式参数
    parser.add_argument("--primary-conf", type=float, default=0.25, help="主模型置信度阈值")
    parser.add_argument("--device", help="推理设备（如0, cuda:0, cpu）")

    # 检测模式参数（RF-DETR）
    parser.add_argument("--det-device", help="RF-DETR推理设备（如cuda:0, cpu）")
    parser.add_argument("--wide-slice", action="store_true",
                        help="启用横切纵拼推理（seg/det模式），结果与主干一起融合")

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
    raise ValueError("seg/det 模式需要提供 --primary-weights")


def _resolve_fusion_iou(args: argparse.Namespace) -> float:
    return args.fusion_iou


class InferencePipelineRunner:
    def __init__(self,
                 args: argparse.Namespace,
                 roi_detector: WeldROIDetector,
                 font_renderer: FontRenderer,
                 debug_root: Optional[Path]):
        self.args = args
        self.mode = args.mode
        self.roi_detector = roi_detector
        self.font_renderer = font_renderer
        self.debug_root = debug_root

        self.det_model_cls = rfdet_pipeline.RFDetrDetectionModel
        self.seg_model_cls = rfdet_pipeline.RFDetrSegmentationModel
        self.detection_config_cls = rfdet_pipeline.RFDetrDetectionConfig
        self.run_detection_func = rfdet_pipeline.run_rfdet_detection
        self.process_segmentation_func = rfdet_pipeline.process_roi_and_segmentation

        self.primary_weights = _resolve_primary_weights(args)
        self.primary_confidence = args.primary_conf
        self.fusion_iou = _resolve_fusion_iou(args)
        self.enable_wide_slice = bool(getattr(args, "wide_slice", False))

        self.primary_model = self._build_primary_model()

    def run(self, image_paths: List[Path]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for image_path in tqdm(image_paths, desc="推理中"):
            try:
                rois, width, height = self._process_image(image_path)
                results.append({
                    "mode": self.mode,
                    "image_path": str(image_path),
                    "width": width,
                    "height": height,
                    "num_rois": len(rois),
                    "rois": rois
                })
            except Exception as exc:
                print(f"[警告] 处理 {image_path} 时出错: {exc}")
        return results

    def _build_primary_model(self):
        if self.mode == "det":
            return self.det_model_cls(
                model_path=self.primary_weights,
                confidence=self.primary_confidence,
                device=self.args.det_device,
                model_variant="large"
            )
        return self.seg_model_cls(
            model_path=self.primary_weights,
            confidence=self.primary_confidence,
            device=self.args.device
        )

    def _image_debug_dir(self, image_path: Path) -> Optional[Path]:
        if self.debug_root is None:
            return None
        return self.debug_root / image_path.stem

    def _process_image(self, image_path: Path) -> Tuple[List[Dict[str, Any]], int, int]:
        image = load_image(image_path)
        h, w = image.shape[:2]
        debug_dir = self._image_debug_dir(image_path)

        wide_slice_cfg = rfdet_pipeline.WideSliceConfig(enabled=True) if self.enable_wide_slice else None
        if self.mode == "det":
            config = self.detection_config_cls(
                roi_detector=self.roi_detector,
                detection_model=self.primary_model,
                enhance_mode=self.args.enhance_mode,
                fusion_iou=self.fusion_iou,
                font_renderer=self.font_renderer,
                debug_root=debug_dir,
                wide_slice=wide_slice_cfg
            )
            rois = self.run_detection_func(image, config)
        else:
            if self.roi_detector is None:
                raise ValueError("seg 模式必须提供 ROI 检测器")
            rois = self.process_segmentation_func(
                image=image,
                roi_detector=self.roi_detector,
                seg_model=self.primary_model,
                enhance_mode=self.args.enhance_mode,
                font_renderer=self.font_renderer,
                debug_dir=debug_dir,
                wide_slice=wide_slice_cfg,
                fusion_iou=self.fusion_iou
            )

        return rois, w, h


def main():
    args = parse_args()
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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


if __name__ == "__main__":
    main()
