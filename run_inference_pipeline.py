#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Weld inference entry point.

Features:
    1. Batch load weld inspection images from a directory
    2. Run ROI detection to focus on weld seams
    3. Apply either segmentation (YOLO mask) or slice classification (sliding window heatmap)
    4. Map results back to the original image and optionally create visualizations
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
from tqdm import tqdm
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from convert.pj.yolo_roi_extractor import WeldROIDetector  # noqa: E402
from utils import read_dataset_yaml  # noqa: E402
from utils.pipeline_utils import FontRenderer, load_image  # noqa: E402
from utils.segmentation_pipeline import process_roi_and_segmentation  # noqa: E402
from utils.slice_classification_pipeline import (  # noqa: E402
    SliceClassificationPipeline, DEFAULT_ALPHA as CLS_DEFAULT_ALPHA,
    DEFAULT_CONFIDENCE_THRESHOLD as CLS_DEFAULT_CONF,
    DEFAULT_DEFECT_CLASSES as CLS_DEFAULT_CLASSES,
    DEFAULT_OVERLAP_RATIO as CLS_DEFAULT_OVERLAP,
    DEFAULT_WINDOW_SIZE as CLS_DEFAULT_WINDOW
)


SUPPORTED_IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


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
        model_names = seg_model.names
        if isinstance(model_names, dict):
            names = [model_names[i] for i in sorted(model_names.keys())]
        elif isinstance(model_names, (list, tuple)):
            names = list(model_names)
    return names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="焊缝缺陷推理脚本（支持分割和切片分类模式）"
    )
    parser.add_argument("--image-dir", required=True, help="待推理图像目录")
    parser.add_argument("--output-dir", default="inference_outputs", help="输出目录")
    parser.add_argument("--results-json", default="inference_results.json",
                        help="结果JSON文件名（相对output_dir）")
    parser.add_argument("--mode", choices=["seg", "cls"], default="seg",
                        help="推理模式：seg=分割，cls=切片分类/热力图")
    parser.add_argument("--visualize", action="store_true", help="是否保存可视化结果")
    parser.add_argument("--visualize-dir", help="可视化输出目录（默认在output_dir下）")
    parser.add_argument("--max-images", type=int, help="最多处理的图像数")

    # ROI配置
    parser.add_argument("--roi-weights", help="ROI检测权重（分割模式必填）")
    parser.add_argument("--roi-conf", type=float, default=0.25, help="ROI检测置信度阈值")
    parser.add_argument("--roi-iou", type=float, default=0.45, help="ROI检测IoU阈值")
    parser.add_argument("--roi-padding", type=float, default=0.1, help="ROI外扩比例")

    # 通用
    parser.add_argument("--enhance-mode", choices=["original", "windowing"],
                        default="windowing", help="图像增强模式")
    parser.add_argument("--font-path", help="可选，指定字体文件以正确显示中文标签")
    parser.add_argument("--font-size", type=int, default=20, help="可视化字体大小")

    # 分割模式参数
    parser.add_argument("--seg-weights", help="分割模型权重（mode=seg时必填）")
    parser.add_argument("--data-config", help="dataset.yaml路径，用于加载类别名称")
    parser.add_argument("--seg-conf", type=float, default=0.25, help="分割置信度阈值")
    parser.add_argument("--imgsz", type=int, default=640, help="分割模型输入尺寸")
    parser.add_argument("--device", help="推理设备（如0, cuda:0, cpu）")

    # 切片分类模式参数
    parser.add_argument("--cls-weights", help="切片分类模型权重（mode=cls时必填）")
    parser.add_argument("--cls-window-size", type=int, nargs=2,
                        default=[CLS_DEFAULT_WINDOW, CLS_DEFAULT_WINDOW],
                        help="切片窗口大小 [height width]")
    parser.add_argument("--cls-overlap", type=float, default=CLS_DEFAULT_OVERLAP,
                        help="切片窗口重叠率 (0-1)")
    parser.add_argument("--cls-confidence", type=float, default=CLS_DEFAULT_CONF,
                        help="缺陷置信度阈值")
    parser.add_argument("--cls-defect-classes", type=int, nargs='+',
                        default=list(CLS_DEFAULT_CLASSES),
                        help="被视为缺陷的分类ID (空格分隔)")
    parser.add_argument("--cls-use-confidence-weight", action="store_true",
                        help="热力图累积使用置信度加权")
    parser.add_argument("--cls-colormap", default="jet",
                        choices=['hot', 'jet', 'turbo', 'viridis', 'plasma',
                                 'coolwarm', 'RdYlBu', 'YlOrRd'],
                        help="热力图颜色映射")
    parser.add_argument("--cls-alpha", type=float, default=CLS_DEFAULT_ALPHA,
                        help="热力图叠加透明度 (0-1)")
    parser.add_argument("--cls-display-mode", default="overlay",
                        choices=['overlay', 'contour', 'sparse'],
                        help="热力图显示模式")
    parser.add_argument("--cls-heatmap-only", action="store_true",
                        help="只保存热力图，不叠加原图")
    parser.add_argument("--cls-no-colorbar", action="store_true",
                        help="不在结果图上添加颜色条")

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


def run_segmentation_mode(args: argparse.Namespace,
                          image_paths: List[Path],
                          roi_detector: WeldROIDetector,
                          visualization_dir: Optional[Path],
                          font_renderer: FontRenderer) -> List[Dict[str, Any]]:
    if not args.seg_weights:
        raise ValueError("seg模式需要提供 --seg-weights")

    print(f"加载分割模型: {args.seg_weights}")
    seg_model = YOLO(args.seg_weights)
    class_names = _load_class_names(args.data_config, seg_model)

    results: List[Dict[str, Any]] = []

    for image_path in tqdm(image_paths, desc="推理中"):
        try:
            image = load_image(image_path)
            rois, vis_path = process_roi_and_segmentation(
                image=image,
                image_path=image_path,
                roi_detector=roi_detector,
                seg_model=seg_model,
                class_names=class_names,
                enhance_mode=args.enhance_mode,
                seg_conf=args.seg_conf,
                imgsz=args.imgsz,
                device=args.device,
                visualize=args.visualize,
                visualization_dir=visualization_dir,
                font_renderer=font_renderer
            )

            h, w = image.shape[:2]
            results.append({
                "mode": "seg",
                "image_path": str(image_path),
                "width": w,
                "height": h,
                "num_rois": len(rois),
                "rois": rois,
                "visualization": vis_path
            })
        except Exception as exc:
            print(f"[警告] 处理 {image_path} 时出错: {exc}")

    return results


def run_classification_mode(args: argparse.Namespace,
                            image_paths: List[Path],
                            roi_detector: Optional[WeldROIDetector],
                            visualization_dir: Optional[Path]) -> List[Dict[str, Any]]:
    if not args.cls_weights:
        raise ValueError("cls模式需要提供 --cls-weights")

    pipeline = SliceClassificationPipeline(
        model_path=args.cls_weights,
        window_size=tuple(args.cls_window_size),
        overlap_ratio=args.cls_overlap,
        enhance_mode=args.enhance_mode,
        confidence_threshold=args.cls_confidence,
        use_confidence_weight=args.cls_use_confidence_weight,
        colormap=args.cls_colormap,
        alpha=args.cls_alpha,
        display_mode=args.cls_display_mode,
        defect_class_ids=args.cls_defect_classes,
        roi_detector=roi_detector
    )

    results: List[Dict[str, Any]] = []

    for image_path in tqdm(image_paths, desc="推理中"):
        try:
            image = load_image(image_path)
            classification_result = pipeline.detect_slice_classify(image, image_id=str(image_path))
            predictions = classification_result['patch_predictions']
            heatmap = pipeline.generate_heatmap(image.shape, predictions)
            stats = SliceClassificationPipeline.generate_statistics(heatmap)

            vis_path = None
            if args.visualize:
                if args.cls_heatmap_only:
                    result_image = pipeline.apply_colormap(heatmap)
                else:
                    result_image = pipeline.create_heatmap_overlay(image, heatmap, apply_gaussian_blur=True)
                if not args.cls_no_colorbar and heatmap.max() > 0:
                    result_image = pipeline.add_colorbar(result_image, heatmap)

                if visualization_dir is not None:
                    visualization_dir.mkdir(parents=True, exist_ok=True)
                    vis_file = visualization_dir / f"{image_path.stem}_heatmap.jpg"
                else:
                    vis_file = image_path.with_name(f"{image_path.stem}_heatmap.jpg")
                cv2.imwrite(str(vis_file), result_image)
                vis_path = str(vis_file)

            classification_result["mode"] = "cls"
            classification_result["visualization"] = vis_path
            classification_result["heatmap_stats"] = stats
            results.append(classification_result)
        except Exception as exc:
            print(f"[警告] 处理 {image_path} 时出错: {exc}")

    return results


def main():
    args = parse_args()
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir = Path(args.visualize_dir) if args.visualize_dir else (output_dir / "visualizations")

    image_paths = collect_images(image_dir, args.max_images)
    roi_detector = build_roi_detector(args)

    if args.mode == "seg" and roi_detector is None:
        raise ValueError("seg模式必须提供 --roi-weights 以执行ROI检测")

    font_renderer = FontRenderer(font_path=args.font_path, font_size=args.font_size)

    if args.mode == "seg":
        results = run_segmentation_mode(args, image_paths, roi_detector, visualization_dir, font_renderer)
    else:
        results = run_classification_mode(args, image_paths, roi_detector, visualization_dir if args.visualize else None)

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
