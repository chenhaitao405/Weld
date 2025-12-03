#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RF-DETR batch inference & visualization helper.

The script loads a trained RF-DETR checkpoint, runs detection on a directory of
images, and saves both structured JSON results and annotated visualizations.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import yaml

from rfdetr import RFDETRMedium #or RFDETRBASE
from utils.pipeline_utils import FontRenderer, draw_detection_instance, ensure_color


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_VAL_DIR = Path(
    "/home/lenovo/code/CHT/datasets/Xray/self/1120/labeled/roi2_640_merge/coco_det/test"
)
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = SCRIPT_DIR / "runs" / "detrmedium" / "checkpoint_best_total.pth"
DEFAULT_OUTPUT = SCRIPT_DIR / "runs" / "detrmedium" / "temp"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RF-DETR 推理脚本：批量推理并生成可视化结果"
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_VAL_DIR,
        help="待推理图片目录（默认验证集）",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="训练好的权重（checkpoint_best_ema.pth）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="结果输出根目录，包含JSON与可视化图",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="置信度阈值，低于该值的预测会被过滤",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="仅处理前N张图，调试/抽检时可用",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "mps"],
        default=None,
        help="可选：指定推理设备，默认跟随模型配置",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="加载后对模型进行推理优化（TorchScript tracing）",
    )
    parser.add_argument(
        "--optimize-batch",
        type=int,
        default=1,
        help="优化推理时使用的 batch size（仅 --optimize 生效）",
    )
    parser.add_argument(
        "--use-half",
        action="store_true",
        help="优化推理时使用 float16 精度（需 GPU 支持）",
    )
    parser.add_argument(
        "--font-path",
        type=Path,
        help="可选：指定支持中文的TTF/TTC字体文件，用于绘制标签",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=22,
        help="字体大小（仅在提供 --font-path 时生效）",
    )
    parser.add_argument(
        "--categories-json",
        type=Path,
        help="可选：COCO标注JSON路径（从其中读取类别名）",
    )
    parser.add_argument(
        "--class-names",
        nargs="+",
        help="可选：直接提供类别名列表（按ID顺序）",
    )
    return parser.parse_args()


def iter_images(root: Path) -> Iterable[Path]:
    if not root.exists():
        raise FileNotFoundError(f"图像目录不存在: {root}")
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS:
            yield path


def load_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def build_class_name_map_from_model(model: RFDETRMedium) -> Dict[int, str]:
    raw_names = getattr(model, "class_names", {}) or {}
    if isinstance(raw_names, dict):
        return {int(k): str(v) for k, v in raw_names.items()}
    if isinstance(raw_names, (list, tuple)):
        return {i: str(name) for i, name in enumerate(raw_names)}
    return {}


def load_class_names_from_coco(json_path: Path) -> Dict[int, str]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    categories = data.get("categories", [])
    class_map: Dict[int, str] = {}
    for cat in categories:
        try:
            cat_id = int(cat["id"])
        except Exception:
            continue
        name = str(cat.get("name", cat_id))
        class_map[cat_id] = name
    return class_map


def infer_class_map_from_dataset_yaml(image_dir: Path) -> Optional[Dict[int, str]]:
    # 适配YOLO数据结构：.../det/images/val -> .../det/dataset.yaml
    candidates = []
    try:
        candidates.append(image_dir.parent.parent / "dataset.yaml")
    except Exception:
        pass
    for candidate in candidates:
        if candidate and candidate.exists():
            with candidate.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            if not isinstance(cfg, dict):
                continue
            label_map = cfg.get("label_id_map")
            if isinstance(label_map, dict):
                try:
                    return {int(v): str(k) for k, v in label_map.items()}
                except Exception:
                    pass
            names = cfg.get("names")
            if isinstance(names, list):
                return {i: str(name) for i, name in enumerate(names)}
            if isinstance(names, dict):
                try:
                    return {int(k): str(v) for k, v in names.items()}
                except Exception:
                    continue
    return None


def build_effective_class_map(args: argparse.Namespace, model: RFDETRMedium) -> Dict[int, str]:
    if args.class_names:
        return {idx: str(name) for idx, name in enumerate(args.class_names)}
    if args.categories_json:
        if not args.categories_json.exists():
            raise FileNotFoundError(f"指定的COCO标注不存在: {args.categories_json}")
        return load_class_names_from_coco(args.categories_json)
    dataset_yaml_map = infer_class_map_from_dataset_yaml(args.image_dir)
    if dataset_yaml_map:
        return dataset_yaml_map
    return build_class_name_map_from_model(model)


def resolve_label(class_id: int, label_map: Dict[int, str]) -> str:
    if class_id in label_map:
        return label_map[class_id]
    if class_id + 1 in label_map:
        return label_map[class_id + 1]
    if class_id - 1 in label_map:
        return label_map[class_id - 1]
    return f"class_{class_id}"


def draw_detections(image: np.ndarray,
                    detections,
                    label_map: Dict[int, str],
                    font_renderer: Optional[FontRenderer]) -> np.ndarray:
    annotated = ensure_color(image.copy())
    if len(getattr(detections, "xyxy", [])) == 0:
        return annotated

    for bbox, score, cls_id in zip(
        detections.xyxy, detections.confidence, detections.class_id
    ):
        label = resolve_label(int(cls_id), label_map)
        draw_detection_instance(
            annotated,
            bbox=bbox.tolist(),
            label=label,
            score=float(score),
            class_id=int(cls_id),
            font_renderer=font_renderer
        )
    return annotated


def main() -> None:
    args = parse_args()
    image_paths: List[Path] = list(iter_images(args.image_dir))
    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        raise RuntimeError(f"未在 {args.image_dir} 中找到可用图像")

    if not args.weights.exists():
        raise FileNotFoundError(f"未找到权重文件: {args.weights}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = args.output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    model_kwargs = {"pretrain_weights": str(args.weights)}
    if args.device:
        model_kwargs["device"] = args.device

    print(f"加载RF-DETR模型: {args.weights}")
    model = RFDETRMedium(**model_kwargs)
    if args.optimize:
        import torch

        dtype = torch.float16 if args.use_half else torch.float32
        print(
            f"优化推理（batch_size={args.optimize_batch}, dtype={dtype}）"
        )
        model.optimize_for_inference(
            batch_size=args.optimize_batch,
            dtype=dtype,
        )

    class_map = build_effective_class_map(args, model)
    font_renderer = FontRenderer(
        font_path=str(args.font_path) if args.font_path else None,
        font_size=args.font_size
    )
    per_class_counter: Counter[int] = Counter()
    inference_records = []

    for image_path in tqdm(image_paths, desc="RF-DETR 推理"):
        pil_image = load_rgb_image(image_path)
        detections = model.predict(pil_image, threshold=args.confidence)
        if isinstance(detections, list):
            raise RuntimeError(
                "模型返回了批量结果，请逐张推理或在此处展开处理逻辑。"
            )

        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            print(f"[警告] 读取失败，跳过: {image_path}")
            continue
        annotated = draw_detections(image_bgr, detections, class_map, font_renderer)

        try:
            rel = image_path.relative_to(args.image_dir)
        except ValueError:
            rel = Path(image_path.name)
        save_path = vis_dir / rel
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), annotated)

        det_list = []
        for bbox, score, cls_id in zip(
            detections.xyxy, detections.confidence, detections.class_id
        ):
            cls_id_int = int(cls_id)
            per_class_counter[cls_id_int] += 1
            det_list.append(
                {
                    "bbox_xyxy": [float(x) for x in bbox.tolist()],
                    "score": float(score),
                    "class_id": cls_id_int,
                    "label": resolve_label(cls_id_int, class_map),
                }
            )

        inference_records.append(
            {
                "image_path": str(image_path),
                "num_detections": len(det_list),
                "visualization_path": str(save_path),
                "detections": det_list,
            }
        )

    summary = {
        "weights": str(args.weights),
        "image_dir": str(args.image_dir),
        "num_images": len(inference_records),
        "confidence_threshold": args.confidence,
        "class_distribution": [
            {
                "class_id": cls_id,
                "label": resolve_label(cls_id, class_map),
                "count": count,
            }
            for cls_id, count in sorted(per_class_counter.items())
        ],
        "images": inference_records,
    }

    json_path = args.output_dir / "predictions.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"推理完成，JSON结果: {json_path}")
    print(f"可视化目录: {vis_dir}")


if __name__ == "__main__":
    main()
