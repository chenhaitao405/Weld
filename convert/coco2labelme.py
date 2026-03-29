#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
COCO <-> LabelMe 标注转换脚本

用法示例：
  python convert/coco2labelme.py --mode coco2labelme --input_dir /path/to/coco_folder
  python convert/coco2labelme.py --mode labelme2coco --input_dir /path/to/labelme_folder

约定：
  - coco2labelme:
      input_dir 下包含图片文件和 _annotations.coco.json
      输出 LabelMe 标注到 input_dir/labelmelabel
  - labelme2coco:
      input_dir 下包含图片文件，LabelMe 标注位于 input_dir/labelmelabel
      输出 COCO 标注到 input_dir/_annotations.coco.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from PIL import Image
except ImportError:
    Image = None

LABELME_VERSION = "5.9.1"
ANNOTATION_FILE_NAME = "_annotations.coco.json"
DEFAULT_LABELME_SUBDIR = "labelmelabel"
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="在 COCO 与 LabelMe 之间双向转换标注",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["coco2labelme", "labelme2coco"],
        default="coco2labelme",
        help="转换方向",
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="输入目录",
    )
    parser.add_argument(
        "--annotation_file",
        default=ANNOTATION_FILE_NAME,
        help="COCO 标注文件名；coco2labelme 时用于读取，labelme2coco 时用于写出",
    )
    parser.add_argument(
        "--labelme_subdir",
        default=DEFAULT_LABELME_SUBDIR,
        help="LabelMe 标注子目录名；coco2labelme 时用于写出，labelme2coco 时用于读取",
    )
    parser.add_argument(
        "--embed_image_data",
        action="store_true",
        help="仅 coco2labelme 使用：将 imageData 内嵌到输出 JSON 中；默认写 null",
    )
    return parser.parse_args()


def load_json(json_path: Path) -> Dict[str, Any]:
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], json_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_image_size(image_path: Path) -> Tuple[int, int]:
    if Image is None:
        raise ImportError("Pillow 未安装，无法读取图像尺寸。请先运行 pip install pillow")

    with Image.open(image_path) as img:
        width, height = img.size
    return int(width), int(height)


def encode_image_data(image_path: Path) -> str:
    import base64

    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def round_point(x: float, y: float) -> List[float]:
    return [round(float(x), 6), round(float(y), 6)]


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def os_path_rel(target: Path, start: Path) -> str:
    return os.path.relpath(str(target), str(start))


def resolve_relative_path(path_value: str) -> Path:
    return Path(path_value.replace("\\", "/"))


def flatten_points(points: Sequence[Sequence[float]]) -> List[float]:
    flattened: List[float] = []
    for point in points:
        flattened.extend([round(float(point[0]), 6), round(float(point[1]), 6)])
    return flattened


def polygon_area(points: Sequence[Sequence[float]]) -> float:
    if len(points) < 3:
        return 0.0

    area = 0.0
    for idx in range(len(points)):
        x1, y1 = float(points[idx][0]), float(points[idx][1])
        x2, y2 = float(points[(idx + 1) % len(points)][0]), float(points[(idx + 1) % len(points)][1])
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def sanitize_points(points: Any, image_width: int, image_height: int) -> List[List[float]]:
    normalized: List[List[float]] = []
    if not isinstance(points, list):
        return normalized

    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        x = clamp(float(point[0]), 0.0, float(image_width))
        y = clamp(float(point[1]), 0.0, float(image_height))
        normalized.append(round_point(x, y))
    return normalized


def points_to_bbox_xyxy(points: Sequence[Sequence[float]]) -> Optional[List[float]]:
    if not points:
        return None

    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    x1 = min(xs)
    y1 = min(ys)
    x2 = max(xs)
    y2 = max(ys)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def bbox_xyxy_to_coco(bbox_xyxy: Sequence[float]) -> List[float]:
    x1, y1, x2, y2 = bbox_xyxy
    return [
        round(float(x1), 6),
        round(float(y1), 6),
        round(float(x2 - x1), 6),
        round(float(y2 - y1), 6),
    ]


def coco_bbox_to_xyxy(bbox: Iterable[Any], image_width: int, image_height: int) -> Optional[List[float]]:
    values = list(bbox)
    if len(values) < 4:
        return None

    x, y, width, height = (float(values[0]), float(values[1]), float(values[2]), float(values[3]))
    if width <= 0 or height <= 0:
        return None

    x1 = clamp(x, 0.0, float(image_width))
    y1 = clamp(y, 0.0, float(image_height))
    x2 = clamp(x + width, 0.0, float(image_width))
    y2 = clamp(y + height, 0.0, float(image_height))
    if x2 <= x1 or y2 <= y1:
        return None

    return [x1, y1, x2, y2]


def rectangle_points_from_bbox_xyxy(bbox_xyxy: Sequence[float]) -> List[List[float]]:
    x1, y1, x2, y2 = bbox_xyxy
    return [
        round_point(x1, y1),
        round_point(x2, y1),
        round_point(x2, y2),
        round_point(x1, y2),
    ]


def circle_to_polygon(points: Sequence[Sequence[float]],
                      image_width: int,
                      image_height: int,
                      num_vertices: int = 32) -> List[List[float]]:
    if len(points) < 2:
        return []

    cx = clamp(float(points[0][0]), 0.0, float(image_width))
    cy = clamp(float(points[0][1]), 0.0, float(image_height))
    px = clamp(float(points[1][0]), 0.0, float(image_width))
    py = clamp(float(points[1][1]), 0.0, float(image_height))
    radius = math.hypot(px - cx, py - cy)
    if radius <= 0:
        return []

    polygon: List[List[float]] = []
    for idx in range(max(8, num_vertices)):
        angle = 2.0 * math.pi * idx / max(8, num_vertices)
        x = clamp(cx + radius * math.cos(angle), 0.0, float(image_width))
        y = clamp(cy + radius * math.sin(angle), 0.0, float(image_height))
        polygon.append(round_point(x, y))
    return polygon


def build_labelme_shape(label: str,
                        points: List[List[float]],
                        shape_type: str,
                        group_id: Optional[int]) -> Dict[str, Any]:
    return {
        "label": label,
        "points": points,
        "group_id": group_id,
        "description": "",
        "shape_type": shape_type,
        "flags": {},
        "mask": None,
    }


def build_labelme_image_path(label_json_path: Path, image_path: Path) -> str:
    rel_path = Path(os_path_rel(image_path.resolve(), label_json_path.parent.resolve()))
    return str(rel_path).replace("/", "\\")


def build_labelme_payload(image_path: Path,
                          label_json_path: Path,
                          image_width: int,
                          image_height: int,
                          shapes: List[Dict[str, Any]],
                          embed_image_data: bool) -> Dict[str, Any]:
    return {
        "version": LABELME_VERSION,
        "flags": {},
        "shapes": shapes,
        "imagePath": build_labelme_image_path(label_json_path, image_path),
        "imageData": encode_image_data(image_path) if embed_image_data else None,
        "imageHeight": int(image_height),
        "imageWidth": int(image_width),
    }


def resolve_coco_image_path(input_dir: Path, file_name: str) -> Path:
    candidate = input_dir / file_name
    if candidate.exists():
        return candidate
    if not candidate.suffix:
        for ext in IMAGE_EXTENSIONS:
            fallback = candidate.with_suffix(ext)
            if fallback.exists():
                return fallback
    raise FileNotFoundError(f"未找到图像文件: {candidate}")


def segmentation_to_polygons(segmentation: Any,
                             image_width: int,
                             image_height: int) -> List[List[List[float]]]:
    polygons: List[List[List[float]]] = []
    if not isinstance(segmentation, list):
        return polygons

    for segment in segmentation:
        if not isinstance(segment, list) or len(segment) < 6 or len(segment) % 2 != 0:
            continue

        polygon: List[List[float]] = []
        for idx in range(0, len(segment), 2):
            x = clamp(float(segment[idx]), 0.0, float(image_width))
            y = clamp(float(segment[idx + 1]), 0.0, float(image_height))
            polygon.append(round_point(x, y))

        if len(polygon) >= 3:
            polygons.append(polygon)

    return polygons


def iter_coco_image_entries(images: List[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    return sorted(images, key=lambda item: int(item.get("id", 0)))


def coco_image_to_label_path(output_dir: Path, file_name: str) -> Path:
    return output_dir / Path(file_name).with_suffix(".json")


def convert_coco_to_labelme(input_dir: Path,
                            annotation_file: str,
                            labelme_subdir: str,
                            embed_image_data: bool = False) -> Dict[str, int]:
    annotation_path = input_dir / annotation_file
    if not annotation_path.exists():
        raise FileNotFoundError(f"未找到 COCO 标注文件: {annotation_path}")

    coco_data = load_json(annotation_path)
    images = coco_data.get("images", [])
    annotations = coco_data.get("annotations", [])
    categories = coco_data.get("categories", [])

    if not isinstance(images, list):
        raise ValueError("COCO images 字段必须为 list")
    if not isinstance(annotations, list):
        raise ValueError("COCO annotations 字段必须为 list")
    if not isinstance(categories, list):
        raise ValueError("COCO categories 字段必须为 list")

    category_map: Dict[int, str] = {}
    for category in categories:
        try:
            category_id = int(category["id"])
        except (KeyError, TypeError, ValueError):
            continue
        category_map[category_id] = str(category.get("name", f"class_{category_id}"))

    annotations_by_image: DefaultDict[int, List[Dict[str, Any]]] = defaultdict(list)
    for annotation in annotations:
        try:
            image_id = int(annotation["image_id"])
        except (KeyError, TypeError, ValueError):
            continue
        annotations_by_image[image_id].append(annotation)

    output_dir = input_dir / labelme_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "images": 0,
        "annotations": 0,
        "polygon_shapes": 0,
        "rectangle_shapes": 0,
    }

    for image_info in iter_coco_image_entries(images):
        try:
            image_id = int(image_info["id"])
        except (KeyError, TypeError, ValueError):
            continue

        file_name = str(image_info.get("file_name", "")).strip()
        if not file_name:
            continue

        image_path = resolve_coco_image_path(input_dir, file_name)
        label_json_path = coco_image_to_label_path(output_dir, file_name)
        label_json_path.parent.mkdir(parents=True, exist_ok=True)

        width = int(image_info.get("width") or 0)
        height = int(image_info.get("height") or 0)
        if width <= 0 or height <= 0:
            width, height = load_image_size(image_path)

        shapes: List[Dict[str, Any]] = []
        for annotation in annotations_by_image.get(image_id, []):
            try:
                category_id = int(annotation["category_id"])
            except (KeyError, TypeError, ValueError):
                category_id = -1
            label = category_map.get(category_id, f"class_{category_id}")
            annotation_id = annotation.get("id")
            group_id = None

            polygons = segmentation_to_polygons(annotation.get("segmentation"), width, height)
            if polygons:
                if len(polygons) > 1 and isinstance(annotation_id, int):
                    group_id = int(annotation_id)
                for polygon in polygons:
                    shapes.append(build_labelme_shape(label, polygon, "polygon", group_id))
                    summary["polygon_shapes"] += 1
            else:
                bbox_xyxy = coco_bbox_to_xyxy(annotation.get("bbox", []), width, height)
                if bbox_xyxy is None:
                    continue
                points = [round_point(bbox_xyxy[0], bbox_xyxy[1]), round_point(bbox_xyxy[2], bbox_xyxy[3])]
                shapes.append(build_labelme_shape(label, points, "rectangle", group_id))
                summary["rectangle_shapes"] += 1

            summary["annotations"] += 1

        payload = build_labelme_payload(
            image_path=image_path,
            label_json_path=label_json_path,
            image_width=width,
            image_height=height,
            shapes=shapes,
            embed_image_data=embed_image_data,
        )
        save_json(payload, label_json_path)
        summary["images"] += 1

    return summary


def iter_image_files(root: Path, excluded_dirs: Sequence[str]) -> Iterable[Path]:
    excluded = {name for name in excluded_dirs if name}
    for current_root, dir_names, file_names in os.walk(root):
        dir_names[:] = [name for name in dir_names if name not in excluded]
        current_path = Path(current_root)
        for file_name in sorted(file_names):
            path = current_path / file_name
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                yield path


def find_labelme_jsons(labelme_dir: Path) -> List[Path]:
    if not labelme_dir.exists():
        raise FileNotFoundError(f"未找到 LabelMe 标注目录: {labelme_dir}")
    return sorted(labelme_dir.rglob("*.json"))


def resolve_labelme_image_path(label_json_path: Path,
                               labelme_dir: Path,
                               input_dir: Path,
                               label_data: Dict[str, Any]) -> Path:
    image_path_value = str(label_data.get("imagePath") or "").strip()
    if image_path_value:
        rel_path = resolve_relative_path(image_path_value)
        candidate = (label_json_path.parent / rel_path).resolve()
        if candidate.exists():
            return candidate

        candidate = (input_dir / rel_path).resolve()
        if candidate.exists():
            return candidate

    rel_stem = label_json_path.relative_to(labelme_dir).with_suffix("")
    for ext in IMAGE_EXTENSIONS:
        candidate = input_dir / rel_stem.with_suffix(ext)
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(f"未找到与 LabelMe 标注对应的图像: {label_json_path}")


def build_labelme_records(input_dir: Path,
                          labelme_dir: Path) -> List[Dict[str, Any]]:
    label_json_paths = find_labelme_jsons(labelme_dir)
    records: List[Dict[str, Any]] = []

    for label_json_path in label_json_paths:
        label_data = load_json(label_json_path)
        image_path = resolve_labelme_image_path(label_json_path, labelme_dir, input_dir, label_data)
        width = int(label_data.get("imageWidth") or 0)
        height = int(label_data.get("imageHeight") or 0)
        if width <= 0 or height <= 0:
            width, height = load_image_size(image_path)

        file_name = image_path.relative_to(input_dir).as_posix()
        records.append({
            "image_path": image_path,
            "file_name": file_name,
            "width": width,
            "height": height,
            "label_json_path": label_json_path,
            "label_data": label_data,
        })

    return sorted(records, key=lambda item: item["file_name"])


def normalize_shape_type(shape: Dict[str, Any]) -> str:
    shape_type = str(shape.get("shape_type") or "polygon").strip().lower()
    return shape_type or "polygon"


def shape_to_bbox_only_annotation(shape: Dict[str, Any],
                                  image_id: int,
                                  category_id: int,
                                  annotation_id: int,
                                  image_width: int,
                                  image_height: int) -> Optional[Dict[str, Any]]:
    points = sanitize_points(shape.get("points"), image_width, image_height)
    bbox_xyxy = points_to_bbox_xyxy(points)
    if bbox_xyxy is None:
        return None

    bbox = bbox_xyxy_to_coco(bbox_xyxy)
    area = round(float(bbox[2] * bbox[3]), 6)
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": bbox,
        "area": area,
        "iscrowd": 0,
        "segmentation": [],
    }


def grouped_polygon_annotation(shapes: List[Dict[str, Any]],
                               label: str,
                               group_id: int,
                               image_id: int,
                               category_id: int,
                               annotation_id: int,
                               image_width: int,
                               image_height: int) -> Optional[Dict[str, Any]]:
    segmentations: List[List[float]] = []
    all_points: List[List[float]] = []
    total_area = 0.0

    for shape in shapes:
        points = sanitize_points(shape.get("points"), image_width, image_height)
        if len(points) < 3:
            continue
        segmentations.append(flatten_points(points))
        all_points.extend(points)
        total_area += polygon_area(points)

    bbox_xyxy = points_to_bbox_xyxy(all_points)
    if not segmentations or bbox_xyxy is None:
        return None

    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": bbox_xyxy_to_coco(bbox_xyxy),
        "area": round(total_area, 6),
        "iscrowd": 0,
        "segmentation": segmentations,
    }


def standalone_shape_annotation(shape: Dict[str, Any],
                                image_id: int,
                                category_id: int,
                                annotation_id: int,
                                image_width: int,
                                image_height: int) -> Optional[Dict[str, Any]]:
    shape_type = normalize_shape_type(shape)

    if shape_type == "rectangle":
        return shape_to_bbox_only_annotation(
            shape,
            image_id,
            category_id,
            annotation_id,
            image_width,
            image_height,
        )

    if shape_type == "circle":
        polygon = circle_to_polygon(shape.get("points", []), image_width, image_height)
        bbox_xyxy = points_to_bbox_xyxy(polygon)
        if not polygon or bbox_xyxy is None:
            return None
        return {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": bbox_xyxy_to_coco(bbox_xyxy),
            "area": round(polygon_area(polygon), 6),
            "iscrowd": 0,
            "segmentation": [flatten_points(polygon)],
        }

    points = sanitize_points(shape.get("points"), image_width, image_height)
    if len(points) >= 3:
        bbox_xyxy = points_to_bbox_xyxy(points)
        if bbox_xyxy is None:
            return None
        return {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": bbox_xyxy_to_coco(bbox_xyxy),
            "area": round(polygon_area(points), 6),
            "iscrowd": 0,
            "segmentation": [flatten_points(points)],
        }

    return shape_to_bbox_only_annotation(
        shape,
        image_id,
        category_id,
        annotation_id,
        image_width,
        image_height,
    )


def convert_labelme_to_coco(input_dir: Path,
                            annotation_file: str,
                            labelme_subdir: str) -> Dict[str, int]:
    labelme_dir = input_dir / labelme_subdir
    records = build_labelme_records(input_dir, labelme_dir)

    label_names = sorted({
        str(shape.get("label")).strip()
        for record in records
        for shape in record["label_data"].get("shapes", [])
        if str(shape.get("label") or "").strip()
    })
    category_name_to_id = {label: idx for idx, label in enumerate(label_names)}
    categories = [
        {"id": idx, "name": label, "supercategory": "none"}
        for label, idx in category_name_to_id.items()
    ]

    coco_images: List[Dict[str, Any]] = []
    coco_annotations: List[Dict[str, Any]] = []
    annotation_id = 1

    summary = {
        "images": 0,
        "annotations": 0,
        "categories": len(categories),
        "polygon_annotations": 0,
        "bbox_only_annotations": 0,
    }

    for image_id, record in enumerate(records, start=1):
        coco_images.append({
            "id": image_id,
            "file_name": record["file_name"],
            "width": int(record["width"]),
            "height": int(record["height"]),
        })
        summary["images"] += 1

        raw_shapes = record["label_data"].get("shapes", [])
        if not isinstance(raw_shapes, list):
            continue

        grouped_shapes: DefaultDict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
        standalone_shapes: List[Dict[str, Any]] = []

        for shape in raw_shapes:
            if not isinstance(shape, dict):
                continue
            label = str(shape.get("label") or "").strip()
            if not label or label not in category_name_to_id:
                continue

            shape_type = normalize_shape_type(shape)
            group_id = shape.get("group_id")
            if shape_type == "polygon" and isinstance(group_id, int):
                grouped_shapes[(label, group_id)].append(shape)
            else:
                standalone_shapes.append(shape)

        for (label, group_id), shapes in grouped_shapes.items():
            annotation = grouped_polygon_annotation(
                shapes=shapes,
                label=label,
                group_id=group_id,
                image_id=image_id,
                category_id=category_name_to_id[label],
                annotation_id=annotation_id,
                image_width=record["width"],
                image_height=record["height"],
            )
            if annotation is None:
                continue
            coco_annotations.append(annotation)
            annotation_id += 1
            summary["annotations"] += 1
            summary["polygon_annotations"] += 1

        for shape in standalone_shapes:
            label = str(shape.get("label") or "").strip()
            annotation = standalone_shape_annotation(
                shape=shape,
                image_id=image_id,
                category_id=category_name_to_id[label],
                annotation_id=annotation_id,
                image_width=record["width"],
                image_height=record["height"],
            )
            if annotation is None:
                continue
            coco_annotations.append(annotation)
            annotation_id += 1
            summary["annotations"] += 1
            if annotation["segmentation"]:
                summary["polygon_annotations"] += 1
            else:
                summary["bbox_only_annotations"] += 1

    coco_payload = {
        "info": {
            "description": "LabelMe to COCO conversion",
            "version": "1.0",
            "year": 2026,
            "contributor": "convert/coco2labelme.py",
            "date_created": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        },
        "licenses": [],
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": categories,
    }

    save_json(coco_payload, input_dir / annotation_file)
    return summary


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"输入路径不是目录: {input_dir}")

    if args.mode == "coco2labelme":
        summary = convert_coco_to_labelme(
            input_dir=input_dir,
            annotation_file=args.annotation_file,
            labelme_subdir=args.labelme_subdir,
            embed_image_data=args.embed_image_data,
        )
        print("\n✅ COCO→LabelMe 转换完成!")
        print(f"  - 图像: {summary['images']}")
        print(f"  - 标注: {summary['annotations']}")
        print(f"  - polygon shapes: {summary['polygon_shapes']}")
        print(f"  - rectangle shapes: {summary['rectangle_shapes']}")
        print(f"📁 输出目录：{input_dir / args.labelme_subdir}")
        return

    summary = convert_labelme_to_coco(
        input_dir=input_dir,
        annotation_file=args.annotation_file,
        labelme_subdir=args.labelme_subdir,
    )
    print("\n✅ LabelMe→COCO 转换完成!")
    print(f"  - 图像: {summary['images']}")
    print(f"  - 标注: {summary['annotations']}")
    print(f"  - 类别: {summary['categories']}")
    print(f"  - polygon annotations: {summary['polygon_annotations']}")
    print(f"  - bbox-only annotations: {summary['bbox_only_annotations']}")
    print(f"📁 输出文件：{input_dir / args.annotation_file}")


if __name__ == "__main__":
    main()
