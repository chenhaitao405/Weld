"""统一的标注读取模块，支持LabelMe与COCO格式。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from utils.label_processing import read_labelme_json


@dataclass
class AnnotationRecord:
    """统一对外的标注实例结构。"""

    annotation_id: str
    class_id: Optional[int]
    class_name: str
    bbox: List[float]
    polygon: List[List[float]]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def polygon_to_bbox(points: Sequence[Sequence[float]]) -> Optional[List[float]]:
    if not points:
        return None
    xs = [float(pt[0]) for pt in points]
    ys = [float(pt[1]) for pt in points]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


class AnnotationLoader:
    """根据配置选择LabelMe或COCO数据源并提供统一接口。"""

    def __init__(self,
                 label_format: str,
                 image_root: Optional[Path] = None,
                 label_root: Optional[Path] = None,
                 label_extension: str = ".json",
                 image_dir_token: Optional[str] = "images",
                 label_dir_token: Optional[str] = "label",
                 coco_json: Optional[Path] = None):
        self.label_format = label_format.lower()
        self.image_root = image_root
        self.label_root = label_root
        self.label_extension = label_extension
        self.image_dir_token = image_dir_token
        self.label_dir_token = label_dir_token

        if self.label_format == "labelme":
            if self.label_root is None:
                raise ValueError("labelme 格式需要提供 label_root")
        elif self.label_format == "coco":
            if coco_json is None:
                raise ValueError("coco 格式需要提供 coco_json")
            self._init_coco_index(coco_json)
        else:
            raise ValueError(f"不支持的标注格式: {label_format}")

    # ---------------------------- LabelMe ----------------------------

    def _resolve_labelme_path(self, image_path: Path) -> Path:
        if self.image_root and image_path.is_relative_to(self.image_root):
            rel_path = image_path.relative_to(self.image_root)
        else:
            rel_path = Path(image_path.name)

        rel_path = rel_path.with_suffix(self.label_extension)
        if self.image_dir_token and self.label_dir_token:
            parts = list(rel_path.parts)
            for idx, part in enumerate(parts[:-1]):
                if part == self.image_dir_token:
                    parts[idx] = self.label_dir_token
                    break
            rel_path = Path(*parts)
        return self.label_root / rel_path

    def _load_labelme(self, image_path: Path) -> List[AnnotationRecord]:
        json_path = self._resolve_labelme_path(image_path)
        if not json_path.exists():
            return []

        data = read_labelme_json(str(json_path))
        shapes = data.get("shapes", [])
        records: List[AnnotationRecord] = []
        for idx, shape in enumerate(shapes):
            points = _shape_to_points(shape)
            bbox = polygon_to_bbox(points)
            if bbox is None:
                continue
            label = str(shape.get("label", ""))
            records.append(AnnotationRecord(
                annotation_id=f"labelme_{json_path.stem}_{idx}",
                class_id=None,
                class_name=label,
                bbox=bbox,
                polygon=[[float(px), float(py)] for px, py in points],
                metadata={
                    "source": "labelme",
                    "shape_type": shape.get("shape_type"),
                    "json_path": str(json_path)
                }
            ))
        return records

    # ----------------------------- COCO -----------------------------

    def _init_coco_index(self, coco_json: Path):
        with open(coco_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        images = data.get("images", [])
        annotations = data.get("annotations", [])
        categories = data.get("categories", [])

        self._coco_image_map: Dict[str, Dict[str, Any]] = {}
        self._coco_ann_map: Dict[int, List[Dict[str, Any]]] = {}
        self._coco_cat_map: Dict[int, Dict[str, Any]] = {
            int(cat["id"]): cat for cat in categories if "id" in cat
        }

        for img in images:
            file_name = str(img.get("file_name"))
            if self.image_root:
                rel_path = Path(file_name)
                self._coco_image_map[rel_path.as_posix()] = img
            self._coco_image_map[file_name] = img
            self._coco_ann_map[int(img["id"])] = []

        for ann in annotations:
            image_id = int(ann.get("image_id"))
            self._coco_ann_map.setdefault(image_id, []).append(ann)

    def _load_coco(self, image_path: Path) -> List[AnnotationRecord]:
        image_info = self._lookup_coco_image(image_path)
        if image_info is None:
            return []

        image_id = int(image_info["id"])
        anns = self._coco_ann_map.get(image_id, [])
        records: List[AnnotationRecord] = []
        for ann in anns:
            bbox_xywh = ann.get("bbox")
            if not bbox_xywh or len(bbox_xywh) != 4:
                continue
            x, y, w, h = bbox_xywh
            if w <= 0 or h <= 0:
                continue
            bbox = [float(x), float(y), float(x + w), float(y + h)]
            segmentation = ann.get("segmentation") or []
            polygon = _coco_segmentation_to_polygon(segmentation)
            if not polygon:
                polygon = [
                    [bbox[0], bbox[1]],
                    [bbox[2], bbox[1]],
                    [bbox[2], bbox[3]],
                    [bbox[0], bbox[3]]
                ]
            cat_id = ann.get("category_id")
            cat_info = self._coco_cat_map.get(int(cat_id)) if cat_id is not None else None
            class_name = str(cat_info.get("name", cat_id)) if cat_info else str(cat_id)
            records.append(AnnotationRecord(
                annotation_id=f"coco_{image_id}_{ann.get('id', len(records))}",
                class_id=int(cat_id) if cat_id is not None else None,
                class_name=class_name,
                bbox=bbox,
                polygon=[[float(px), float(py)] for px, py in polygon],
                metadata={
                    "source": "coco",
                    "annotation": ann
                }
            ))
        return records

    def _lookup_coco_image(self, image_path: Path) -> Optional[Dict[str, Any]]:
        candidates: List[str] = []
        if self.image_root and image_path.is_relative_to(self.image_root):
            rel = image_path.relative_to(self.image_root)
            candidates.append(rel.as_posix())
        candidates.append(image_path.name)
        for candidate in candidates:
            if candidate in self._coco_image_map:
                return self._coco_image_map[candidate]
        return None

    # ------------------------------ API -----------------------------

    def load(self, image_path: Path) -> List[AnnotationRecord]:
        if self.label_format == "labelme":
            return self._load_labelme(image_path)
        if self.label_format == "coco":
            return self._load_coco(image_path)
        return []


def _shape_to_points(shape: Dict[str, Any]) -> List[Tuple[float, float]]:
    pts = shape.get("points")
    if not pts and shape.get("shape_type") == "rectangle":
        pts = shape.get("points", [])
    if not pts:
        return []
    if len(pts) == 2:
        (x1, y1), (x2, y2) = pts
        return [
            (float(x1), float(y1)),
            (float(x2), float(y1)),
            (float(x2), float(y2)),
            (float(x1), float(y2)),
        ]
    return [(float(p[0]), float(p[1])) for p in pts]


def _coco_segmentation_to_polygon(segmentation: Any) -> List[Tuple[float, float]]:
    if isinstance(segmentation, list) and segmentation:
        first = segmentation[0]
        if isinstance(first, (list, tuple)):
            coords = first
        else:
            coords = segmentation
        if len(coords) % 2 != 0:
            coords = coords[:-1]
        polygon: List[Tuple[float, float]] = []
        for idx in range(0, len(coords), 2):
            polygon.append((float(coords[idx]), float(coords[idx + 1])))
        return polygon
    if isinstance(segmentation, dict):
        counts = segmentation.get("counts")
        if counts:
            # RLE暂不展开，回退为bbox
            return []
    return []


def annotations_to_dict(records: Iterable[AnnotationRecord]) -> List[Dict[str, Any]]:
    return [rec.to_dict() for rec in records]

