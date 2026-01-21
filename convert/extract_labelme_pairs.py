#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 LabelMe 标注中提取包含指定标签的样本，并在更新后放回原位置。

用法示例：
  # 提取
  python convert/extract_labelme_pairs.py extract \
    --labels-root datasets/labels \
    --images-root datasets/images \
    --target-dir /tmp/round_defect \
    --label "圆形缺陷"

  # 更新（把 /tmp/round_defect 中修改过的 json 放回原位置）
  python convert/extract_labelme_pairs.py update \
    --target-dir /tmp/round_defect
"""

import argparse
import json
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


# 将项目根目录加入 sys.path，便于导入 utils
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

from utils import read_labelme_json, save_labelme_json  # noqa: E402
from utils.constants import IMAGE_EXTENSIONS  # noqa: E402


def iter_labelme_jsons(labels_root: Path) -> Iterable[Path]:
    for path in labels_root.rglob("*.json"):
        if path.is_file():
            yield path


def has_target_label(shapes: List[Dict], target: str, match_mode: str) -> bool:
    if not shapes:
        return False
    for shape in shapes:
        label = str(shape.get("label", ""))
        if match_mode == "exact" and label == target:
            return True
        if match_mode == "contains" and target in label:
            return True
        if match_mode == "regex" and re.search(target, label):
            return True
    return False


def compute_rel_without_label(json_path: Path, labels_root: Path) -> Path:
    rel = json_path.relative_to(labels_root)
    parts = list(rel.parts)
    if len(parts) >= 2 and parts[-2].lower() == "label":
        parts.pop(-2)
    return Path(*parts)


def resolve_image_path(
    json_path: Path,
    labels_root: Path,
    images_root: Path,
    image_path_from_json: Optional[str],
) -> Optional[Path]:
    rel_no_label = compute_rel_without_label(json_path, labels_root)
    rel_dir = rel_no_label.parent
    img_dir = images_root / rel_dir

    candidates: List[Path] = []

    if image_path_from_json:
        img_path = Path(image_path_from_json)
        if img_path.is_absolute() and img_path.exists():
            return img_path

        # LabelMe 常见情况：imagePath 为相对路径或包含 ..\\
        image_name = Path(image_path_from_json).name
        if image_name:
            candidates.append(img_dir / image_name)
            candidates.append(json_path.parent / image_name)

    # 兜底：按 json 文件名搜索
    stem = json_path.stem
    for ext in IMAGE_EXTENSIONS:
        candidates.append(img_dir / f"{stem}{ext}")
        candidates.append(json_path.parent / f"{stem}{ext}")

    for path in candidates:
        if path.exists():
            return path

    return None


def ensure_unique_base(target_dir: Path, base: str, exts: List[str], overwrite: bool) -> str:
    if overwrite:
        return base

    candidate = base
    counter = 1
    while True:
        conflict = False
        for ext in exts:
            if (target_dir / f"{candidate}{ext}").exists():
                conflict = True
                break
        if not conflict:
            return candidate
        candidate = f"{base}_{counter}"
        counter += 1


def build_output_names(rel_no_label: Path, image_ext: str, target_dir: Path, overwrite: bool) -> Tuple[str, str]:
    base = "_".join(rel_no_label.with_suffix("").parts)
    base = ensure_unique_base(target_dir, base, [".json", image_ext], overwrite)
    return f"{base}.json", f"{base}{image_ext}"


def write_manifest(manifest_path: Path, payload: Dict):
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_manifest(manifest_path: Path) -> Dict:
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_mode(args: argparse.Namespace) -> int:
    labels_root = Path(args.labels_root).resolve()
    images_root = Path(args.images_root).resolve()
    target_dir = Path(args.target_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    records = []
    total = 0
    matched = 0
    extracted = 0
    missing_image = 0

    for json_path in iter_labelme_jsons(labels_root):
        total += 1
        try:
            data = read_labelme_json(str(json_path))
        except Exception as exc:
            print(f"[WARN] 读取失败: {json_path} ({exc})")
            continue

        original_image_path = data.get("imagePath")
        shapes = data.get("shapes", [])
        if not has_target_label(shapes, args.label, args.match):
            continue

        matched += 1
        image_path = resolve_image_path(json_path, labels_root, images_root, data.get("imagePath"))
        if not image_path:
            missing_image += 1
            print(f"[WARN] 未找到图片: {json_path}")
            if not args.allow_missing_image:
                continue

        rel_no_label = compute_rel_without_label(json_path, labels_root)
        image_ext = image_path.suffix if image_path else ".png"
        out_json_name, out_img_name = build_output_names(rel_no_label, image_ext, target_dir, args.overwrite)

        out_json_path = target_dir / out_json_name
        out_img_path = target_dir / out_img_name

        if args.dry_run:
            print(f"[DRY] {json_path} -> {out_json_path}")
            if image_path:
                print(f"[DRY] {image_path} -> {out_img_path}")
            extracted += 1
        else:
            if image_path:
                shutil.copy2(image_path, out_img_path)

            if args.rewrite_imagepath and image_path:
                data["imagePath"] = out_img_name

            save_labelme_json(data, str(out_json_path))
            extracted += 1

        rel_image = None
        if image_path:
            try:
                rel_image = str(image_path.relative_to(images_root))
            except ValueError:
                rel_image = None

        records.append({
            "src_json": str(json_path),
            "src_image": str(image_path) if image_path else None,
            "rel_json": str(json_path.relative_to(labels_root)),
            "rel_image": rel_image,
            "extracted_json": out_json_name,
            "extracted_image": out_img_name if image_path else None,
            "image_path_original": original_image_path,
        })

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "labels_root": str(labels_root),
        "images_root": str(images_root),
        "target_dir": str(target_dir),
        "label": args.label,
        "match_mode": args.match,
        "rewrite_imagepath": args.rewrite_imagepath,
        "records": records,
    }

    if not args.dry_run:
        manifest_path = target_dir / args.manifest_name
        write_manifest(manifest_path, manifest)

    print("\n=== 提取完成 ===")
    print(f"总扫描: {total}")
    print(f"匹配到标签: {matched}")
    print(f"成功提取: {extracted}")
    if missing_image:
        print(f"缺失图片: {missing_image}")
    if not args.dry_run:
        print(f"清单文件: {target_dir / args.manifest_name}")

    return 0


def restore_original_image_path(data: Dict, record: Dict) -> None:
    if record.get("image_path_original"):
        data["imagePath"] = record["image_path_original"]


def update_mode(args: argparse.Namespace) -> int:
    target_dir = Path(args.target_dir).resolve()
    manifest_path = target_dir / args.manifest_name

    if not manifest_path.exists() and not args.use_filename_reconstruct:
        print(f"[ERROR] 未找到清单文件: {manifest_path}")
        print("可使用 --use-filename-reconstruct 进行文件名还原（可能不可靠）。")
        return 1

    records = []
    labels_root = Path(args.labels_root).resolve() if args.labels_root else None
    if manifest_path.exists():
        manifest = load_manifest(manifest_path)
        records = manifest.get("records", [])
        if labels_root is None and manifest.get("labels_root"):
            labels_root = Path(manifest.get("labels_root")).resolve()
    else:
        manifest = {}

    updated = 0
    missing = 0

    if records:
        for record in records:
            extracted_json = record.get("extracted_json")
            if not extracted_json:
                continue
            extracted_path = target_dir / extracted_json
            if not extracted_path.exists():
                missing += 1
                print(f"[WARN] 未找到提取文件: {extracted_path}")
                continue

            src_json = Path(record["src_json"]) if record.get("src_json") else None
            if (not src_json or not src_json.exists()) and labels_root and record.get("rel_json"):
                src_json = labels_root / record["rel_json"]

            if not src_json:
                print(f"[WARN] 无法定位原始位置: {extracted_path}")
                continue

            if args.dry_run:
                print(f"[DRY] {extracted_path} -> {src_json}")
                updated += 1
                continue

            try:
                data = read_labelme_json(str(extracted_path))
            except Exception as exc:
                print(f"[WARN] 读取失败: {extracted_path} ({exc})")
                continue

            restore_original_image_path(data, record)
            save_labelme_json(data, str(src_json))
            updated += 1

    elif args.use_filename_reconstruct:
        if not args.labels_root:
            print("[ERROR] 需要提供 --labels-root 以进行文件名还原。")
            return 1

        labels_root = Path(args.labels_root).resolve()
        json_files = [p for p in target_dir.glob("*.json") if p.name != args.manifest_name]
        for extracted_path in json_files:
            stem = extracted_path.stem
            parts = stem.split("_")
            if len(parts) < 2:
                print(f"[WARN] 文件名无法还原路径: {extracted_path}")
                continue

            # 尝试还原: labels_root / parts[0] / parts[1] / label / 原文件名.json
            rel_path = Path(*parts[:-1]) / "label" / f"{parts[-1]}.json"
            src_json = labels_root / rel_path

            if args.dry_run:
                print(f"[DRY] {extracted_path} -> {src_json}")
                updated += 1
                continue

            try:
                data = read_labelme_json(str(extracted_path))
            except Exception as exc:
                print(f"[WARN] 读取失败: {extracted_path} ({exc})")
                continue

            save_labelme_json(data, str(src_json))
            updated += 1
    else:
        print("[ERROR] 未提供可用记录。")
        return 1

    print("\n=== 更新完成 ===")
    print(f"已更新: {updated}")
    if missing:
        print(f"缺失文件: {missing}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="提取包含指定标签的 LabelMe 样本，并将更新后的 json 放回原位置。"
    )

    subparsers = parser.add_subparsers(dest="mode", required=True)

    p_extract = subparsers.add_parser("extract", help="提取包含指定标签的样本")
    p_extract.add_argument("--labels-root", default="datasets/labels", help="labels 根目录")
    p_extract.add_argument("--images-root", default="datasets/images", help="images 根目录")
    p_extract.add_argument("--target-dir", required=True, help="输出目录（json 与 image 放在同一目录）")
    p_extract.add_argument("--label", required=True, help="目标标签名，例如：圆形缺陷")
    p_extract.add_argument("--match", choices=["exact", "contains", "regex"], default="exact", help="匹配方式")
    p_extract.add_argument("--overwrite", action="store_true", help="覆盖已存在文件")
    p_extract.add_argument("--rewrite-imagepath", action="store_true", default=True, help="提取后重写 imagePath 为输出文件名")
    p_extract.add_argument("--no-rewrite-imagepath", dest="rewrite_imagepath", action="store_false", help="不重写 imagePath")
    p_extract.add_argument("--manifest-name", default="extraction_manifest.json", help="清单文件名")
    p_extract.add_argument("--allow-missing-image", action="store_true", help="允许缺失图片仍提取 json")
    p_extract.add_argument("--dry-run", action="store_true", help="仅打印不实际复制")

    p_update = subparsers.add_parser("update", help="将更新后的 json 放回原位置")
    p_update.add_argument("--target-dir", required=True, help="提取输出目录")
    p_update.add_argument("--manifest-name", default="extraction_manifest.json", help="清单文件名")
    p_update.add_argument("--labels-root", default=None, help="清单缺失时用于还原路径")
    p_update.add_argument("--use-filename-reconstruct", action="store_true", help="无清单时使用文件名还原路径")
    p_update.add_argument("--dry-run", action="store_true", help="仅打印不实际复制")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "extract":
        return extract_mode(args)
    if args.mode == "update":
        return update_mode(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
