#!/usr/bin/env python3
"""
扫描 LabelMe JSON 标注文件，定位 imageData 为空导致 LabelMe 无法打开的问题，
并尝试自动从原始图像补全 imageData。
默认路径与 convert/mergeLabel.py 保持一致，亦可通过命令行参数覆盖。
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Optional, Tuple

# 与 mergeLabel.py 保持一致，便于直接套用相同的数据目录
DEFAULT_LABEL_ROOT = Path("/datasets/PAR/Xray/self/1120/1215adjust")
DEFAULT_DATASETS = ["D1", "D2", "D3", "D4", "img20250608", "img20250609"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="检查并修复 LabelMe JSON 的 imageData 字段缺失问题"
    )
    parser.add_argument(
        "--label-root",
        type=Path,
        default=DEFAULT_LABEL_ROOT,
        help="待检查标签根目录（默认为 mergeLabel.py 使用的输出目录）",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="要处理的数据集名称，默认与 mergeLabel.py 相同；传入 auto 则自动遍历 label-root 下的所有数据集目录",
    )
    parser.add_argument(
        "--label-files",
        nargs="*",
        default=None,
        help="仅处理指定 JSON 文件（可为绝对路径或相对 label-root 的路径），提供该参数时将忽略 --datasets",
    )
    parser.add_argument(
        "--image-roots",
        nargs="*",
        default=None,
        help="搜索原始图像时的候选根目录，默认自动根据 label-root 推断（含 label-root、本地 labeled 目录等）",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="在定位到缺失 imageData 的同时尝试写回修复",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅输出检查结果，不写入文件（即使指定了 --fix）",
    )
    return parser.parse_args()


def normalize_rel_path(path_value: Optional[str]) -> Optional[PurePosixPath]:
    if not path_value:
        return None
    return PurePosixPath(path_value.replace("\\", "/"))


def build_default_image_roots(label_root: Path) -> List[Path]:
    """根据标签目录推断可用的图像根目录"""
    candidates: List[Path] = []
    roots_to_try = [
        label_root,
        label_root.parent / "labeled",
    ]

    # /datasets/PAR/Xray/self/1120/1215adjust -> parents[2] == /datasets/PAR/Xray
    # 尝试 self 同级的 1210reback/1120 目录
    parents = list(label_root.parents)
    if len(parents) >= 3:
        candidate = parents[2] / "1210reback" / "1120"
        roots_to_try.append(candidate)

    for path in roots_to_try:
        if path.exists() and path not in candidates:
            candidates.append(path)
    return candidates


def collect_label_files(
    label_root: Path, datasets: Optional[Iterable[str]], explicit_files: Optional[List[str]]
) -> List[Path]:
    if explicit_files:
        files: List[Path] = []
        for file_arg in explicit_files:
            path = Path(file_arg)
            if not path.is_absolute():
                path = (label_root / file_arg).resolve()
            if path.is_file():
                files.append(path)
            else:
                print(f"[WARN] 指定的标签文件不存在: {path}", file=sys.stderr)
        return files

    if not datasets:
        datasets = DEFAULT_DATASETS
    elif len(datasets) == 1 and datasets[0].lower() == "auto":
        datasets = sorted(
            p.name for p in label_root.iterdir() if p.is_dir() and (p / "label").exists()
        )

    label_files: List[Path] = []
    for dataset in datasets:
        label_dir = label_root / dataset / "label"
        if not label_dir.exists():
            print(f"[WARN] 标签目录不存在: {label_dir}", file=sys.stderr)
            continue
        label_files.extend(sorted(label_dir.glob("*.json")))
    return label_files


def load_json(path: Path) -> Tuple[Optional[dict], Optional[str]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f), None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def locate_image_file(
    json_path: Path,
    normalized_rel: Optional[PurePosixPath],
    image_roots: List[Path],
    cache: Dict[str, Optional[Path]],
) -> Optional[Path]:
    dataset_name = json_path.parent.parent.name
    file_name = None

    if normalized_rel:
        rel_path = Path(str(normalized_rel))
        file_name = rel_path.name
        candidate = (json_path.parent / rel_path).resolve()
        if candidate.exists():
            return candidate
    if not file_name:
        file_name = json_path.stem

    # 先尝试 dataset 子目录
    for root in image_roots:
        dataset_dir = root / dataset_name
        candidate = dataset_dir / file_name
        if candidate.exists():
            return candidate
        # 例如 root 直接存放原图
        candidate = root / file_name
        if candidate.exists():
            return candidate

    # 避免多次递归搜索
    cache_key = f"{file_name}|{dataset_name}"
    if cache_key in cache:
        return cache[cache_key]

    for root in image_roots:
        try:
            for hit in root.rglob(file_name):
                cache[cache_key] = hit
                return hit
        except PermissionError:
            continue

    cache[cache_key] = None
    return None


def encode_image(path: Path) -> str:
    data = path.read_bytes()
    return base64.b64encode(data).decode("utf-8")


def process_label(
    json_path: Path, image_roots: List[Path], allow_fix: bool, dry_run: bool, cache: Dict[str, Optional[Path]]
) -> Tuple[str, List[str]]:
    data, error = load_json(json_path)
    issues: List[str] = []

    if error:
        issues.append(f"json_error: {error}")
        return "error", issues

    image_data = data.get("imageData")
    if not isinstance(image_data, str) or not image_data.strip():
        issues.append("imageData_missing")
        if allow_fix and not dry_run:
            normalized_rel = normalize_rel_path(data.get("imagePath"))
            image_file = locate_image_file(json_path, normalized_rel, image_roots, cache)
            if image_file:
                try:
                    data["imageData"] = encode_image(image_file)
                    with json_path.open("w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    issues.append(f"fixed_with_image: {image_file}")
                    return "fixed", issues
                except Exception as exc:  # noqa: BLE001
                    issues.append(f"fix_failed: {exc}")
            else:
                issues.append("image_file_not_found")
        return "needs_fix", issues

    return "ok", issues


def main() -> None:
    args = parse_args()
    label_root = args.label_root.resolve()
    if not label_root.exists():
        print(f"[ERROR] 标签根目录不存在: {label_root}", file=sys.stderr)
        sys.exit(1)

    image_roots = (
        [Path(p).resolve() for p in args.image_roots]
        if args.image_roots
        else build_default_image_roots(label_root)
    )
    if not image_roots:
        print("[ERROR] 未找到可用的图像根目录，可通过 --image-roots 指定", file=sys.stderr)
        sys.exit(1)

    label_files = collect_label_files(label_root, args.datasets, args.label_files)
    if not label_files:
        print("[WARN] 未发现任何需要处理的标签文件", file=sys.stderr)
        return

    cache: Dict[str, Optional[Path]] = {}
    stats = defaultdict(int)

    print(f"检查标签总数: {len(label_files)}")
    print(f"图像搜索根目录: {', '.join(str(p) for p in image_roots)}")

    for json_path in label_files:
        status, issues = process_label(json_path, image_roots, args.fix, args.dry_run, cache)
        stats[status] += 1
        if issues:
            issue_text = "; ".join(issues)
            print(f"[{status.upper()}] {json_path}: {issue_text}")

    print("\n处理统计：")
    for key in ["ok", "needs_fix", "fixed", "error"]:
        if stats[key]:
            print(f"  {key}: {stats[key]}")


if __name__ == "__main__":
    main()
