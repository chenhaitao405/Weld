#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""从 verified_image 目录回溯原始 LabelMe 标签."""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import yaml

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def resolve_path(value: str, base: Optional[str] = None) -> str:
    value = os.path.expanduser(value)
    if os.path.isabs(value):
        return os.path.abspath(value)
    base = base or os.getcwd()
    return os.path.abspath(os.path.join(base, value))


def load_profile(config_path: str, profile_name: Optional[str]) -> Dict[str, any]:
    cfg_path = Path(resolve_path(config_path))
    if not cfg_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    profiles = config.get("profiles") or {}
    if not profiles:
        raise ValueError("配置文件缺少 profiles 定义")

    profile_name = profile_name or config.get("default_profile")
    if profile_name not in profiles:
        raise KeyError(f"配置文件中没有 profile: {profile_name}")

    profile = profiles[profile_name]
    paths = profile.get("paths") or {}
    base_path = resolve_path(paths.get("base_path"))
    json_base = resolve_path(paths.get("json_base_path"))
    datasets = profile.get("datasets") or []
    if not datasets:
        raise ValueError(f"profile {profile_name} 未配置 datasets")

    return {
        "config_path": str(cfg_path),
        "profile": profile_name,
        "base_path": base_path,
        "json_base_path": json_base,
        "datasets": datasets,
    }


def guess_dataset(filename: str, datasets: List[str]) -> Optional[str]:
    name_lower = filename.lower()
    for ds in datasets:
        token = ds.lower()
        if name_lower.startswith(f"{token}_") or name_lower.startswith(f"{token}-"):
            return ds
    return None


def collect_images(image_dir: Path) -> List[Path]:
    results: List[Path] = []
    for path in sorted(image_dir.iterdir()):
        if path.is_dir():
            continue
        if path.suffix.lower() in SUPPORTED_IMAGE_EXTS:
            results.append(path)
    return results


def _strip_dataset_prefix(stem: str, dataset: str) -> str:
    candidates = [f"{dataset}_", f"{dataset}-"]
    for prefix in candidates:
        if stem.startswith(prefix):
            return stem[len(prefix):]
    return stem


def locate_label(json_base_path: str, dataset: str, image_name: str) -> Path:
    label_dir = Path(json_base_path) / dataset / "label"
    stem = Path(image_name).stem
    stripped = _strip_dataset_prefix(stem, dataset)
    return label_dir / f"{stripped}.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="根据 profile 设置将 verified_image 中的图片回溯到原 LabelMe 标签"
    )
    parser.add_argument("--image-dir", required=True, help="verified_image 目录")
    parser.add_argument("--config-path", default="../configs/pipeline_profiles.yaml",
                        help="pipeline 配置文件路径")
    parser.add_argument("--profile", required=True,
                        help="用于解析原始路径的数据 profile")
    args = parser.parse_args()

    profile = load_profile(args.config_path, args.profile)
    image_dir = Path(resolve_path(args.image_dir))
    if not image_dir.exists():
        raise FileNotFoundError(f"image_dir 不存在: {image_dir}")

    output_label_dir = image_dir / "label"
    output_label_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(image_dir)
    if not images:
        print("未找到任何图片文件")
        return

    copied = 0
    missing_dataset = 0
    missing_label = 0

    for image_path in images:
        dataset = guess_dataset(image_path.name, profile["datasets"])
        if not dataset:
            missing_dataset += 1
            print(f"[跳过] 无法从文件名推断数据集: {image_path.name}")
            continue

        src_label = locate_label(profile["json_base_path"], dataset, image_path.name)
        if not src_label.exists():
            missing_label += 1
            print(f"[缺失] 找不到标签: {src_label}")
            continue

        target_label = output_label_dir / f"{image_path.stem}.json"
        target_label.write_text(src_label.read_text(encoding="utf-8"), encoding="utf-8")
        copied += 1

    print("复制完成:"
          f"\n  profile: {profile['profile']}"
          f"\n  标签输出: {output_label_dir}"
          f"\n  成功复制: {copied}"
          f"\n  未识别数据集: {missing_dataset}"
          f"\n  标签缺失: {missing_label}")


if __name__ == "__main__":
    main()
