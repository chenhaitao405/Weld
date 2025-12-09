#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将 YOLO 数据集中的 BMP 图像批量转换为 JPG，以降低磁盘占用。

示例：
  python convert/convertYOLOjpg.py \
      --dataset-dir /home/lenovo/code/CHT/datasets/Xray/opensource/SWRD8bit/swr_pipeline/ROI \
      --output-dir /home/lenovo/code/CHT/datasets/Xray/opensource/SWRD8bit/swr_pipeline/ROI_jpg \
      --quality 90
"""

import argparse
import sys
import shutil
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("本脚本需要 Pillow，请先执行 `pip install pillow`")
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 YOLO 数据集( images/labels )中的 BMP 图像转为 JPG 保存",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--dataset-dir", required=True,
                        help="YOLO 数据集根目录（包含 images/ 与 labels/）")
    parser.add_argument("--output-dir",
                        help="转换后的完整 YOLO 数据集输出目录（默认为覆盖至输入目录）")
    parser.add_argument("--images-subdir", default="images",
                        help="图像所在子目录名称")
    parser.add_argument("--quality", type=int, default=95,
                        help="JPG 保存质量（1-100）")
    parser.add_argument("--keep-original", action="store_true",
                        help="保留原始 BMP 文件（默认转换后删除 BMP）")
    parser.add_argument("--overwrite", action="store_true",
                        help="若 JPG 已存在则覆盖重写（默认跳过已有 JPG）")
    return parser.parse_args()


def convert_bmp_to_jpg(source_path: Path,
                       destination_path: Path,
                       quality: int,
                       delete_source: bool,
                       overwrite: bool) -> bool:
    if destination_path.exists():
        if not overwrite:
            return False
        destination_path.unlink()

    destination_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(source_path) as img:
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        elif img.mode == "L":
            img = img.convert("RGB")
        img.save(destination_path, format="JPEG", quality=quality, optimize=True)

    if delete_source and source_path.exists():
        source_path.unlink()

    return True


def copy_labels_and_yaml(source_root: Path, target_root: Path, overwrite: bool):
    labels_src = source_root / "labels"
    labels_dst = target_root / "labels"
    if labels_src.exists() and labels_src.is_dir():
        if labels_dst.exists():
            if overwrite:
                shutil.rmtree(labels_dst)
            else:
                print(f"⚠️ labels 目录已存在于 {labels_dst}，跳过复制（使用 --overwrite 可强制覆盖）")
                labels_dst = None
        if labels_dst is not None:
            shutil.copytree(labels_src, labels_dst, dirs_exist_ok=True)

    yaml_src = source_root / "dataset.yaml"
    if yaml_src.exists():
        yaml_dst = target_root / "dataset.yaml"
        if yaml_dst.exists() and not overwrite:
            print(f"⚠️ dataset.yaml 已存在于 {yaml_dst}，跳过覆盖（使用 --overwrite 可重写）")
        else:
            yaml_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(yaml_src, yaml_dst)


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    images_root = dataset_dir / args.images_subdir

    if not images_root.exists():
        raise FileNotFoundError(f"未找到图像目录：{images_root}")

    target_root = Path(args.output_dir).expanduser().resolve() if args.output_dir else dataset_dir
    inplace = target_root == dataset_dir or args.output_dir is None
    images_target_root = target_root / args.images_subdir

    if not inplace:
        if target_root.exists() and any(target_root.iterdir()) and not args.overwrite:
            raise FileExistsError(f"输出目录 {target_root} 已存在，使用 --overwrite 允许覆盖")
        target_root.mkdir(parents=True, exist_ok=True)
        if images_target_root.exists() and args.overwrite:
            shutil.rmtree(images_target_root)
        images_target_root.mkdir(parents=True, exist_ok=True)
        copy_labels_and_yaml(dataset_dir, target_root, args.overwrite)

    bmp_files = []
    other_files = []
    for path in images_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() == ".bmp":
            bmp_files.append(path)
        else:
            other_files.append(path)

    if not bmp_files:
        print(f"未在 {images_root} 下找到 BMP 图像。")
        # 如果需要复制完整数据集且无 bmp，也需复制非bmp
        if not inplace:
            copied = 0
            for file_path in other_files:
                rel_path = file_path.relative_to(images_root)
                dst = images_target_root / rel_path
                if dst.exists() and not args.overwrite:
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file_path, dst)
                copied += 1
            print(f"✅ 复制完成：{copied} 张非 BMP 图像。")
        return

    print(f"在 {images_root} 下找到 {len(bmp_files)} 张 BMP，即将转换为 JPG …")
    converted = skipped = failed = 0

    delete_source = (not args.keep_original) and inplace

    for bmp_path in bmp_files:
        rel_path = bmp_path.relative_to(images_root)
        dst_path = images_target_root / rel_path.with_suffix(".jpg")
        try:
            changed = convert_bmp_to_jpg(
                bmp_path, dst_path, args.quality, delete_source, args.overwrite or not dst_path.exists()
            )
        except Exception as exc:
            failed += 1
            print(f"❌ 转换失败 {bmp_path}: {exc}")
            continue

        if changed:
            converted += 1
        else:
            skipped += 1

    copied_non_bmp = 0
    if not inplace:
        for file_path in other_files:
            rel_path = file_path.relative_to(images_root)
            dst = images_target_root / rel_path
            if dst.exists() and not args.overwrite:
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, dst)
            copied_non_bmp += 1

    print(f"✅ 转换完成：{converted} 张已转换，{skipped} 张已存在 JPG 被跳过，{failed} 张失败。")
    if not inplace:
        print(f"📁 新数据集路径：{target_root}")
        print(f"  - 复制非 BMP 图像：{copied_non_bmp} 张")


if __name__ == "__main__":
    main()
