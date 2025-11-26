#!/usr/bin/env python3
"""
YOLO数据集旋转处理脚本
功能：将竖图（高度>宽度）逆时针旋转90度，同时转换对应的YOLO标签坐标
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
from PIL import Image
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description='旋转YOLO数据集中的竖图为横图')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='输入YOLO数据集路径')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='输出YOLO数据集路径')
    return parser.parse_args()


def rotate_yolo_label(label_content: str) -> str:
    """
    逆时针旋转90度时的YOLO分割标签坐标转换

    YOLO分割格式: class_id x1 y1 x2 y2 x3 y3 ... xn yn (归一化的多边形顶点坐标)

    逆时针旋转90度时，每个点的坐标转换:
    - x_new = y_old
    - y_new = 1 - x_old
    """
    new_lines = []

    for line in label_content.strip().split('\n'):
        if not line.strip():
            continue

        parts = line.strip().split()
        if len(parts) < 3:  # 至少需要 class_id 和一个点
            continue

        class_id = parts[0]
        coords = parts[1:]  # 剩余的都是坐标

        # 坐标应该是成对的 (x, y)
        if len(coords) % 2 != 0:
            print(f"警告: 坐标数量不是偶数，跳过此行: {line[:50]}...")
            continue

        new_coords = []
        for i in range(0, len(coords), 2):
            x_old = float(coords[i])
            y_old = float(coords[i + 1])

            # 逆时针旋转90度的坐标转换
            x_new = y_old
            y_new = 1.0 - x_old

            new_coords.append(f"{x_new:.6f}")
            new_coords.append(f"{y_new:.6f}")

        new_line = class_id + " " + " ".join(new_coords)
        new_lines.append(new_line)

    return '\n'.join(new_lines)


def process_image_and_label(img_src: Path, label_src: Path,
                            img_dst: Path, label_dst: Path) -> dict:
    """
    处理单张图片及其标签
    返回处理信息字典
    """
    result = {
        'image': str(img_src),
        'rotated': False,
        'error': None
    }

    try:
        # 读取图片
        with Image.open(img_src) as img:
            width, height = img.size

            # 判断是否需要旋转（高度 > 宽度）
            if height > width:
                # 逆时针旋转90度
                rotated_img = img.rotate(90, expand=True)
                rotated_img.save(img_dst)
                result['rotated'] = True
                result['original_size'] = (width, height)
                result['new_size'] = rotated_img.size

                # 处理对应的标签文件
                if label_src.exists():
                    with open(label_src, 'r') as f:
                        label_content = f.read()

                    new_label_content = rotate_yolo_label(label_content)

                    label_dst.parent.mkdir(parents=True, exist_ok=True)
                    with open(label_dst, 'w') as f:
                        f.write(new_label_content)
                else:
                    # 没有标签文件，创建空文件（保持一致性）
                    label_dst.parent.mkdir(parents=True, exist_ok=True)
                    label_dst.touch()
            else:
                # 不需要旋转，直接复制
                img_dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img_src, img_dst)

                if label_src.exists():
                    label_dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(label_src, label_dst)
                else:
                    label_dst.parent.mkdir(parents=True, exist_ok=True)
                    label_dst.touch()

    except Exception as e:
        result['error'] = str(e)

    return result


def update_yaml(input_yaml: Path, output_yaml: Path, output_root: Path):
    """
    更新dataset.yaml中的路径信息
    """
    with open(input_yaml, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    # 更新路径
    output_root_str = str(output_root.resolve())

    if 'path' in data:
        data['path'] = output_root_str

    if 'train' in data:
        # 保持相对路径结构
        data['train'] = os.path.join(output_root_str, 'images', 'train')

    if 'val' in data:
        data['val'] = os.path.join(output_root_str, 'images', 'val')

    if 'test' in data:
        data['test'] = os.path.join(output_root_str, 'images', 'test')

    # 保存更新后的yaml
    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(output_yaml, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    return data


def process_split(input_root: Path, output_root: Path, split: str):
    """
    处理一个数据分割(train/val/test)
    """
    img_input_dir = input_root / 'images' / split
    label_input_dir = input_root / 'labels' / split
    img_output_dir = output_root / 'images' / split
    label_output_dir = output_root / 'labels' / split

    if not img_input_dir.exists():
        print(f"  跳过 {split}：目录不存在")
        return {'total': 0, 'rotated': 0, 'errors': 0}

    # 创建输出目录
    img_output_dir.mkdir(parents=True, exist_ok=True)
    label_output_dir.mkdir(parents=True, exist_ok=True)

    # 支持的图片格式
    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

    # 获取所有图片文件
    img_files = [f for f in img_input_dir.iterdir()
                 if f.is_file() and f.suffix.lower() in img_extensions]

    stats = {'total': len(img_files), 'rotated': 0, 'errors': 0}

    print(f"  处理 {split}：共 {len(img_files)} 张图片")

    for i, img_file in enumerate(img_files):
        # 获取对应的标签文件路径
        label_file = label_input_dir / (img_file.stem + '.txt')

        # 输出路径
        img_dst = img_output_dir / img_file.name
        label_dst = label_output_dir / (img_file.stem + '.txt')

        result = process_image_and_label(img_file, label_file, img_dst, label_dst)

        if result['error']:
            stats['errors'] += 1
            print(f"    错误 [{img_file.name}]: {result['error']}")
        elif result['rotated']:
            stats['rotated'] += 1

        # 进度显示
        if (i + 1) % 100 == 0 or (i + 1) == len(img_files):
            print(f"    进度: {i + 1}/{len(img_files)}")

    return stats


def main():
    args = parse_args()

    input_root = Path(args.input).resolve()
    output_root = Path(args.output).resolve()

    print(f"输入数据集: {input_root}")
    print(f"输出数据集: {output_root}")
    print()

    # 验证输入目录
    if not input_root.exists():
        print(f"错误: 输入目录不存在: {input_root}")
        sys.exit(1)

    if not (input_root / 'images').exists():
        print(f"错误: 找不到images目录: {input_root / 'images'}")
        sys.exit(1)

    # 创建输出根目录
    output_root.mkdir(parents=True, exist_ok=True)

    # 处理dataset.yaml
    input_yaml = input_root / 'dataset.yaml'
    output_yaml = output_root / 'dataset.yaml'

    if input_yaml.exists():
        print("处理 dataset.yaml...")
        updated_data = update_yaml(input_yaml, output_yaml, output_root)
        print(f"  已更新路径配置")
        print()
    else:
        print("警告: 未找到 dataset.yaml")
        print()

    # 处理各个数据分割
    total_stats = {'total': 0, 'rotated': 0, 'errors': 0}

    for split in ['train', 'val', 'test']:
        print(f"处理 {split} 分割...")
        stats = process_split(input_root, output_root, split)

        total_stats['total'] += stats['total']
        total_stats['rotated'] += stats['rotated']
        total_stats['errors'] += stats['errors']

        if stats['total'] > 0:
            print(f"    完成: {stats['total']} 张图片, "
                  f"{stats['rotated']} 张旋转, "
                  f"{stats['errors']} 个错误")
        print()

    # 复制其他可能存在的文件（如notes.json等）
    for item in input_root.iterdir():
        if item.is_file() and item.name not in ['dataset.yaml']:
            shutil.copy2(item, output_root / item.name)
            print(f"复制文件: {item.name}")

    print()
    print("=" * 50)
    print("处理完成!")
    print(f"  总图片数: {total_stats['total']}")
    print(f"  旋转数量: {total_stats['rotated']}")
    print(f"  错误数量: {total_stats['errors']}")
    print(f"  输出路径: {output_root}")


if __name__ == '__main__':
    main()