#!/usr/bin/env python3
"""
YOLO数据集旋转处理脚本
功能：遍历数据集，将高度>宽度的图像（竖图）逆时针旋转90°变成横图，同时转换对应的标签
用法：python rotate_yolo_dataset.py --input /path/to/dataset --output /path/to/output
"""

import os
import argparse
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def rotate_yolo_label(label_content: str) -> str:
    """
    逆时针旋转90°后，YOLO标签的坐标变换

    YOLO格式: class_id x_center y_center width height (归一化坐标)

    逆时针旋转90°的变换规则：
    - new_x_center = old_y_center
    - new_y_center = 1 - old_x_center
    - new_width = old_height
    - new_height = old_width
    """
    new_lines = []
    for line in label_content.strip().split('\n'):
        if not line.strip():
            continue
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        class_id = parts[0]
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])

        # 逆时针旋转90°的坐标变换
        new_x_center = y_center
        new_y_center = 1 - x_center
        new_width = height
        new_height = width

        # 保留可能存在的额外信息（如关键点、分割点等）
        extra = ' '.join(parts[5:]) if len(parts) > 5 else ''

        if extra:
            # 如果有额外的点坐标，也需要旋转（假设是成对的x,y坐标）
            extra_parts = extra.split()
            new_extra_parts = []
            for i in range(0, len(extra_parts), 2):
                if i + 1 < len(extra_parts):
                    try:
                        ex = float(extra_parts[i])
                        ey = float(extra_parts[i + 1])
                        new_ex = ey
                        new_ey = 1 - ex
                        new_extra_parts.extend([f"{new_ex:.6f}", f"{new_ey:.6f}"])
                    except ValueError:
                        new_extra_parts.extend([extra_parts[i], extra_parts[i + 1]])
                else:
                    new_extra_parts.append(extra_parts[i])
            extra = ' '.join(new_extra_parts)
            new_line = f"{class_id} {new_x_center:.6f} {new_y_center:.6f} {new_width:.6f} {new_height:.6f} {extra}"
        else:
            new_line = f"{class_id} {new_x_center:.6f} {new_y_center:.6f} {new_width:.6f} {new_height:.6f}"

        new_lines.append(new_line)

    return '\n'.join(new_lines)


def process_image_and_label(img_path: Path, label_path: Path,
                            out_img_path: Path, out_label_path: Path,
                            rotate: bool) -> dict:
    """
    处理单张图像及其标签

    Args:
        img_path: 输入图像路径
        label_path: 输入标签路径
        out_img_path: 输出图像路径
        out_label_path: 输出标签路径
        rotate: 是否需要旋转

    Returns:
        处理信息字典
    """
    result = {
        'image': img_path.name,
        'rotated': rotate,
        'success': True,
        'error': None
    }

    try:
        # 确保输出目录存在
        out_img_path.parent.mkdir(parents=True, exist_ok=True)
        out_label_path.parent.mkdir(parents=True, exist_ok=True)

        if rotate:
            # 旋转图像（逆时针90°）
            with Image.open(img_path) as img:
                # PIL的rotate是逆时针，expand=True保持完整图像
                rotated_img = img.rotate(90, expand=True)
                rotated_img.save(out_img_path)

            # 旋转标签
            if label_path.exists():
                with open(label_path, 'r') as f:
                    label_content = f.read()
                new_label_content = rotate_yolo_label(label_content)
                with open(out_label_path, 'w') as f:
                    f.write(new_label_content)
            else:
                # 标签文件不存在，创建空文件
                out_label_path.touch()
        else:
            # 直接复制
            shutil.copy2(img_path, out_img_path)
            if label_path.exists():
                shutil.copy2(label_path, out_label_path)
            else:
                out_label_path.touch()

    except Exception as e:
        result['success'] = False
        result['error'] = str(e)

    return result


def get_label_path(img_path: Path, labels_base: Path) -> Path:
    """根据图像路径获取对应的标签路径"""
    # 获取相对路径（train/xxx.jpg 或 val/xxx.jpg）
    relative = img_path.parent.name  # train 或 val
    label_name = img_path.stem + '.txt'
    return labels_base / relative / label_name


def process_dataset(input_path: str, output_path: str) -> None:
    """
    处理整个YOLO数据集

    Args:
        input_path: 输入数据集路径
        output_path: 输出数据集路径
    """
    input_dir = Path(input_path)
    output_dir = Path(output_path)

    # 验证输入路径
    if not input_dir.exists():
        raise FileNotFoundError(f"输入路径不存在: {input_dir}")

    images_dir = input_dir / 'images'
    labels_dir = input_dir / 'labels'

    if not images_dir.exists():
        raise FileNotFoundError(f"images目录不存在: {images_dir}")

    # 创建输出目录结构
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (output_dir / 'images' / 'val').mkdir(parents=True, exist_ok=True)
    (output_dir / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
    (output_dir / 'labels' / 'val').mkdir(parents=True, exist_ok=True)

    # 处理dataset.yaml（如果存在）- 修改路径信息
    yaml_file = input_dir / 'dataset.yaml'
    if yaml_file.exists():
        with open(yaml_file, 'r') as f:
            lines = f.readlines()

        new_lines = []
        output_abs = str(output_dir.resolve())
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('path:'):
                new_lines.append(f'path: {output_abs}\n')
            elif stripped.startswith('train:'):
                new_lines.append(f'train: {output_abs}/images/train\n')
            elif stripped.startswith('val:'):
                new_lines.append(f'val: {output_abs}/images/val\n')
            else:
                new_lines.append(line)

        with open(output_dir / 'dataset.yaml', 'w') as f:
            f.writelines(new_lines)
        print(f"✓ 已更新 dataset.yaml（路径已修改为输出目录）")

    # 支持的图像格式
    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    # 收集所有图像文件
    all_images = []
    for split in ['train', 'val']:
        split_dir = images_dir / split
        if split_dir.exists():
            for img_path in split_dir.iterdir():
                if img_path.suffix.lower() in img_extensions:
                    all_images.append(img_path)

    if not all_images:
        print("警告: 未找到任何图像文件!")
        return

    print(f"\n找到 {len(all_images)} 张图像，开始处理...\n")

    # 统计信息
    stats = {
        'total': len(all_images),
        'rotated': 0,
        'copied': 0,
        'failed': 0
    }

    # 处理每张图像
    for img_path in tqdm(all_images, desc="处理进度"):
        try:
            # 获取图像尺寸
            with Image.open(img_path) as img:
                width, height = img.size

            # 判断是否需要旋转（高度 > 宽度，即竖图转横图）
            need_rotate = height > width

            # 计算输出路径
            split = img_path.parent.name  # train 或 val
            out_img_path = output_dir / 'images' / split / img_path.name

            # 获取标签路径
            label_path = get_label_path(img_path, labels_dir)
            out_label_path = output_dir / 'labels' / split / (img_path.stem + '.txt')

            # 处理图像和标签
            result = process_image_and_label(
                img_path, label_path,
                out_img_path, out_label_path,
                rotate=need_rotate
            )

            if result['success']:
                if need_rotate:
                    stats['rotated'] += 1
                else:
                    stats['copied'] += 1
            else:
                stats['failed'] += 1
                print(f"\n✗ 处理失败: {img_path.name} - {result['error']}")

        except Exception as e:
            stats['failed'] += 1
            print(f"\n✗ 处理出错: {img_path.name} - {str(e)}")

    # 打印统计信息
    print("\n" + "=" * 50)
    print("处理完成！统计信息：")
    print("=" * 50)
    print(f"  总图像数:     {stats['total']}")
    print(f"  已旋转:       {stats['rotated']} (高>宽，竖图)")
    print(f"  直接复制:     {stats['copied']} (宽≥高，横图)")
    print(f"  处理失败:     {stats['failed']}")
    print(f"\n输出路径: {output_dir}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description='YOLO数据集旋转处理脚本 - 将高>宽的竖图逆时针旋转90°变成横图',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例用法:
  python rotate_yolo_dataset.py --input ./my_dataset --output ./rotated_dataset
  python rotate_yolo_dataset.py -i /data/yolo_data -o /data/yolo_rotated

输入数据集结构:
  ├── dataset.yaml
  ├── images
  │   ├── train
  │   └── val
  └── labels
      ├── train
      └── val
        '''
    )

    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='输入YOLO数据集路径'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='输出数据集路径'
    )

    args = parser.parse_args()

    print("\n" + "=" * 50)
    print("YOLO数据集旋转处理脚本")
    print("=" * 50)
    print(f"输入路径: {args.input}")
    print(f"输出路径: {args.output}")
    print("旋转条件: 图像高度 > 宽度（竖图）→ 逆时针旋转90°变横图")
    print("=" * 50)

    try:
        process_dataset(args.input, args.output)
    except Exception as e:
        print(f"\n错误: {str(e)}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())