"""
脚本名称: weld_preprocessor.py - 性能优化版
功能概述: 焊缝X射线图像预处理工具，实现滑动窗口裁剪、图像增强和标签格式转换
优化特性:
    - 多进程并行处理
    - 批量处理减少I/O
    - 优化的16位直方图均衡
    - 减少不必要的打印输出
详细说明:
    - 输入格式: 原始图像（tif格式，支持8位和16位）+ LabelMe JSON标注
    - 处理流程: 滑动窗口裁剪 → 图像增强 → 标签调整 → 可选的YOLO格式转换
    - 输出格式: 处理后的图像patches（支持8位和16位）+ 标注文件（LabelMe或YOLO格式）
    - 16位图像处理: 全程保持16位精度，输出为tif格式
"""

import os
import sys
import json
import cv2
import numpy as np
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from collections import OrderedDict
from multiprocessing import Pool, cpu_count
from functools import partial
import warnings

# 抑制不必要的警告
warnings.filterwarnings('ignore', category=UserWarning)

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import (
    # 图像处理
    enhance_image, sliding_window_crop, calculate_stride,
    # 标签处理
    read_labelme_json, save_labelme_json, save_yolo_labels,
    # 数据集管理
    create_directory_structure, balance_dataset,
    train_val_split, create_dataset_yaml
)
from utils.constants import DEFAULT_OVERLAP_RATIO, DEFAULT_WINDOW_SIZE

# 默认PNG压缩级别（0-9，0为无压缩，9为最大压缩）
# 降低压缩级别以加快速度
DEFAULT_PNG_COMPRESSION = 1


def process_single_patch(args):
    """
    处理单个patch的函数（用于多进程）

    Args:
        args: 包含所有必要参数的元组

    Returns:
        处理结果字典
    """
    (patch_data, patch_index, base_name, annotations,
     enhance_mode, output_bits, output_format, label_mode,
     filter_label, unify_to_crack, output_image_dir, output_label_dir,
     label_id_map) = args  # 添加label_id_map参数

    patch_info = patch_data

    # 图像增强
    enhanced_patch = enhance_image(patch_info['patch'], enhance_mode, output_bits)

    # 调整标注
    x, y = patch_info['position']
    patch_w, patch_h = patch_info['size']

    # 调整标注坐标
    adjusted_annotations = _adjust_annotations_simple(
        annotations, x, y, patch_w, patch_h, filter_label
    )

    # 生成文件名
    patch_name = f"{base_name}_patch_{patch_index:04d}"

    # 保存图像 - 优化版本
    image_save_path = Path(output_image_dir) / f"{patch_name}.tif"
    cv2.imwrite(str(image_save_path), enhanced_patch)

    # 保存标注
    has_defects = False
    detected_labels = []  # 收集检测到的标签

    if output_format == 'labelme':
        label_save_path = Path(output_label_dir) / f"{patch_name}.json"
        save_labelme_json(adjusted_annotations, str(label_save_path))
        has_defects = len(adjusted_annotations.get('shapes', [])) > 0
    else:  # yolo format
        yolo_labels = _convert_to_yolo_with_map(
            adjusted_annotations, patch_w, patch_h,
            filter_label, unify_to_crack, label_mode, label_id_map
        )
        label_save_path = Path(output_label_dir) / f"{patch_name}.txt"
        save_yolo_labels(yolo_labels, str(label_save_path), label_mode)
        has_defects = len(yolo_labels) > 0

        # 收集检测到的标签名
        for shape in adjusted_annotations.get('shapes', []):
            label = shape['label']
            if filter_label and label == filter_label:
                continue
            if unify_to_crack:
                label = 'crack'
            if label not in detected_labels:
                detected_labels.append(label)

    return {
        'has_defects': has_defects,
        'patch_name': patch_name,
        'image_path': str(image_save_path),
        'label_path': str(label_save_path),
        'detected_labels': detected_labels  # 返回检测到的标签
    }


def _adjust_annotations_simple(annotations, crop_x, crop_y, crop_w, crop_h, filter_label):
    """简化的标注调整函数"""
    new_annotations = {
        'version': annotations.get('version', '4.5.7'),
        'flags': {},
        'shapes': [],
        'imagePath': '',
        'imageData': None,
        'imageHeight': crop_h,
        'imageWidth': crop_w
    }

    for shape in annotations.get('shapes', []):
        # 过滤标签
        if filter_label and shape.get('label') == filter_label:
            continue

        points = np.array(shape['points'])
        min_x, min_y = points.min(axis=0)
        max_x, max_y = points.max(axis=0)

        # 判断是否与裁剪区域有交集
        if (max_x >= crop_x and min_x < crop_x + crop_w and
            max_y >= crop_y and min_y < crop_y + crop_h):

            # 调整坐标
            new_points = np.clip(points - [crop_x, crop_y],
                               [0, 0], [crop_w-1, crop_h-1])

            new_shape = {
                'label': shape['label'],
                'points': new_points.tolist(),
                'group_id': shape.get('group_id'),
                'shape_type': shape['shape_type'],
                'flags': {}
            }
            new_annotations['shapes'].append(new_shape)

    return new_annotations


def _convert_to_yolo_with_map(annotations, img_w, img_h,
                              filter_label, unify_to_crack, label_mode, label_id_map):
    """
    使用预定义的label_id_map转换为YOLO格式

    Args:
        annotations: LabelMe标注
        img_w, img_h: 图像尺寸
        filter_label: 要过滤的标签
        unify_to_crack: 是否统一为crack
        label_mode: 检测或分割模式
        label_id_map: 预定义的标签ID映射

    Returns:
        YOLO格式的标签列表
    """
    yolo_labels = []

    for shape in annotations.get('shapes', []):
        label = shape['label']

        # 过滤指定标签
        if filter_label and label == filter_label:
            continue

        # 统一标签
        if unify_to_crack:
            label = 'crack'

        # 使用传入的label_id_map获取ID
        if label in label_id_map:
            label_id = label_id_map[label]
        else:
            # 如果标签不在映射中，跳过（理论上不应该发生）
            continue

        # 根据形状类型和模式转换
        if shape['shape_type'] == 'rectangle' and label_mode == 'det':
            points = shape['points']
            if len(points) == 2:
                x1, y1 = points[0]
                x2, y2 = points[1]
                x_center = (x1 + x2) / 2.0 / img_w
                y_center = (y1 + y2) / 2.0 / img_h
                width = abs(x2 - x1) / img_w
                height = abs(y2 - y1) / img_h
                yolo_labels.append([label_id, x_center, y_center, width, height])

        elif shape['shape_type'] in ['polygon', 'rectangle'] and label_mode == 'seg':
            points = shape['points']

            # 如果是矩形，转换为多边形
            if shape['shape_type'] == 'rectangle' and len(points) == 2:
                x1, y1 = points[0]
                x2, y2 = points[1]
                points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

            if len(points) >= 3:
                yolo_label = [label_id]
                for point in points:
                    yolo_label.extend([point[0] / img_w, point[1] / img_h])
                yolo_labels.append(yolo_label)

        else:
            # 其他形状转边界框（仅检测模式）
            if label_mode == 'det':
                points = np.array(shape['points'])
                if len(points) > 0:
                    x_min, y_min = points.min(axis=0)
                    x_max, y_max = points.max(axis=0)

                    x_center = (x_min + x_max) / 2.0 / img_w
                    y_center = (y_min + y_max) / 2.0 / img_h
                    width = (x_max - x_min) / img_w
                    height = (y_max - y_min) / img_h

                    yolo_labels.append([label_id, x_center, y_center, width, height])

    return yolo_labels


def _convert_to_yolo_simple(annotations, img_w, img_h,
                           filter_label, unify_to_crack, label_mode):
    """简化的YOLO转换函数（已废弃，保留用于兼容）"""
    yolo_labels = []
    label_map = {}

    for shape in annotations.get('shapes', []):
        label = shape['label']

        if filter_label and label == filter_label:
            continue

        if unify_to_crack:
            label = 'crack'

        if label not in label_map:
            label_map[label] = len(label_map)
        label_id = label_map[label]

        if shape['shape_type'] == 'rectangle' and label_mode == 'det':
            points = shape['points']
            if len(points) == 2:
                x1, y1 = points[0]
                x2, y2 = points[1]
                x_center = (x1 + x2) / 2.0 / img_w
                y_center = (y1 + y2) / 2.0 / img_h
                width = abs(x2 - x1) / img_w
                height = abs(y2 - y1) / img_h
                yolo_labels.append([label_id, x_center, y_center, width, height])

    return yolo_labels


class WeldImagePreprocessor:
    """焊缝X射线图像预处理器 - 性能优化版"""

    def __init__(self,
                 overlap_ratio: float = DEFAULT_OVERLAP_RATIO,
                 enhance_mode: str = 'original',
                 output_format: str = 'labelme',
                 label_mode: str = 'det',
                 filter_label: str = None,
                 unify_to_crack: bool = False,
                 output_bits: int = 8,
                 num_workers: int = None):
        """
        初始化预处理器

        Args:
            overlap_ratio: 滑动窗口重叠率
            enhance_mode: 增强模式 ('original' 或 'windowing')
            output_format: 输出格式 ('labelme' 或 'yolo')
            label_mode: 标签模式 ('det' 或 'seg')
            filter_label: 要过滤的标签名称
            unify_to_crack: 是否将所有标签统一为"crack"
            output_bits: 输出位深度（8或16）
            num_workers: 并行处理的进程数（None则自动检测）
        """
        self.overlap_ratio = overlap_ratio
        self.enhance_mode = enhance_mode
        self.output_format = output_format
        self.label_mode = label_mode
        self.filter_label = filter_label
        self.unify_to_crack = unify_to_crack
        self.output_bits = output_bits
        self.num_workers = num_workers or max(1, cpu_count() - 1)

        # 统计信息
        self.total_patches = 0
        self.patches_with_defects = 0
        self.patches_without_defects = 0

        # 如果输出YOLO格式，需要收集标签映射
        if output_format == 'yolo':
            self.label_id_map = OrderedDict()

        print(f"WeldImagePreprocessor initialized with:")
        print(f"  - Overlap ratio: {self.overlap_ratio}")
        print(f"  - Enhancement mode: {self.enhance_mode}")
        print(f"  - Output format: {self.output_format}")
        print(f"  - Output bits: {self.output_bits}")
        print(f"  - Parallel workers: {self.num_workers}")

    def _adjust_labelme_annotations(self, annotations: Dict, crop_x: int, crop_y: int,
                                   crop_w: int, crop_h: int) -> Dict:
        """调整LabelMe标注以适应裁剪后的图像"""
        return _adjust_annotations_simple(annotations, crop_x, crop_y,
                                         crop_w, crop_h, self.filter_label)

    def _build_label_map(self, all_file_pairs):
        """
        预先扫描所有标注文件，构建完整的标签映射

        Args:
            all_file_pairs: 所有的(图像路径, json路径, ...)元组列表

        Returns:
            标签ID映射字典
        """
        label_set = set()

        print("扫描标签类别...")
        for img_path, json_path, *_ in tqdm(all_file_pairs, desc="扫描标签"):
            try:
                annotations = read_labelme_json(json_path)
                for shape in annotations.get('shapes', []):
                    label = shape['label']

                    # 应用过滤
                    if self.filter_label and label == self.filter_label:
                        continue

                    # 应用统一标签
                    if self.unify_to_crack:
                        label = 'crack'

                    label_set.add(label)
            except:
                continue

        # 创建有序的标签映射
        label_id_map = OrderedDict()
        for i, label in enumerate(sorted(label_set)):
            label_id_map[label] = i

        print(f"发现 {len(label_id_map)} 个标签类别: {list(label_id_map.keys())}")
        return label_id_map

    def process_single_image_parallel(self, image_path: str, json_path: str,
                                     output_image_dir: str, output_label_dir: str) -> None:
        """使用多进程并行处理单张图像"""
        # 读取图像
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"无法读取图像: {image_path}")
            return

        # 读取标注
        annotations = read_labelme_json(json_path)

        # 确定滑动窗口大小
        h, w = image.shape[:2]
        window_size = min(DEFAULT_WINDOW_SIZE, min(h, w))
        window_size = (window_size, window_size)

        # 计算步长
        stride = calculate_stride(window_size, self.overlap_ratio)

        # 滑动窗口裁剪
        patches = sliding_window_crop(image, window_size, stride)

        # 准备多进程参数
        base_name = Path(image_path).stem
        process_args = []

        for i, patch_info in enumerate(patches):
            args = (
                patch_info, i, base_name, annotations,
                self.enhance_mode, self.output_bits, self.output_format,
                self.label_mode, self.filter_label, self.unify_to_crack,
                output_image_dir, output_label_dir,
                self.label_id_map  # 传递label_id_map
            )
            process_args.append(args)

        # 使用进程池并行处理
        with Pool(self.num_workers) as pool:
            results = list(pool.map(process_single_patch, process_args))

        # 更新统计信息
        for result in results:
            self.total_patches += 1
            if result['has_defects']:
                self.patches_with_defects += 1
            else:
                self.patches_without_defects += 1

    def process_single_image(self, image_path: str, json_path: str,
                           output_image_dir: str, output_label_dir: str,
                           return_patches: bool = False) -> Optional[List[Dict]]:
        """处理单张图像（兼容接口）"""
        if return_patches:
            # 如果需要返回patches，使用原始单进程版本
            return self._process_single_image_original(
                image_path, json_path, output_image_dir, output_label_dir, True
            )
        else:
            # 否则使用多进程版本
            self.process_single_image_parallel(
                image_path, json_path, output_image_dir, output_label_dir
            )
            return None


    def process_dataset(self, input_base_dir: str, output_base_dir: str,
                       balance: bool = True, val_size: float = 0.2):
        """处理整个数据集"""
        input_base = Path(input_base_dir)
        output_base = Path(output_base_dir)

        # 创建输出目录
        if self.output_format == 'yolo':
            create_directory_structure(output_base)
            output_image_base = output_base / 'images'
            output_label_base = output_base / 'labels'
        else:
            output_image_base = output_base / 'images'
            output_label_base = output_base / 'labels'
            output_image_base.mkdir(parents=True, exist_ok=True)
            output_label_base.mkdir(parents=True, exist_ok=True)

        # 收集所有需要处理的文件
        all_file_pairs = []

        for weld_type in ['L', 'T']:
            for sub_type in ['1', '2']:
                image_dir = input_base / 'crop_weld_images' / weld_type / sub_type
                json_dir = input_base / 'crop_weld_jsons' / weld_type / sub_type

                if not image_dir.exists():
                    continue

                image_files = sorted(list(image_dir.glob('*.tif')))

                for image_file in image_files:
                    json_file = json_dir / f"{image_file.stem}.json"
                    if json_file.exists():
                        all_file_pairs.append((str(image_file), str(json_file), weld_type, sub_type))

        print(f"找到 {len(all_file_pairs)} 对图像-标注文件")

        # 如果是YOLO格式，预先构建标签映射
        if self.output_format == 'yolo':
            self.label_id_map = self._build_label_map(all_file_pairs)

        print(f"输出位深度: {self.output_bits} bits")
        print(f"使用 {self.num_workers} 个进程并行处理")

        # 处理数据集
        if self.output_format == 'yolo':
            train_pairs, val_pairs = train_val_split(all_file_pairs, val_size)

            # 处理训练集
            print("\n处理训练集...")
            for img_path, json_path, weld_type, sub_type in tqdm(train_pairs, desc="训练集"):
                self.process_single_image(
                    img_path, json_path,
                    str(output_image_base / 'train'),
                    str(output_label_base / 'train')
                )

            # 处理验证集
            print("\n处理验证集...")
            for img_path, json_path, weld_type, sub_type in tqdm(val_pairs, desc="验证集"):
                self.process_single_image(
                    img_path, json_path,
                    str(output_image_base / 'val'),
                    str(output_label_base / 'val')
                )
        else:
            # LabelMe格式
            for img_path, json_path, weld_type, sub_type in tqdm(all_file_pairs, desc="处理图像"):
                output_img_dir = output_image_base / weld_type / sub_type
                output_lbl_dir = output_label_base / weld_type / sub_type
                output_img_dir.mkdir(parents=True, exist_ok=True)
                output_lbl_dir.mkdir(parents=True, exist_ok=True)

                self.process_single_image(
                    img_path, json_path,
                    str(output_img_dir),
                    str(output_lbl_dir)
                )

        # 打印统计信息
        stats = self.get_statistics()
        print("\n处理完成！")
        print(f"总patch数: {stats['total_patches']}")
        print(f"有缺陷的patch数: {stats['patches_with_defects']}")
        print(f"无缺陷的patch数: {stats['patches_without_defects']}")

        # 平衡数据集
        if balance:
            print("\n平衡数据集...")
            if self.output_format == 'yolo':
                balance_stats_train = balance_dataset(
                    str(output_label_base / 'train'),
                    str(output_image_base / 'train'),
                    image_ext='.tif'
                )
                balance_stats_val = balance_dataset(
                    str(output_label_base / 'val'),
                    str(output_image_base / 'val'),
                    image_ext='.tif'
                )
                print(f"训练集: 保留 {balance_stats_train['kept_unlabeled']} 个无缺陷样本")
                print(f"验证集: 保留 {balance_stats_val['kept_unlabeled']} 个无缺陷样本")
            else:
                balance_stats = balance_dataset(
                    str(output_label_base),
                    str(output_image_base),
                    image_ext='.tif'
                )
                print(f"保留 {balance_stats['kept_unlabeled']} 个无缺陷样本")

        # YOLO格式创建dataset.yaml
        if self.output_format == 'yolo':
            print("\n创建dataset.yaml...")
            class_names = list(self.label_id_map.keys()) if self.label_id_map else ['defect']
            create_dataset_yaml(
                str(output_base / 'dataset.yaml'),
                str(output_image_base / 'train'),
                str(output_image_base / 'val'),
                len(class_names),
                class_names
            )
            print(f"类别: {class_names}")

    def get_statistics(self) -> Dict:
        """获取处理统计信息"""
        return {
            'total_patches': self.total_patches,
            'patches_with_defects': self.patches_with_defects,
            'patches_without_defects': self.patches_without_defects
        }


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description='焊缝X射线图像预处理工具（性能优化版）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 使用默认并行处理
  python weld_preprocessor.py --input_dir ./data --output_dir ./output
  
  # 指定进程数
  python weld_preprocessor.py --input_dir ./data --output_dir ./output --num_workers 8
  
  # 16位输出，窗宽窗位增强
  python weld_preprocessor.py --input_dir ./data --output_dir ./output --output_bits 16 --enhance_mode windowing
        """
    )

    parser.add_argument('--input_dir', type=str, required=True,
                       help='输入目录路径')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='输出目录路径')
    parser.add_argument('--output_format', type=str, choices=['labelme', 'yolo'],
                       default='yolo',
                       help='输出格式')
    parser.add_argument('--label_mode', type=str, choices=['det', 'seg'],
                       default='det',
                       help='标签模式')
    parser.add_argument('--enhance_mode', type=str, choices=['original', 'windowing'],
                       default='original',
                       help='图像增强模式')
    parser.add_argument('--output_bits', type=int, choices=[8, 16],
                       default=8,
                       help='输出位深度')
    parser.add_argument('--overlap', type=float, default=0.5,
                       help='滑动窗口重叠率')
    parser.add_argument('--balance', action='store_true', default=True,
                       help='平衡数据集')
    parser.add_argument('--no-balance', dest='balance', action='store_false',
                       help='不平衡数据集')
    parser.add_argument('--val_size', type=float, default=0.2,
                       help='验证集比例')
    parser.add_argument('--filter_label', type=str, default=None,
                       help='要过滤的标签')
    parser.add_argument('--unify_to_crack', action='store_true',
                       help='统一标签为crack')
    parser.add_argument('--num_workers', type=int, default=None,
                       help='并行进程数（默认自动检测）')

    args = parser.parse_args()

    # 创建预处理器
    preprocessor = WeldImagePreprocessor(
        overlap_ratio=args.overlap,
        enhance_mode=args.enhance_mode,
        output_format=args.output_format,
        label_mode=args.label_mode,
        filter_label=args.filter_label,
        unify_to_crack=args.unify_to_crack,
        output_bits=args.output_bits,
        num_workers=args.num_workers
    )

    # 处理数据集
    preprocessor.process_dataset(
        args.input_dir,
        args.output_dir,
        balance=args.balance,
        val_size=args.val_size
    )


if __name__ == "__main__":
    main()