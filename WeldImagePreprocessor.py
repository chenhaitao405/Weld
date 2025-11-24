"""
脚本名称: weld_preprocessor.py
功能概述: 焊缝X射线图像预处理工具，实现滑动窗口裁剪、图像增强和标签格式转换
详细说明:
    - 输入格式: 原始图像（tif格式）+ LabelMe JSON标注
    - 处理流程: 滑动窗口裁剪 → 图像增强 → 标签调整 → 可选的YOLO格式转换
    - 输出格式: 处理后的图像patches + 标注文件（LabelMe或YOLO格式）
依赖模块: utils.image_processing, utils.label_processing, utils.dataset_management
使用示例:
    # 保持LabelMe格式
    python weld_preprocessor.py --input_dir ./data --output_dir ./output --output_format labelme
+5
    # 转换为YOLO检测格式
    python weld_preprocessor.py --input_dir ./data --output_dir ./output --output_format yolo --label_mode det

    # 转换为YOLO分割格式
    python weld_preprocessor.py --input_dir ./data --output_dir ./output --output_format yolo --label_mode seg
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
from utils.constants import DEFAULT_OVERLAP_RATIO, DEFAULT_WINDOW_SIZE, DEFAULT_JPEG_QUALITY


class WeldImagePreprocessor:
    """焊缝X射线图像预处理器，支持LabelMe和YOLO两种输出格式"""

    def __init__(self,
                 overlap_ratio: float = DEFAULT_OVERLAP_RATIO,
                 enhance_mode: str = 'original',
                 output_format: str = 'labelme',
                 label_mode: str = 'det',
                 filter_label: str = None,
                 unify_to_crack: bool = False):
        """
        初始化预处理器

        Args:
            overlap_ratio: 滑动窗口重叠率
            enhance_mode: 增强模式 ('original' 或 'windowing')
            output_format: 输出格式 ('labelme' 或 'yolo')
            label_mode: 标签模式 ('det' 或 'seg')，仅在output_format='yolo'时有效
            filter_label: 要过滤的标签名称
            unify_to_crack: 是否将所有标签统一为"crack"
        """
        self.overlap_ratio = overlap_ratio
        self.enhance_mode = enhance_mode
        self.output_format = output_format
        self.label_mode = label_mode
        self.filter_label = filter_label
        self.unify_to_crack = unify_to_crack

        # 统计信息
        self.total_patches = 0
        self.patches_with_defects = 0
        self.patches_without_defects = 0

        # 如果输出YOLO格式，需要收集标签映射
        if output_format == 'yolo':
            self.label_id_map = OrderedDict()
            self.temp_labelme_files = []  # 临时存储LabelMe文件路径

        print(f"WeldImagePreprocessor initialized with:")
        print(f"  - Overlap ratio: {self.overlap_ratio}")
        print(f"  - Enhancement mode: {self.enhance_mode}")
        print(f"  - Output format: {self.output_format}")
        if self.output_format == 'yolo':
            print(f"  - Label mode: {self.label_mode}")
            if self.filter_label:
                print(f"  - Filter label: {self.filter_label}")
            if self.unify_to_crack:
                print(f"  - Unify to crack: True")

    def _adjust_labelme_annotations(self, annotations: Dict, crop_x: int, crop_y: int,
                                   crop_w: int, crop_h: int) -> Dict:
        """调整LabelMe标注以适应裁剪后的图像"""
        new_annotations = {
            'version': annotations.get('version', '4.5.7'),
            'flags': annotations.get('flags', {}),
            'shapes': [],
            'imagePath': annotations.get('imagePath', ''),
            'imageData': None,
            'imageHeight': crop_h,
            'imageWidth': crop_w
        }

        for shape in annotations.get('shapes', []):
            # 获取标注的边界
            points = np.array(shape['points'])
            min_x, min_y = points.min(axis=0)
            max_x, max_y = points.max(axis=0)

            # 判断是否与裁剪区域有交集
            if (max_x >= crop_x and min_x < crop_x + crop_w and
                max_y >= crop_y and min_y < crop_y + crop_h):

                # 调整坐标
                new_points = []
                for point in points:
                    new_x = max(0, min(point[0] - crop_x, crop_w - 1))
                    new_y = max(0, min(point[1] - crop_y, crop_h - 1))
                    new_points.append([float(new_x), float(new_y)])

                new_shape = {
                    'label': shape['label'],
                    'points': new_points,
                    'group_id': shape.get('group_id'),
                    'shape_type': shape['shape_type'],
                    'flags': shape.get('flags', {})
                }
                new_annotations['shapes'].append(new_shape)

        return new_annotations

    def _convert_labelme_to_yolo(self, annotations: Dict, img_w: int, img_h: int) -> List[List[float]]:
        """将LabelMe标注转换为YOLO格式"""
        yolo_labels = []

        for shape in annotations.get('shapes', []):
            label = shape['label']

            # 过滤指定标签
            if self.filter_label and label == self.filter_label:
                continue

            # 统一标签
            if self.unify_to_crack:
                label = 'crack'

            # 获取或分配标签ID
            if label not in self.label_id_map:
                self.label_id_map[label] = len(self.label_id_map)
            label_id = self.label_id_map[label]

            # 根据形状类型转换
            if shape['shape_type'] == 'rectangle' and self.label_mode == 'det':
                # 矩形转检测框
                points = shape['points']
                if len(points) == 2:
                    x1, y1 = points[0]
                    x2, y2 = points[1]

                    x_center = (x1 + x2) / 2.0 / img_w
                    y_center = (y1 + y2) / 2.0 / img_h
                    width = abs(x2 - x1) / img_w
                    height = abs(y2 - y1) / img_h

                    yolo_labels.append([label_id, x_center, y_center, width, height])

            elif shape['shape_type'] in ['polygon', 'rectangle'] and self.label_mode == 'seg':
                # 多边形或矩形转分割
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
                # 其他形状转边界框
                if self.label_mode == 'det':
                    points = np.array(shape['points'])
                    x_min, y_min = points.min(axis=0)
                    x_max, y_max = points.max(axis=0)

                    x_center = (x_min + x_max) / 2.0 / img_w
                    y_center = (y_min + y_max) / 2.0 / img_h
                    width = (x_max - x_min) / img_w
                    height = (y_max - y_min) / img_h

                    yolo_labels.append([label_id, x_center, y_center, width, height])

        return yolo_labels

    def process_single_image(self, image_path: str, json_path: str,
                           output_image_dir: str, output_label_dir: str,
                           return_patches: bool = False) -> Optional[List[Dict]]:
        """处理单张图像"""
        # 读取图像
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"无法读取图像: {image_path}")
            return None

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

        # 用于返回的patches信息
        processed_patches = [] if return_patches else None

        # 处理每个patch
        base_name = Path(image_path).stem
        for i, patch_info in enumerate(patches):
            # 图像增强
            enhanced_patch = enhance_image(patch_info['patch'], self.enhance_mode)

            # 调整标注
            x, y = patch_info['position']
            patch_w, patch_h = patch_info['size']
            adjusted_annotations = self._adjust_labelme_annotations(
                annotations, x, y, patch_w, patch_h
            )

            # 生成文件名
            patch_name = f"{base_name}_patch_{i:04d}"

            # 保存图像
            image_save_path = Path(output_image_dir) / f"{patch_name}.jpg"
            cv2.imwrite(str(image_save_path), enhanced_patch,
                       [cv2.IMWRITE_JPEG_QUALITY, DEFAULT_JPEG_QUALITY])

            # 保存标注并统计
            if self.output_format == 'labelme':
                # 保存LabelMe格式
                label_save_path = Path(output_label_dir) / f"{patch_name}.json"
                save_labelme_json(adjusted_annotations, str(label_save_path))

                # 如果需要转换为YOLO，记录文件路径
                if hasattr(self, 'temp_labelme_files'):
                    self.temp_labelme_files.append((str(image_save_path), str(label_save_path)))

                # LabelMe格式：基于过滤后的shapes统计
                has_defects = False
                for shape in adjusted_annotations.get('shapes', []):
                    label = shape.get('label', '')
                    # 跳过被过滤的标签
                    if self.filter_label and label == self.filter_label:
                        continue
                    has_defects = True
                    break

            else:  # yolo format
                # 直接转换为YOLO格式
                yolo_labels = self._convert_labelme_to_yolo(
                    adjusted_annotations, patch_w, patch_h
                )
                label_save_path = Path(output_label_dir) / f"{patch_name}.txt"
                save_yolo_labels(yolo_labels, str(label_save_path), self.label_mode)

                # YOLO格式：基于转换后的标签统计
                has_defects = len(yolo_labels) > 0

            # 更新统计信息
            self.total_patches += 1
            if has_defects:
                self.patches_with_defects += 1
            else:
                self.patches_without_defects += 1

            # 如果需要返回patches信息
            if return_patches:
                processed_patches.append({
                    'original_patch': patch_info['patch'],
                    'enhanced_patch': enhanced_patch,
                    'position': patch_info['position'],
                    'size': patch_info['size'],
                    'annotations': adjusted_annotations,
                    'patch_name': patch_name
                })

        return processed_patches

    def process_dataset(self, input_base_dir: str, output_base_dir: str,
                       balance: bool = True, val_size: float = 0.2):
        """处理整个数据集"""
        input_base = Path(input_base_dir)
        output_base = Path(output_base_dir)

        # 创建输出目录
        if self.output_format == 'yolo':
            # YOLO格式的目录结构
            create_directory_structure(output_base)
            output_image_base = output_base / 'images'
            output_label_base = output_base / 'labels'
        else:
            # LabelMe格式的目录结构
            output_image_base = output_base / 'images'
            output_label_base = output_base / 'labels'
            output_image_base.mkdir(parents=True, exist_ok=True)
            output_label_base.mkdir(parents=True, exist_ok=True)

        # 收集所有需要处理的文件
        all_file_pairs = []

        # 遍历所有焊缝类型和子类型
        for weld_type in ['L', 'T']:
            for sub_type in ['1', '2']:
                image_dir = input_base / 'crop_weld_images' / weld_type / sub_type
                json_dir = input_base / 'crop_weld_jsons' / weld_type / sub_type

                if not image_dir.exists():
                    continue

                # 获取所有图像文件
                image_files = sorted(list(image_dir.glob('*.tif')))

                for image_file in image_files:
                    json_file = json_dir / f"{image_file.stem}.json"

                    if json_file.exists():
                        all_file_pairs.append((str(image_file), str(json_file), weld_type, sub_type))
                    else:
                        print(f"警告: 找不到标注文件 {json_file}")

        print(f"找到 {len(all_file_pairs)} 对图像-标注文件")

        # 如果是YOLO格式，需要划分训练集和验证集
        if self.output_format == 'yolo':
            train_pairs, val_pairs = train_val_split(all_file_pairs, val_size)

            # 处理训练集
            print("处理训练集...")
            for img_path, json_path, weld_type, sub_type in tqdm(train_pairs):
                self.process_single_image(
                    img_path, json_path,
                    str(output_image_base / 'train'),
                    str(output_label_base / 'train')
                )

            # 处理验证集
            print("处理验证集...")
            for img_path, json_path, weld_type, sub_type in tqdm(val_pairs):
                self.process_single_image(
                    img_path, json_path,
                    str(output_image_base / 'val'),
                    str(output_label_base / 'val')
                )
        else:
            # LabelMe格式，保持原始目录结构
            for img_path, json_path, weld_type, sub_type in tqdm(all_file_pairs):
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
                # 分别平衡训练集和验证集
                balance_stats_train = balance_dataset(
                    str(output_label_base / 'train'),
                    str(output_image_base / 'train')
                )
                balance_stats_val = balance_dataset(
                    str(output_label_base / 'val'),
                    str(output_image_base / 'val')
                )
                print(f"训练集: 保留 {balance_stats_train['kept_unlabeled']} 个无缺陷样本")
                print(f"验证集: 保留 {balance_stats_val['kept_unlabeled']} 个无缺陷样本")
            else:
                balance_stats = balance_dataset(
                    str(output_label_base),
                    str(output_image_base)
                )
                print(f"保留 {balance_stats['kept_unlabeled']} 个无缺陷样本")

        # 如果是YOLO格式，创建dataset.yaml
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
        description='焊缝X射线图像预处理工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 保持LabelMe格式
  python weld_preprocessor.py --input_dir ./data --output_dir ./output --output_format labelme
  
  # 转换为YOLO检测格式
  python weld_preprocessor.py --input_dir ./data --output_dir ./output --output_format yolo --label_mode det
  
  # 转换为YOLO分割格式
  python weld_preprocessor.py --input_dir ./data --output_dir ./output --output_format yolo --label_mode seg
  
  # 使用窗宽窗位增强模式
  python weld_preprocessor.py --input_dir ./data --output_dir ./output --enhance_mode windowing
        """
    )

    parser.add_argument('--input_dir', type=str, required=True,
                       help='输入目录路径，包含crop_weld_images和crop_weld_jsons')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='输出目录路径')
    parser.add_argument('--output_format', type=str, choices=['labelme', 'yolo'],
                       default='yolo',
                       help='输出格式: labelme(JSON) 或 yolo(TXT)')
    parser.add_argument('--label_mode', type=str, choices=['det', 'seg'],
                       default='det',
                       help='标签模式(仅YOLO): det(检测) 或 seg(分割)')
    parser.add_argument('--enhance_mode', type=str, choices=['original', 'windowing'],
                       default='original',
                       help='图像增强模式: original 或 windowing')
    parser.add_argument('--overlap', type=float, default=0.5,
                       help='滑动窗口重叠率 (0.0-1.0)')
    parser.add_argument('--balance', action='store_true', default=True,
                       help='平衡数据集')
    parser.add_argument('--no-balance', dest='balance', action='store_false',
                       help='不平衡数据集')
    parser.add_argument('--val_size', type=float, default=0.2,
                       help='验证集比例(仅YOLO)')
    parser.add_argument('--filter_label', type=str, default=None,
                       help='要过滤的标签名称')
    parser.add_argument('--unify_to_crack', action='store_true',
                       help='将所有标签统一为"crack"')

    args = parser.parse_args()

    # 创建预处理器
    preprocessor = WeldImagePreprocessor(
        overlap_ratio=args.overlap,
        enhance_mode=args.enhance_mode,
        output_format=args.output_format,
        label_mode=args.label_mode,
        filter_label=args.filter_label,
        unify_to_crack=args.unify_to_crack
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