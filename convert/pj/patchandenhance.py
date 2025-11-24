import os
import sys
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import random
import shutil

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

current_script_path = os.path.abspath(__file__)
pj_dir = os.path.dirname(current_script_path)
convert_dir = os.path.dirname(pj_dir)
dataprocess_dir = os.path.dirname(convert_dir)
# 5. 将 dataprocess 目录添加到 Python 搜索路径
sys.path.append(dataprocess_dir)

from utils import (
    # 图像处理
    enhance_image, sliding_window_crop, calculate_stride,
    # 标签处理
    read_yolo_labels, save_yolo_labels,
    denormalize_bbox, normalize_bbox,
    adjust_bbox_for_crop, clip_polygon_to_window,
    calculate_polygon_area,
    # 数据集管理
    create_directory_structure, balance_dataset,
    read_dataset_yaml, update_dataset_yaml
)
from utils.constants import (
    DEFAULT_OVERLAP_RATIO, DEFAULT_WINDOW_SIZE,
    DEFAULT_JPEG_QUALITY, MIN_BBOX_RATIO, MIN_POLYGON_AREA_RATIO
)


class YOLOSlidingWindowProcessor:
    """YOLO格式数据的滑动窗口处理器"""

    def __init__(self,
                 overlap_ratio: float = DEFAULT_OVERLAP_RATIO,
                 enhance_mode: str = 'original',
                 label_mode: str = 'det',
                 min_bbox_ratio: float = MIN_BBOX_RATIO,
                 min_polygon_area_ratio: float = MIN_POLYGON_AREA_RATIO,
                 no_slice: bool = False):
        """
        初始化处理器

        Args:
            overlap_ratio: 滑动窗口重叠率
            enhance_mode: 增强模式 ('original' 或 'windowing')
            label_mode: 标签模式 ('det' 或 'seg')
            min_bbox_ratio: 最小边界框尺寸比例
            min_polygon_area_ratio: 最小多边形面积比例
            no_slice: 是否不进行切片，仅增强图像
        """
        self.overlap_ratio = overlap_ratio
        self.enhance_mode = enhance_mode
        self.label_mode = label_mode
        self.min_bbox_ratio = min_bbox_ratio
        self.min_polygon_area_ratio = min_polygon_area_ratio
        self.no_slice = no_slice

        # 统计信息
        self.stats = {
            'train': {'processed': 0, 'with_defects': 0, 'without_defects': 0},
            'val': {'processed': 0, 'with_defects': 0, 'without_defects': 0}
        }

        print(f"YOLO{'增强' if no_slice else '滑动窗口'}处理器初始化:")
        if not no_slice:
            print(f"  - 重叠率: {overlap_ratio}")
        print(f"  - 增强模式: {enhance_mode}")
        print(f"  - 标签模式: {label_mode}")
        print(f"  - 仅增强不切片: {no_slice}")

    def adjust_yolo_labels_for_crop(self, labels: List[List[float]],
                                    crop_x: int, crop_y: int,
                                    crop_w: int, crop_h: int,
                                    original_w: int, original_h: int) -> List[List[float]]:
        """
        根据裁剪区域调整YOLO标签

        Args:
            labels: 原始YOLO标签列表
            crop_x, crop_y: 裁剪区域左上角
            crop_w, crop_h: 裁剪区域尺寸
            original_w, original_h: 原始图像尺寸

        Returns:
            调整后的标签列表
        """
        adjusted_labels = []

        if self.label_mode == 'det':
            # 检测模式：处理边界框
            for label in labels:
                if len(label) < 5:
                    continue

                adjusted_bbox = adjust_bbox_for_crop(
                    label, crop_x, crop_y, crop_w, crop_h,
                    original_w, original_h
                )

                if adjusted_bbox and adjusted_bbox[3] > self.min_bbox_ratio and \
                        adjusted_bbox[4] > self.min_bbox_ratio:
                    adjusted_labels.append(adjusted_bbox)

        elif self.label_mode == 'seg':
            # 分割模式：处理多边形
            for label in labels:
                if len(label) < 7:  # class_id + 至少3个点
                    continue

                class_id = int(label[0])
                polygon_points = label[1:]

                # 转换为像素坐标
                pixel_points = []
                for i in range(0, len(polygon_points), 2):
                    if i + 1 >= len(polygon_points):
                        break
                    x = polygon_points[i] * original_w
                    y = polygon_points[i + 1] * original_h
                    pixel_points.extend([x, y])

                # 调整到裁剪窗口坐标系
                adjusted_pixel_points = []
                for i in range(0, len(pixel_points), 2):
                    x = pixel_points[i] - crop_x
                    y = pixel_points[i + 1] - crop_y
                    adjusted_pixel_points.extend([x, y])

                # 转换为归一化坐标
                norm_points = []
                for i in range(0, len(adjusted_pixel_points), 2):
                    norm_x = adjusted_pixel_points[i] / crop_w
                    norm_y = adjusted_pixel_points[i + 1] / crop_h
                    norm_points.extend([norm_x, norm_y])

                # 裁剪多边形到窗口内
                clipped_points = clip_polygon_to_window(norm_points, (0.0, 0.0, 1.0, 1.0))

                if len(clipped_points) >= 6:
                    # 计算面积，过滤太小的
                    area = calculate_polygon_area(clipped_points)
                    if area > self.min_polygon_area_ratio:
                        new_label = [class_id] + clipped_points
                        adjusted_labels.append(new_label)

        return adjusted_labels

    def process_single_image_no_slice(self, image_path: str, label_path: str,
                                      output_image_dir: str, output_label_dir: str) -> Dict:
        """
        处理单张图像（仅增强，不切片）

        Args:
            image_path: 输入图像路径
            label_path: YOLO格式标签文件路径
            output_image_dir: 输出图像目录
            output_label_dir: 输出标签目录

        Returns:
            处理统计信息
        """
        # 读取图像
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"无法读取图像: {image_path}")
            return {'processed': 0, 'with_defects': 0, 'without_defects': 0}

        # 读取YOLO标签
        labels = read_yolo_labels(label_path, self.label_mode)

        # 图像增强
        enhanced_image = enhance_image(image, self.enhance_mode)

        # 生成文件名
        base_name = Path(image_path).stem

        # 保存增强后的图像
        image_save_path = Path(output_image_dir) / f"{base_name}.jpg"
        cv2.imwrite(str(image_save_path), enhanced_image,
                    [cv2.IMWRITE_JPEG_QUALITY, DEFAULT_JPEG_QUALITY])

        # 直接复制标签文件
        label_save_path = Path(output_label_dir) / f"{base_name}.txt"
        if Path(label_path).exists():
            shutil.copy2(label_path, label_save_path)
        else:
            # 创建空标签文件
            with open(label_save_path, 'w') as f:
                pass

        # 统计信息
        stats = {'processed': 1, 'with_defects': 0, 'without_defects': 0}
        if len(labels) > 0:
            stats['with_defects'] = 1
        else:
            stats['without_defects'] = 1

        return stats

    def process_single_image(self, image_path: str, label_path: str,
                             output_image_dir: str, output_label_dir: str,
                             window_size: Tuple[int, int] = None) -> Dict:
        """
        处理单张YOLO格式标注的图像

        Args:
            image_path: 输入图像路径
            label_path: YOLO格式标签文件路径
            output_image_dir: 输出图像目录
            output_label_dir: 输出标签目录
            window_size: 窗口大小

        Returns:
            处理统计信息
        """
        # 如果不切片，调用专门的处理函数
        if self.no_slice:
            return self.process_single_image_no_slice(
                image_path, label_path, output_image_dir, output_label_dir
            )

        # 原有的切片处理逻辑
        # 读取图像
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"无法读取图像: {image_path}")
            return {'processed': 0, 'with_defects': 0, 'without_defects': 0}

        h, w = image.shape[:2]

        # 读取YOLO标签
        labels = read_yolo_labels(label_path, self.label_mode)

        # 确定滑动窗口大小
        if window_size is None:
            window_size = min(DEFAULT_WINDOW_SIZE, min(h, w))
            window_size = (window_size, window_size)

        # 计算步长
        stride = calculate_stride(window_size, self.overlap_ratio)

        # 滑动窗口裁剪
        patches = sliding_window_crop(image, window_size, stride)

        # 统计信息
        stats = {'processed': 0, 'with_defects': 0, 'without_defects': 0}

        # 处理每个patch
        base_name = Path(image_path).stem
        for i, patch_info in enumerate(patches):
            # 图像增强
            enhanced_patch = enhance_image(patch_info['patch'], self.enhance_mode)

            # 调整标签
            x, y = patch_info['position']
            patch_w, patch_h = patch_info['size']
            adjusted_labels = self.adjust_yolo_labels_for_crop(
                labels, x, y, patch_w, patch_h, w, h
            )

            # 生成文件名
            patch_name = f"{base_name}_patch_{i:04d}"

            # 保存图像
            image_save_path = Path(output_image_dir) / f"{patch_name}.jpg"
            cv2.imwrite(str(image_save_path), enhanced_patch,
                        [cv2.IMWRITE_JPEG_QUALITY, DEFAULT_JPEG_QUALITY])

            # 保存标签
            label_save_path = Path(output_label_dir) / f"{patch_name}.txt"
            save_yolo_labels(adjusted_labels, str(label_save_path), self.label_mode)

            # 更新统计
            stats['processed'] += 1
            if len(adjusted_labels) > 0:
                stats['with_defects'] += 1
            else:
                stats['without_defects'] += 1

        return stats

    def process_dataset(self, input_dir: str, output_dir: str,
                        window_size: Tuple[int, int] = None):
        """
        处理整个YOLO数据集

        Args:
            input_dir: 输入数据集目录
            output_dir: 输出数据集目录
            window_size: 窗口大小
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)

        # 验证输入目录结构
        if not (input_path / 'images').exists() or not (input_path / 'labels').exists():
            raise ValueError(f"输入目录必须包含images/和labels/子目录")

        # 创建输出目录结构
        create_directory_structure(output_path)

        processing_type = "增强" if self.no_slice else "滑动窗口处理"
        print(f"开始{processing_type}YOLO数据集...")
        print(f"  - 输入目录: {input_dir}")
        print(f"  - 输出目录: {output_dir}")
        if not self.no_slice:
            print(f"  - 窗口大小: {window_size if window_size else '自动'}")

        # 处理train和val数据
        for split in ['train', 'val']:
            print(f"\n处理{split}数据集...")

            input_image_dir = input_path / 'images' / split
            input_label_dir = input_path / 'labels' / split
            output_image_dir = output_path / 'images' / split
            output_label_dir = output_path / 'labels' / split

            if not input_image_dir.exists():
                print(f"跳过{split}（不存在）")
                continue

            # 获取所有图像文件
            image_files = list(input_image_dir.glob('*.jpg')) + \
                          list(input_image_dir.glob('*.png')) + \
                          list(input_image_dir.glob('*.bmp'))

            print(f"找到{len(image_files)}张图像")

            # 处理每张图像
            for image_file in tqdm(image_files, desc=f"处理{split}"):
                label_file = input_label_dir / f"{image_file.stem}.txt"

                # 即使标签文件不存在也处理
                if not label_file.exists():
                    # 创建空标签文件
                    with open(label_file, 'w') as f:
                        pass

                stats = self.process_single_image(
                    str(image_file),
                    str(label_file),
                    str(output_image_dir),
                    str(output_label_dir),
                    window_size=window_size
                )

                # 累加统计
                for key in stats:
                    self.stats[split][key] += stats[key]

        # 复制并更新dataset.yaml
        self._update_dataset_yaml(input_path, output_path, window_size)

        # 打印统计信息
        self._print_statistics()

    def _update_dataset_yaml(self, input_path: Path, output_path: Path,
                             window_size: Optional[Tuple[int, int]]):
        """更新dataset.yaml文件"""
        input_yaml = input_path / 'dataset.yaml'
        output_yaml = output_path / 'dataset.yaml'

        if input_yaml.exists():
            yaml_data = read_dataset_yaml(str(input_yaml))

            # 更新路径
            yaml_data['train'] = str(output_path.absolute() / 'images' / 'train')
            yaml_data['val'] = str(output_path.absolute() / 'images' / 'val')

            # 添加处理信息
            preprocessing_info = {
                'enhance_mode': self.enhance_mode,
                'label_mode': self.label_mode,
                'no_slice': self.no_slice
            }

            if not self.no_slice:
                preprocessing_info['window_size'] = list(window_size) if window_size else 'auto'
                preprocessing_info['overlap_ratio'] = self.overlap_ratio

            yaml_data['preprocessing'] = preprocessing_info

            # 保存更新后的yaml
            update_dataset_yaml(str(output_yaml), yaml_data)

            print(f"\ndataset.yaml已保存到: {output_yaml}")
        else:
            print(f"警告: 未找到{input_yaml}")

    def _print_statistics(self):
        """打印统计信息"""
        print(f"\n{'=' * 60}")
        print("处理完成统计:")
        print(f"{'=' * 60}")

        unit = "images" if self.no_slice else "patches"

        for split in ['train', 'val']:
            stats = self.stats[split]
            if stats['processed'] > 0:
                print(f"\n{split.upper()}数据集:")
                print(f"  总{unit}数: {stats['processed']}")
                print(
                    f"  有缺陷{unit}: {stats['with_defects']} ({stats['with_defects'] / stats['processed'] * 100:.1f}%)")
                print(
                    f"  无缺陷{unit}: {stats['without_defects']} ({stats['without_defects'] / stats['processed'] * 100:.1f}%)")

        total_processed = sum(s['processed'] for s in self.stats.values())
        total_with_defects = sum(s['with_defects'] for s in self.stats.values())
        total_without_defects = sum(s['without_defects'] for s in self.stats.values())

        print(f"\n总计:")
        print(f"  总{unit}数: {total_processed}")
        print(f"  有缺陷{unit}: {total_with_defects} ({total_with_defects / max(1, total_processed) * 100:.1f}%)")
        print(f"  无缺陷{unit}: {total_without_defects} ({total_without_defects / max(1, total_processed) * 100:.1f}%)")

    def balance_dataset(self, dataset_path: str, target_ratio: float = 1.0):
        """
        平衡数据集

        Args:
            dataset_path: 数据集路径
            target_ratio: 目标比例（无缺陷/有缺陷）
        """
        print(f"\n开始平衡数据集...")
        dataset_path = Path(dataset_path)

        # 分别平衡训练集和验证集
        for split in ['train', 'val']:
            label_dir = dataset_path / 'labels' / split
            image_dir = dataset_path / 'images' / split

            if not label_dir.exists():
                continue

            balance_stats = balance_dataset(
                str(label_dir),
                str(image_dir),
                target_ratio=target_ratio
            )

            print(f"{split}集: 保留{balance_stats['kept_unlabeled']}个无缺陷样本，"
                  f"删除{balance_stats['removed_count']}个")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description='YOLO ROI数据集滑动窗口处理工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基本使用（检测模式）
  python patchandenhance.py --input_dir ./roi_dataset --output_dir ./patched_dataset

  # 仅增强不切片
  python patchandenhance.py --input_dir ./roi_dataset --output_dir ./enhanced_dataset --no_slice

  # 分割模式
  python patchandenhance.py --input_dir ./roi_dataset --output_dir ./patched_dataset --label_mode seg

  # 设置窗口大小为512x512
  python patchandenhance.py --input_dir ./roi_dataset --output_dir ./patched_dataset --window_size 512 512

  # 设置重叠率为0.3
  python patchandenhance.py --input_dir ./roi_dataset --output_dir ./patched_dataset --overlap 0.3

  # 使用窗宽窗位增强模式
  python patchandenhance.py --input_dir ./roi_dataset --output_dir ./patched_dataset --enhance_mode windowing

  # 仅增强不切片，使用窗宽窗位增强
  python patchandenhance.py --input_dir ./roi_dataset --output_dir ./enhanced_dataset --no_slice --enhance_mode windowing

  # 平衡数据集（1:1比例）
  python patchandenhance.py --input_dir ./roi_dataset --output_dir ./patched_dataset --balance

  # 平衡数据集（1:2比例）
  python patchandenhance.py --input_dir ./roi_dataset --output_dir ./patched_dataset --balance --balance_ratio 2.0
        """
    )

    parser.add_argument('--input_dir', type=str, required=True,
                        help='输入目录路径（YOLO格式，包含images/和labels/）')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出目录路径')
    parser.add_argument('--window_size', type=int, nargs=2, default=None,
                        help='窗口大小 [width height]，默认自动确定（仅在切片模式下有效）')
    parser.add_argument('--overlap', type=float, default=0.5,
                        help='滑动窗口重叠率 (0.0-1.0)，默认0.5（仅在切片模式下有效）')
    parser.add_argument('--enhance_mode', type=str, choices=['original', 'windowing'],
                        default='original',
                        help='图像增强模式: original(直方图均衡+CLAHE) 或 windowing(窗宽窗位)')
    parser.add_argument('--label_mode', type=str, choices=['det', 'seg'],
                        default='det',
                        help='标签模式: det(检测边界框) 或 seg(分割多边形)')
    parser.add_argument('--no_slice', action='store_true',
                        help='不进行切片，仅对图像进行增强')
    parser.add_argument('--balance', action='store_true',
                        help='平衡数据集')
    parser.add_argument('--balance_ratio', type=float, default=1.0,
                        help='平衡比例（无缺陷/有缺陷），默认1.0')

    args = parser.parse_args()

    # 处理窗口大小参数
    window_size = tuple(args.window_size) if args.window_size else None

    # 如果选择不切片，窗口大小参数无效
    if args.no_slice and args.window_size:
        print("警告: --no_slice 模式下，--window_size 参数将被忽略")
        window_size = None

    # 创建处理器
    processor = YOLOSlidingWindowProcessor(
        overlap_ratio=args.overlap,
        enhance_mode=args.enhance_mode,
        label_mode=args.label_mode,
        no_slice=args.no_slice
    )

    # 处理数据集
    processor.process_dataset(
        args.input_dir,
        args.output_dir,
        window_size=window_size
    )

    # 如果需要平衡数据集
    if args.balance:
        processor.balance_dataset(
            args.output_dir,
            target_ratio=args.balance_ratio
        )

        # 重新打印统计信息
        print("\n平衡后的数据集已保存")


if __name__ == "__main__":
    main()