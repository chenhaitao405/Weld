import os
import sys
import cv2
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm
import random
import shutil

# 模式3相关常量
MODE3_ASPECT_THRESHOLD = 4.0
MODE3_WINDOW_RATIO = 2.0
MODE3_OVERLAP = 0.3
MODE3_TARGET_SIZE = 1120

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
from utils.wide_slice_utils import (
    WideSliceParams,
    WideSlicePatch,
    WideSlicePlan,
    build_wide_slice_plan,
    resize_slice_to_square,
    stack_wide_slice_pair
)


class YOLOSlidingWindowProcessor:
    """YOLO格式数据的滑动窗口处理器"""

    def __init__(self,
                 overlap_ratio: float = DEFAULT_OVERLAP_RATIO,
                 enhance_mode: str = 'original',
                 label_mode: str = 'det',
                 min_bbox_ratio: float = MIN_BBOX_RATIO,
                 min_polygon_area_ratio: float = MIN_POLYGON_AREA_RATIO,
                 no_slice: bool = False,
                 slice_mode: int = 2):
        """
        初始化处理器

        Args:
            overlap_ratio: 滑动窗口重叠率
            enhance_mode: 增强模式 ('original' 或 'windowing')
            label_mode: 标签模式 ('det' 或 'seg')
            min_bbox_ratio: 最小边界框尺寸比例
            min_polygon_area_ratio: 最小多边形面积比例
            no_slice: 是否不进行切片，仅增强图像（兼容旧参数，等价于 slice_mode=1）
            slice_mode: 1=仅增强, 2=滑动裁剪, 3=横切纵拼方形
        """
        if no_slice:
            slice_mode = 1
        if slice_mode not in (1, 2, 3):
            raise ValueError("slice_mode 必须是 1/2/3")

        self.overlap_ratio = overlap_ratio
        self.enhance_mode = enhance_mode
        self.label_mode = label_mode
        self.min_bbox_ratio = min_bbox_ratio
        self.min_polygon_area_ratio = min_polygon_area_ratio
        self.slice_mode = slice_mode
        self.no_slice = slice_mode == 1

        # 统计信息
        self.stats = {
            'train': {'processed': 0, 'with_defects': 0, 'without_defects': 0},
            'val': {'processed': 0, 'with_defects': 0, 'without_defects': 0}
        }

        print(f"YOLO处理器初始化 - 模式 {slice_mode}: {self._describe_slice_mode()}")
        if slice_mode == 2:
            print(f"  - 重叠率: {overlap_ratio}")
        print(f"  - 增强模式: {enhance_mode}")
        print(f"  - 标签模式: {label_mode}")
        print(f"  - 模式切换参数: {slice_mode}")

    def _describe_slice_mode(self) -> str:
        descriptions = {
            1: "仅增强（不切片）",
            2: "滑动窗口裁剪",
            3: "横切纵拼成方形"
        }
        return descriptions.get(self.slice_mode, "未知模式")

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

    def _save_mode3_output(self, image: np.ndarray, labels: List[List[float]],
                            output_image_dir: str, output_label_dir: str,
                            name: str) -> Dict[str, int]:
        """辅助函数：保存模式3生成的图像和标签"""
        resized = resize_slice_to_square(image, MODE3_TARGET_SIZE)
        image_save_path = Path(output_image_dir) / f"{name}.jpg"
        label_save_path = Path(output_label_dir) / f"{name}.txt"

        image_save_path.parent.mkdir(parents=True, exist_ok=True)
        label_save_path.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(image_save_path), resized, [cv2.IMWRITE_JPEG_QUALITY, DEFAULT_JPEG_QUALITY])
        save_yolo_labels(labels, str(label_save_path), self.label_mode)

        stats = {'processed': 1, 'with_defects': 0, 'without_defects': 0}
        if len(labels) > 0:
            stats['with_defects'] = 1
        else:
            stats['without_defects'] = 1
        return stats

    def _generate_mode3_patches(self, image: np.ndarray, labels: List[List[float]],
                                original_w: int, original_h: int) -> Tuple[WideSlicePlan, List[Dict[str, Any]]]:
        """按照模式3要求沿宽度滑动裁剪，并调整对应的标签"""
        params = WideSliceParams(
            aspect_ratio_threshold=MODE3_ASPECT_THRESHOLD,
            window_ratio=MODE3_WINDOW_RATIO,
            overlap=MODE3_OVERLAP,
            target_size=MODE3_TARGET_SIZE
        )
        plan: WideSlicePlan = build_wide_slice_plan(image, params)
        patch_entries: List[Dict[str, Any]] = []

        for patch in plan.patches:
            crop_x = int(round(patch.x_offset))
            adjusted_labels = self.adjust_yolo_labels_for_crop(
                labels,
                crop_x,
                0,
                patch.width,
                patch.height,
                original_w,
                original_h
            )
            patch_entries.append({
                'patch': patch,
                'labels': adjusted_labels
            })

        return plan, patch_entries

    def _merge_labels_for_vertical_stack(self,
                                         top_labels: List[List[float]],
                                         bottom_labels: List[List[float]],
                                         top_h: int,
                                         bottom_h: int) -> List[List[float]]:
        """将上下两个patch的标签映射到纵向拼接后的坐标系"""
        total_h = top_h + bottom_h
        merged: List[List[float]] = []

        if total_h == 0:
            return merged

        if self.label_mode == 'det':
            for labels, offset, patch_h in (
                (top_labels, 0, top_h),
                (bottom_labels, top_h, bottom_h)
            ):
                for label in labels:
                    if len(label) < 5:
                        continue
                    new_label = label.copy()
                    new_label[2] = (label[2] * patch_h + offset) / total_h
                    new_label[4] = (label[4] * patch_h) / total_h
                    merged.append(new_label)
        else:  # seg
            for labels, offset, patch_h in (
                (top_labels, 0, top_h),
                (bottom_labels, top_h, bottom_h)
            ):
                for label in labels:
                    if len(label) < 7:
                        continue
                    new_label = [label[0]]
                    coords: List[float] = []
                    for i in range(1, len(label), 2):
                        if i + 1 >= len(label):
                            break
                        x = label[i]
                        y = label[i + 1]
                        adjusted_y = (y * patch_h + offset) / total_h
                        coords.extend([x, adjusted_y])
                    if len(coords) >= 6:
                        new_label.extend(coords)
                        merged.append(new_label)

        return merged

    def process_single_image_mode3(self, image_path: str, label_path: str,
                                   output_image_dir: str,
                                   output_label_dir: str) -> Dict[str, int]:
        """模式3：横切纵拼生成方形"""
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"无法读取图像: {image_path}")
            return {'processed': 0, 'with_defects': 0, 'without_defects': 0}

        labels = read_yolo_labels(label_path, self.label_mode)
        h, w = image.shape[:2]
        if h == 0 or w == 0:
            return {'processed': 0, 'with_defects': 0, 'without_defects': 0}

        base_name = Path(image_path).stem
        stats = {'processed': 0, 'with_defects': 0, 'without_defects': 0}

        plan, patches = self._generate_mode3_patches(image, labels, w, h)
        if not patches:
            return stats

        # 先切片，再对每个切片单独增强，确保拼接前的图像质量一致
        for entry in patches:
            slice_patch: WideSlicePatch = entry['patch']
            slice_patch.image = enhance_image(slice_patch.image, self.enhance_mode)

        pending_full: Optional[Dict[str, Any]] = None
        pair_idx = 0
        single_idx = 0

        def update_stats(result: Dict[str, int]):
            for key in stats:
                stats[key] += result[key]

        if not plan.is_wide:
            output_name = f"{base_name}_sq1120"
            result = self._save_mode3_output(
                patches[0]['patch'].image,
                patches[0]['labels'],
                output_image_dir,
                output_label_dir,
                output_name
            )
            update_stats(result)
            return stats

        for entry in patches:
            patch: WideSlicePatch = entry['patch']
            patch_labels = entry['labels']
            if patch.is_full_window:
                if pending_full is None:
                    pending_full = entry
                else:
                    stacked_image = stack_wide_slice_pair(pending_full['patch'], patch)
                    merged_labels = self._merge_labels_for_vertical_stack(
                        pending_full['labels'], patch_labels,
                        pending_full['patch'].height, patch.height
                    )
                    name = f"{base_name}_pair_{pair_idx:04d}"
                    pair_idx += 1
                    result = self._save_mode3_output(
                        stacked_image, merged_labels,
                        output_image_dir, output_label_dir, name
                    )
                    update_stats(result)
                    pending_full = None
            else:
                if pending_full is not None:
                    name = f"{base_name}_single_{single_idx:04d}"
                    single_idx += 1
                    result = self._save_mode3_output(
                        pending_full['patch'].image, pending_full['labels'],
                        output_image_dir, output_label_dir, name
                    )
                    update_stats(result)
                    pending_full = None

                name = f"{base_name}_tail_{single_idx:04d}"
                single_idx += 1
                result = self._save_mode3_output(
                    patch.image, patch_labels,
                    output_image_dir, output_label_dir, name
                )
                update_stats(result)

        if pending_full is not None:
            name = f"{base_name}_single_{single_idx:04d}"
            single_idx += 1
            result = self._save_mode3_output(
                pending_full['patch'].image, pending_full['labels'],
                output_image_dir, output_label_dir, name
            )
            update_stats(result)

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
        # 根据模式分派
        if self.slice_mode == 1:
            return self.process_single_image_no_slice(
                image_path, label_path, output_image_dir, output_label_dir
            )
        if self.slice_mode == 3:
            return self.process_single_image_mode3(
                image_path, label_path, output_image_dir, output_label_dir
            )

        # 模式2：原有的切片处理逻辑
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

        mode_desc = self._describe_slice_mode()
        print(f"开始{mode_desc}流程...")
        print(f"  - 输入目录: {input_dir}")
        print(f"  - 输出目录: {output_dir}")
        if self.slice_mode == 2:
            print(f"  - 窗口大小: {window_size if window_size else '自动'}")
            print(f"  - 重叠率: {self.overlap_ratio}")
        elif self.slice_mode == 3:
            print(f"  - 模式3窗口宽度系数: {MODE3_WINDOW_RATIO}")
            print(f"  - 模式3滑动重叠率: {MODE3_OVERLAP}")

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
                'no_slice': self.slice_mode == 1,
                'slice_mode': self.slice_mode
            }

            if self.slice_mode == 2:
                preprocessing_info['window_size'] = list(window_size) if window_size else 'auto'
                preprocessing_info['overlap_ratio'] = self.overlap_ratio
            elif self.slice_mode == 3:
                preprocessing_info['mode3'] = {
                    'target_size': MODE3_TARGET_SIZE,
                    'aspect_ratio_threshold': MODE3_ASPECT_THRESHOLD,
                    'window_ratio': MODE3_WINDOW_RATIO,
                    'overlap': MODE3_OVERLAP
                }

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

        unit = "images" if self.slice_mode == 1 else "patches"

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
  python patchandenhance.py --input_dir ./roi_dataset --output_dir ./enhanced_dataset --slice_mode 1

  # 分割模式
  python patchandenhance.py --input_dir ./roi_dataset --output_dir ./patched_dataset --label_mode seg

  # 设置窗口大小为512x512
  python patchandenhance.py --input_dir ./roi_dataset --output_dir ./patched_dataset --window_size 512 512

  # 设置重叠率为0.3
  python patchandenhance.py --input_dir ./roi_dataset --output_dir ./patched_dataset --overlap 0.3

  # 使用窗宽窗位增强模式
  python patchandenhance.py --input_dir ./roi_dataset --output_dir ./patched_dataset --enhance_mode windowing

  # 仅增强不切片，使用窗宽窗位增强
  python patchandenhance.py --input_dir ./roi_dataset --output_dir ./enhanced_dataset --slice_mode 1 --enhance_mode windowing

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
                        help='窗口大小 [width height]，默认自动（仅模式2生效）')
    parser.add_argument('--overlap', type=float, default=0.5,
                        help='滑动窗口重叠率 (0.0-1.0)，默认0.5（仅模式2生效）')
    parser.add_argument('--enhance_mode', type=str, choices=['original', 'windowing'],
                        default='original',
                        help='图像增强模式: original(直方图均衡+CLAHE) 或 windowing(窗宽窗位)')
    parser.add_argument('--label_mode', type=str, choices=['det', 'seg'],
                        default='det',
                        help='标签模式: det(检测边界框) 或 seg(分割多边形)')
    parser.add_argument('--slice_mode', type=int, choices=[1, 2, 3], default=2,
                        help='切片模式: 1=仅增强, 2=滑动裁剪, 3=横裁纵拼')
    parser.add_argument('--no_slice', action='store_true',
                        help='兼容参数，等价于 --slice_mode 1')
    parser.add_argument('--balance', action='store_true',
                        help='平衡数据集')
    parser.add_argument('--balance_ratio', type=float, default=1.0,
                        help='平衡比例（无缺陷/有缺陷），默认1.0')

    args = parser.parse_args()

    slice_mode = args.slice_mode
    if args.no_slice:
        print("提示: --no_slice 已弃用，等价于 --slice_mode 1")
        slice_mode = 1

    window_size = tuple(args.window_size) if args.window_size else None
    if slice_mode != 2 and window_size is not None:
        print(f"提示: slice_mode={slice_mode} 下忽略 --window_size 参数")
        window_size = None

    if slice_mode != 2 and abs(args.overlap - DEFAULT_OVERLAP_RATIO) > 1e-6:
        print(f"提示: slice_mode={slice_mode} 下 --overlap 参数将被忽略")

    # 创建处理器
    processor = YOLOSlidingWindowProcessor(
        overlap_ratio=args.overlap,
        enhance_mode=args.enhance_mode,
        label_mode=args.label_mode,
        slice_mode=slice_mode
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
