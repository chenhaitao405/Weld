"""
脚本名称: yolo_roi_extractor.py
功能概述: 使用YOLO模型从数据集中提取ROI区域并重新计算标签
详细说明:
    - 输入格式: YOLO格式数据集 + YOLO模型权重
    - 处理流程: 模型推理 → ROI检测 → 图像裁剪 → 标签重计算
    - 输出格式: 仅包含ROI区域的YOLO数据集
依赖模块: utils.label_processing, utils.dataset_management, ultralytics
使用示例:
    # 基本使用（检测模式）
    python yolo_roi_extractor.py --input_dir ./dataset --output_dir ./roi_dataset --model_path ./weights/best.pt

    # 分割模式
    python yolo_roi_extractor.py --input_dir ./dataset --output_dir ./roi_dataset --model_path ./weights/best.pt --mode seg

    # 调整ROI检测阈值
    python yolo_roi_extractor.py --input_dir ./dataset --output_dir ./roi_dataset --model_path ./weights/best.pt --roi_conf 0.5

    # 增加ROI区域padding
    python yolo_roi_extractor.py --input_dir ./dataset --output_dir ./roi_dataset --model_path ./weights/best.pt --padding 0.2
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from tqdm import tqdm
from ultralytics import YOLO

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


current_script_path = os.path.abspath(__file__)
pj_dir = os.path.dirname(current_script_path)
convert_dir = os.path.dirname(pj_dir)
dataprocess_dir = os.path.dirname(convert_dir)
# 5. 将 dataprocess 目录添加到 Python 搜索路径
sys.path.append(dataprocess_dir)


from utils import (
    read_yolo_labels,
    save_yolo_labels,
    denormalize_bbox,
    normalize_bbox,
    clip_polygon_to_window,
    create_directory_structure,
    read_dataset_yaml,
    update_dataset_yaml
)


class YOLOROIExtractor:
    """YOLO ROI区域提取器"""

    def __init__(self,
                 input_dir: str,
                 output_dir: str,
                 model_path: str,
                 mode: str = 'det',
                 roi_conf_threshold: float = 0.25,
                 roi_iou_threshold: float = 0.45,
                 padding_ratio: float = 0.1):
        """
        初始化ROI提取器

        Args:
            input_dir: 输入YOLO数据集目录
            output_dir: 输出YOLO数据集目录
            model_path: YOLO模型权重路径
            mode: 'det'(检测) 或 'seg'(分割)
            roi_conf_threshold: ROI检测置信度阈值
            roi_iou_threshold: ROI检测IOU阈值
            padding_ratio: ROI区域padding比例
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.model_path = model_path
        self.mode = mode
        self.roi_conf_threshold = roi_conf_threshold
        self.roi_iou_threshold = roi_iou_threshold
        self.padding_ratio = padding_ratio

        # 加载YOLO模型
        print(f"加载模型: {model_path}")
        self.model = YOLO(model_path)

        # 创建输出目录结构
        create_directory_structure(self.output_dir)

        # 统计信息
        self.total_processed = 0
        self.total_roi_found = 0
        self.total_labels_adjusted = 0

        print(f"YOLO ROI提取器初始化:")
        print(f"  - 输入目录: {input_dir}")
        print(f"  - 输出目录: {output_dir}")
        print(f"  - 模式: {mode}")
        print(f"  - ROI置信度阈值: {roi_conf_threshold}")
        print(f"  - ROI IOU阈值: {roi_iou_threshold}")
        print(f"  - Padding比例: {padding_ratio}")

    def _detect_roi(self, image_path: str) -> List[Tuple[int, int, int, int]]:
        """
        使用YOLO模型检测ROI区域

        Args:
            image_path: 图像路径

        Returns:
            ROI边界框列表 [(x1, y1, x2, y2), ...]
        """
        results = self.model(
            image_path,
            conf=self.roi_conf_threshold,
            iou=self.roi_iou_threshold,
            verbose=False
        )

        roi_boxes = []
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2 = box
                    roi_boxes.append((int(x1), int(y1), int(x2), int(y2)))

        return roi_boxes

    def _add_padding(self, x1: int, y1: int, x2: int, y2: int,
                    img_width: int, img_height: int) -> Tuple[int, int, int, int]:
        """
        为ROI区域添加padding

        Args:
            x1, y1, x2, y2: ROI边界框
            img_width, img_height: 图像尺寸

        Returns:
            添加padding后的边界框
        """
        width = x2 - x1
        height = y2 - y1

        # 计算padding
        pad_x = int(width * self.padding_ratio)
        pad_y = int(height * self.padding_ratio)

        # 添加padding并确保不超出图像边界
        x1_padded = max(0, x1 - pad_x)
        y1_padded = max(0, y1 - pad_y)
        x2_padded = min(img_width, x2 + pad_x)
        y2_padded = min(img_height, y2 + pad_y)

        return x1_padded, y1_padded, x2_padded, y2_padded

    def _process_detection_label(self, label: list, roi_x1: int, roi_y1: int,
                                roi_x2: int, roi_y2: int,
                                img_width: int, img_height: int,
                                cropped_width: int, cropped_height: int) -> Optional[list]:
        """
        处理检测模式的标签

        Args:
            label: [class_id, x_center, y_center, width, height]
            roi_*: ROI区域像素坐标
            img_*: 原始图像尺寸
            cropped_*: 裁剪后图像尺寸

        Returns:
            调整后的标签或None
        """
        class_id = int(label[0])

        # 转换为像素坐标
        x1, y1, x2, y2 = denormalize_bbox(
            label[1], label[2], label[3], label[4],
            img_width, img_height
        )

        # 计算与ROI的交集
        intersect_x1 = max(x1, roi_x1)
        intersect_y1 = max(y1, roi_y1)
        intersect_x2 = min(x2, roi_x2)
        intersect_y2 = min(y2, roi_y2)

        # 如果没有交集
        if intersect_x1 >= intersect_x2 or intersect_y1 >= intersect_y2:
            return None

        # 转换为相对于裁剪图像的坐标
        new_x1 = max(0, intersect_x1 - roi_x1)
        new_y1 = max(0, intersect_y1 - roi_y1)
        new_x2 = min(cropped_width, intersect_x2 - roi_x1)
        new_y2 = min(cropped_height, intersect_y2 - roi_y1)

        # 转换回归一化坐标
        new_x_center, new_y_center, new_width, new_height = normalize_bbox(
            new_x1, new_y1, new_x2, new_y2, cropped_width, cropped_height
        )

        # 过滤太小的边界框
        if new_width <= 0.01 or new_height <= 0.01:
            return None

        return [class_id, new_x_center, new_y_center, new_width, new_height]

    def _process_segmentation_label(self, label: list, roi_x1: int, roi_y1: int,
                                   roi_x2: int, roi_y2: int,
                                   img_width: int, img_height: int,
                                   cropped_width: int, cropped_height: int) -> Optional[list]:
        """
        处理分割模式的标签

        Args:
            label: [class_id, x1, y1, x2, y2, ...]
            roi_*: ROI区域像素坐标
            img_*: 原始图像尺寸
            cropped_*: 裁剪后图像尺寸

        Returns:
            调整后的标签或None
        """
        class_id = int(label[0])
        points = label[1:]

        # 转换为像素坐标并调整到ROI区域
        new_points = []
        for i in range(0, len(points), 2):
            if i + 1 < len(points):
                x = points[i] * img_width
                y = points[i + 1] * img_height

                # 调整到ROI区域坐标系
                x_adjusted = x - roi_x1
                y_adjusted = y - roi_y1

                # 归一化到裁剪图像
                new_x = x_adjusted / cropped_width
                new_y = y_adjusted / cropped_height

                new_points.extend([new_x, new_y])

        # 裁剪多边形到窗口内
        clipped_points = clip_polygon_to_window(new_points, (0.0, 0.0, 1.0, 1.0))

        # 检查是否有效
        if len(clipped_points) < 6:  # 至少3个点
            return None

        # 计算多边形面积，过滤太小的
        x_coords = clipped_points[::2]
        y_coords = clipped_points[1::2]

        if not x_coords or not y_coords:
            return None

        poly_width = max(x_coords) - min(x_coords)
        poly_height = max(y_coords) - min(y_coords)

        if poly_width <= 0.01 or poly_height <= 0.01:
            return None

        return [class_id] + clipped_points

    def _process_single_image(self, image_path: Path, label_path: Path,
                            split_type: str):
        """
        处理单张图像

        Args:
            image_path: 图像路径
            label_path: 标签路径
            split_type: 'train' 或 'val'
        """
        # 读取图像
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"警告: 无法读取图像 {image_path}")
            return

        img_height, img_width = img.shape[:2]

        # 检测ROI区域
        roi_boxes = self._detect_roi(str(image_path))

        if not roi_boxes:
            print(f"警告: 未检测到ROI {image_path}")
            return

        self.total_roi_found += len(roi_boxes)

        # 读取原始标签
        original_labels = read_yolo_labels(str(label_path), self.mode)

        # 处理每个ROI区域
        base_name = image_path.stem
        for roi_idx, (roi_x1, roi_y1, roi_x2, roi_y2) in enumerate(roi_boxes):
            # 添加padding
            roi_x1, roi_y1, roi_x2, roi_y2 = self._add_padding(
                roi_x1, roi_y1, roi_x2, roi_y2, img_width, img_height
            )

            # 裁剪图像
            cropped_img = img[roi_y1:roi_y2, roi_x1:roi_x2]
            cropped_height, cropped_width = cropped_img.shape[:2]

            # 生成新文件名
            new_img_name = f"{base_name}_roi_{roi_idx}.jpg"
            new_label_name = f"{base_name}_roi_{roi_idx}.txt"

            # 保存裁剪后的图像
            output_img_path = self.output_dir / 'images' / split_type / new_img_name
            cv2.imwrite(str(output_img_path), cropped_img,
                       [cv2.IMWRITE_JPEG_QUALITY, 95])

            # 处理标签
            new_labels = []
            for label in original_labels:
                if self.mode == 'det':
                    new_label = self._process_detection_label(
                        label, roi_x1, roi_y1, roi_x2, roi_y2,
                        img_width, img_height, cropped_width, cropped_height
                    )
                else:  # seg mode
                    new_label = self._process_segmentation_label(
                        label, roi_x1, roi_y1, roi_x2, roi_y2,
                        img_width, img_height, cropped_width, cropped_height
                    )

                if new_label is not None:
                    new_labels.append(new_label)
                    self.total_labels_adjusted += 1

            # 保存新的标签文件
            output_label_path = self.output_dir / 'labels' / split_type / new_label_name
            save_yolo_labels(new_labels, str(output_label_path), self.mode)

        self.total_processed += 1

    def process_dataset(self):
        """处理整个数据集"""
        print(f"开始处理数据集...")

        # 处理训练集和验证集
        for split_type in ['train', 'val']:
            image_dir = self.input_dir / 'images' / split_type
            label_dir = self.input_dir / 'labels' / split_type

            if not image_dir.exists():
                print(f"跳过{split_type}（不存在）")
                continue

            # 获取所有图像文件
            image_files = list(image_dir.glob('*.jpg')) + \
                         list(image_dir.glob('*.jpeg')) + \
                         list(image_dir.glob('*.png')) + \
                          list(image_dir.glob('*.tif')) + \
                          list(image_dir.glob('*.bmp'))

            print(f"\n处理{split_type}集: {len(image_files)}张图像")

            # 处理每张图像
            for image_path in tqdm(image_files, desc=f"处理{split_type}"):
                # 构造对应的标签文件路径
                label_path = label_dir / f"{image_path.stem}.txt"

                # 即使标签文件不存在也处理
                if not label_path.exists():
                    # 创建空标签文件
                    label_path = Path("/dev/null")

                self._process_single_image(image_path, label_path, split_type)

        # 复制并更新dataset.yaml
        self._update_dataset_yaml()

        # 打印统计信息
        self._print_statistics()

    def _update_dataset_yaml(self):
        """更新dataset.yaml文件"""
        input_yaml = self.input_dir / 'dataset.yaml'
        output_yaml = self.output_dir / 'dataset.yaml'

        if input_yaml.exists():
            yaml_data = read_dataset_yaml(str(input_yaml))

            # 更新路径
            yaml_data['train'] = str(self.output_dir / 'images' / 'train')
            yaml_data['val'] = str(self.output_dir / 'images' / 'val')

            # 添加ROI提取信息
            yaml_data['roi_extraction'] = {
                'model_path': str(self.model_path),
                'conf_threshold': self.roi_conf_threshold,
                'iou_threshold': self.roi_iou_threshold,
                'padding_ratio': self.padding_ratio
            }

            # 保存更新后的yaml
            update_dataset_yaml(str(output_yaml), yaml_data)

            print(f"dataset.yaml已保存到: {output_yaml}")
        else:
            print(f"警告: 未找到{input_yaml}")

    def _print_statistics(self):
        """打印统计信息"""
        print(f"\n{'='*60}")
        print(f"✅ ROI提取完成！")
        print(f"📊 统计信息:")
        print(f"  - 处理图像数: {self.total_processed}")
        print(f"  - 检测到的ROI数: {self.total_roi_found}")
        print(f"  - 调整的标签数: {self.total_labels_adjusted}")
        print(f"  - 平均每张图像ROI数: {self.total_roi_found/max(1, self.total_processed):.2f}")
        print(f"  - 输出目录: {self.output_dir}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='使用YOLO模型从数据集中提取ROI区域',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基本使用（检测模式）
  python yolo_roi_extractor.py --input_dir ./dataset --output_dir ./roi_dataset --model_path ./weights/best.pt
  
  # 分割模式
  python yolo_roi_extractor.py --input_dir ./dataset --output_dir ./roi_dataset --model_path ./weights/best.pt --mode seg
  
  # 调整ROI检测阈值
  python yolo_roi_extractor.py --input_dir ./dataset --output_dir ./roi_dataset --model_path ./weights/best.pt --roi_conf 0.5 --roi_iou 0.7
  
  # 增加ROI区域padding（20%）
  python yolo_roi_extractor.py --input_dir ./dataset --output_dir ./roi_dataset --model_path ./weights/best.pt --padding 0.2
        """
    )

    parser.add_argument('--input_dir', type=str, required=True,
                       help='输入YOLO数据集目录')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='输出ROI数据集目录')
    parser.add_argument('--model_path', type=str, required=True,
                       help='YOLO模型权重路径（.pt文件）')
    parser.add_argument('--mode', type=str, choices=['det', 'seg'], default='det',
                       help='数据集模式: det(检测) 或 seg(分割) (默认: det)')
    parser.add_argument('--roi_conf', type=float, default=0.25,
                       help='ROI检测置信度阈值 (默认: 0.25)')
    parser.add_argument('--roi_iou', type=float, default=0.45,
                       help='ROI检测IOU阈值 (默认: 0.45)')
    parser.add_argument('--padding', type=float, default=0.1,
                       help='ROI区域padding比例 (默认: 0.1)')

    args = parser.parse_args()

    # 创建ROI提取器
    extractor = YOLOROIExtractor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        mode=args.mode,
        roi_conf_threshold=args.roi_conf,
        roi_iou_threshold=args.roi_iou,
        padding_ratio=args.padding
    )

    # 处理数据集
    extractor.process_dataset()


if __name__ == '__main__':
    main()