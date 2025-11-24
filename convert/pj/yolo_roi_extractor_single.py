"""
脚本名称: yolo_roi_image_extractor.py
功能概述: 使用YOLO模型从图像中提取ROI区域（无标签版本）
详细说明:
    - 输入格式: 图像文件夹 + YOLO模型权重
    - 处理流程: 模型推理 → ROI检测 → 图像裁剪 → 保存
    - 输出格式: 裁剪后的ROI图像
    - 特殊功能: 未检测到ROI的图像会保存到./withoutROI子文件夹
使用示例:
    # 基本使用
    python yolo_roi_image_extractor.py --input_dir ./images --output_dir ./roi_images --model_path ./weights/best.pt

    # 调整ROI检测阈值
    python yolo_roi_image_extractor.py --input_dir ./images --output_dir ./roi_images --model_path ./weights/best.pt --roi_conf 0.5

    # 增加ROI区域padding
    python yolo_roi_image_extractor.py --input_dir ./images --output_dir ./roi_images --model_path ./weights/best.pt --padding 0.2

    # 递归处理子文件夹
    python yolo_roi_image_extractor.py --input_dir ./images --output_dir ./roi_images --model_path ./weights/best.pt --recursive
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm
from ultralytics import YOLO
import json
import shutil


class YOLOROIImageExtractor:
    """YOLO ROI图像提取器（无标签版本）"""

    def __init__(self,
                 input_dir: str,
                 output_dir: str,
                 model_path: str,
                 roi_conf_threshold: float = 0.25,
                 roi_iou_threshold: float = 0.45,
                 padding_ratio: float = 0.1,
                 save_format: str = 'jpg',
                 jpeg_quality: int = 95):
        """
        初始化ROI提取器

        Args:
            input_dir: 输入图像目录
            output_dir: 输出目录
            model_path: YOLO模型权重路径
            roi_conf_threshold: ROI检测置信度阈值
            roi_iou_threshold: ROI检测IOU阈值
            padding_ratio: ROI区域padding比例
            save_format: 保存格式 ('jpg', 'png')
            jpeg_quality: JPEG保存质量 (1-100)
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.model_path = model_path
        self.roi_conf_threshold = roi_conf_threshold
        self.roi_iou_threshold = roi_iou_threshold
        self.padding_ratio = padding_ratio
        self.save_format = save_format.lower()
        self.jpeg_quality = jpeg_quality

        # 验证输入目录
        if not self.input_dir.exists():
            raise ValueError(f"输入目录不存在: {input_dir}")

        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 创建无ROI图像的保存目录
        self.without_roi_dir = self.output_dir / 'withoutROI'
        self.without_roi_dir.mkdir(parents=True, exist_ok=True)

        # 加载YOLO模型
        print(f"加载模型: {model_path}")
        try:
            self.model = YOLO(model_path)
            print(f"模型加载成功！")
        except Exception as e:
            raise ValueError(f"模型加载失败: {e}")

        # 统计信息
        self.stats = {
            'total_images': 0,
            'images_with_roi': 0,
            'images_without_roi': 0,
            'total_roi_found': 0,
            'roi_per_class': {},
            'failed_images': [],
            'images_without_roi_list': []  # 新增：记录无ROI的图像列表
        }

        # 保存ROI信息用于后续分析
        self.roi_info = []

        # 保存无ROI图像信息
        self.no_roi_images = []

        print(f"\nYOLO ROI图像提取器初始化:")
        print(f"  - 输入目录: {input_dir}")
        print(f"  - 输出目录: {output_dir}")
        print(f"  - 无ROI图像目录: {self.without_roi_dir}")
        print(f"  - ROI置信度阈值: {roi_conf_threshold}")
        print(f"  - ROI IOU阈值: {roi_iou_threshold}")
        print(f"  - Padding比例: {padding_ratio}")
        print(f"  - 保存格式: {save_format}")

    def _normalize_image(self, img: np.ndarray) -> np.ndarray:
        """
        归一化图像到8位用于显示和保存

        Args:
            img: 输入图像

        Returns:
            归一化后的8位图像
        """
        if img.dtype == np.uint8:
            return img

        # 对于16位或其他深度的图像，进行归一化
        # 使用percentile来避免极端值的影响
        if len(img.shape) == 2:
            # 灰度图
            p2, p98 = np.percentile(img, (2, 98))
            img_scaled = np.clip(img, p2, p98)
            img_scaled = ((img_scaled - p2) / (p98 - p2) * 255).astype(np.uint8)
        else:
            # 多通道图像
            img_scaled = np.zeros_like(img, dtype=np.uint8)
            for i in range(img.shape[2]):
                p2, p98 = np.percentile(img[:, :, i], (2, 98))
                channel = np.clip(img[:, :, i], p2, p98)
                img_scaled[:, :, i] = ((channel - p2) / (p98 - p2) * 255).astype(np.uint8)

        return img_scaled

    def _detect_roi(self, image_path: str) -> List[Dict]:
        """
        使用YOLO模型检测ROI区域

        Args:
            image_path: 图像路径

        Returns:
            ROI信息列表，每个ROI包含边界框、置信度和类别
        """
        try:
            # 读取图像以检查通道数和位深度
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"无法读取图像: {image_path}")
                return []

            # 归一化图像到8位
            img_normalized = self._normalize_image(img)

            # 如果是单通道图像，转换为3通道
            # 确保图像是3通道RGB格式
            if len(img_normalized.shape) == 2:
                # 单通道灰度图 -> 3通道RGB
                img_rgb = cv2.cvtColor(img_normalized, cv2.COLOR_GRAY2RGB)
            elif len(img_normalized.shape) == 3:
                if img_normalized.shape[2] == 1:
                    # 单通道 -> 3通道RGB
                    img_rgb = cv2.cvtColor(img_normalized, cv2.COLOR_GRAY2RGB)
                elif img_normalized.shape[2] == 3:
                    # 已经是3通道，直接使用
                    img_rgb = img_normalized
                elif img_normalized.shape[2] == 4:
                    # 4通道（RGBA或其他）-> 3通道RGB，丢弃第4通道
                    img_rgb = img_normalized[:, :, :3]
                else:
                    # 其他通道数，取前3个通道
                    print(f"警告: 图像有 {img_normalized.shape[2]} 个通道，仅使用前3个通道")
                    img_rgb = img_normalized[:, :, :3]
            else:
                print(f"警告: 不支持的图像形状 {img_normalized.shape}")
                return []

            # 使用3通道图像进行推理
            results = self.model(
                img_rgb,
                conf=self.roi_conf_threshold,
                iou=self.roi_iou_threshold,
                verbose=False
            )

            roi_list = []
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confs = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy()

                    for box, conf, cls in zip(boxes, confs, classes):
                        x1, y1, x2, y2 = box
                        roi_info = {
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'confidence': float(conf),
                            'class_id': int(cls),
                            'class_name': self.model.names.get(int(cls), f'class_{int(cls)}')
                        }
                        roi_list.append(roi_info)

                        # 更新类别统计
                        class_name = roi_info['class_name']
                        if class_name not in self.stats['roi_per_class']:
                            self.stats['roi_per_class'][class_name] = 0
                        self.stats['roi_per_class'][class_name] += 1

            return roi_list
        except Exception as e:
            print(f"检测失败 {image_path}: {e}")
            return []

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

    def _save_image_without_roi(self, image_path: Path) -> Optional[str]:
        """
        保存未检测到ROI的图像到专用文件夹

        Args:
            image_path: 原始图像路径

        Returns:
            保存后的图像路径，失败返回None
        """
        try:
            # 生成目标文件路径
            # 保持原始文件名，但可以根据需要修改格式
            if self.save_format in ['jpg', 'jpeg']:
                target_name = f"{image_path.stem}.{self.save_format}"
            else:
                target_name = image_path.name  # 保持原格式

            target_path = self.without_roi_dir / target_name

            # 如果需要转换格式或调整质量
            if self.save_format in ['jpg', 'jpeg'] and image_path.suffix.lower() not in ['.jpg', '.jpeg']:
                # 读取并保存为JPEG格式
                img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
                if img is not None:
                    img_normalized = self._normalize_image(img)
                    cv2.imwrite(str(target_path), img_normalized,
                               [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                else:
                    # 如果读取失败，直接复制
                    shutil.copy2(image_path, target_path)
            else:
                # 直接复制文件
                shutil.copy2(image_path, target_path)

            return str(target_path)

        except Exception as e:
            print(f"保存无ROI图像失败 {image_path}: {e}")
            return None

    def _process_single_image(self, image_path: Path) -> bool:
        """
        处理单张图像

        Args:
            image_path: 图像路径

        Returns:
            是否成功处理
        """
        # 读取图像（保持原始格式）
        img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"警告: 无法读取图像 {image_path}")
            self.stats['failed_images'].append(str(image_path))
            return False

        # 记录原始图像信息
        original_dtype = img.dtype

        # 处理图像维度
        if len(img.shape) == 2:
            # 单通道图像
            img_height, img_width = img.shape[:2]
            img_channels = 1
        else:
            img_height, img_width = img.shape[:2]
            img_channels = img.shape[2] if len(img.shape) > 2 else 1

        # 检测ROI区域
        roi_list = self._detect_roi(str(image_path))

        if not roi_list:
            # 没有检测到ROI，保存到专用文件夹
            self.stats['images_without_roi'] += 1
            self.stats['images_without_roi_list'].append(str(image_path))

            # 保存到withoutROI文件夹
            saved_path = self._save_image_without_roi(image_path)

            # 记录无ROI图像信息
            self.no_roi_images.append({
                'source_image': str(image_path),
                'saved_to': saved_path,
                'image_size': [img_width, img_height],
                'channels': img_channels,
                'original_dtype': str(original_dtype)
            })

            return True

        self.stats['images_with_roi'] += 1
        self.stats['total_roi_found'] += len(roi_list)

        # 处理每个ROI区域
        base_name = image_path.stem
        for roi_idx, roi_info in enumerate(roi_list):
            x1, y1, x2, y2 = roi_info['bbox']

            # 添加padding
            x1, y1, x2, y2 = self._add_padding(
                x1, y1, x2, y2, img_width, img_height
            )

            # 裁剪图像
            cropped_img = img[y1:y2, x1:x2]

            if cropped_img.size == 0:
                print(f"警告: 裁剪区域无效 {image_path} ROI_{roi_idx}")
                continue

            # 归一化裁剪后的图像用于保存
            cropped_img_normalized = self._normalize_image(cropped_img)

            # 生成新文件名（包含类别信息和置信度）
            class_name = roi_info['class_name'].replace(' ', '_')
            confidence = roi_info['confidence']
            new_img_name = f"{base_name}_roi{roi_idx:02d}_{class_name}_conf{confidence:.3f}.{self.save_format}"

            # 保存裁剪后的图像
            output_img_path = self.output_dir / new_img_name

            if self.save_format == 'jpg' or self.save_format == 'jpeg':
                cv2.imwrite(str(output_img_path), cropped_img_normalized,
                           [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            else:
                cv2.imwrite(str(output_img_path), cropped_img_normalized)

            # 保存ROI信息
            self.roi_info.append({
                'source_image': str(image_path),
                'roi_image': str(output_img_path),
                'bbox': [x1, y1, x2, y2],
                'original_bbox': list(roi_info['bbox']),
                'confidence': roi_info['confidence'],
                'class_id': roi_info['class_id'],
                'class_name': roi_info['class_name'],
                'image_size': [img_width, img_height],
                'roi_size': [x2-x1, y2-y1],
                'channels': img_channels,
                'original_dtype': str(original_dtype)
            })

        return True

    def process_images(self, recursive: bool = False):
        """
        处理所有图像

        Args:
            recursive: 是否递归处理子目录
        """
        print(f"\n开始处理图像...")

        # 支持的图像格式
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']

        # 收集所有图像文件
        image_files = []
        for ext in image_extensions:
            if recursive:
                image_files.extend(self.input_dir.rglob(ext))
            else:
                image_files.extend(self.input_dir.glob(ext))

        # 去重并排序
        image_files = sorted(list(set(image_files)))

        if not image_files:
            print(f"警告: 在 {self.input_dir} 中未找到图像文件")
            return

        print(f"找到 {len(image_files)} 张图像")

        # 处理每张图像
        for image_path in tqdm(image_files, desc="处理图像"):
            self.stats['total_images'] += 1
            self._process_single_image(image_path)

        # 保存ROI信息到JSON文件
        self._save_roi_info()

        # 打印统计信息
        self._print_statistics()

    def _save_roi_info(self):
        """保存ROI信息到JSON文件"""
        info_file = self.output_dir / 'roi_info.json'

        # 准备保存的数据
        save_data = {
            'model_path': str(self.model_path),
            'parameters': {
                'roi_conf_threshold': self.roi_conf_threshold,
                'roi_iou_threshold': self.roi_iou_threshold,
                'padding_ratio': self.padding_ratio
            },
            'statistics': self.stats,
            'roi_details': self.roi_info,
            'images_without_roi': self.no_roi_images  # 新增：无ROI图像详情
        }

        # 保存JSON文件
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)

        print(f"\nROI信息已保存到: {info_file}")

    def _print_statistics(self):
        """打印统计信息"""
        print(f"\n{'='*60}")
        print(f"✅ ROI提取完成！")
        print(f"\n📊 统计信息:")
        print(f"  - 处理图像总数: {self.stats['total_images']}")
        print(f"  - 包含ROI的图像: {self.stats['images_with_roi']}")
        print(f"  - 无ROI的图像: {self.stats['images_without_roi']}")
        print(f"  - 检测到的ROI总数: {self.stats['total_roi_found']}")

        if self.stats['images_with_roi'] > 0:
            avg_roi = self.stats['total_roi_found'] / self.stats['images_with_roi']
            print(f"  - 平均每张图像ROI数: {avg_roi:.2f}")

        if self.stats['roi_per_class']:
            print(f"\n📈 各类别ROI数量:")
            for class_name, count in sorted(self.stats['roi_per_class'].items()):
                print(f"    - {class_name}: {count}")

        if self.stats['images_without_roi'] > 0:
            print(f"\n📁 无ROI图像:")
            print(f"    - 数量: {self.stats['images_without_roi']} 张")
            print(f"    - 保存位置: {self.without_roi_dir}")
            # 显示前5个无ROI的图像
            if len(self.stats['images_without_roi_list']) > 0:
                print(f"    - 示例:")
                for img_path in self.stats['images_without_roi_list'][:5]:
                    print(f"      • {Path(img_path).name}")
                if len(self.stats['images_without_roi_list']) > 5:
                    print(f"      ... 还有 {len(self.stats['images_without_roi_list'])-5} 张")

        if self.stats['failed_images']:
            print(f"\n⚠️ 失败的图像 ({len(self.stats['failed_images'])} 个):")
            for img_path in self.stats['failed_images'][:5]:  # 只显示前5个
                print(f"    - {img_path}")
            if len(self.stats['failed_images']) > 5:
                print(f"    ... 还有 {len(self.stats['failed_images'])-5} 个")

        print(f"\n💾 输出目录: {self.output_dir}")
        print(f"   - ROI图像: {self.stats['total_roi_found']} 个")
        print(f"   - 无ROI图像: {self.stats['images_without_roi']} 个 (保存在 ./withoutROI/)")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='使用YOLO模型从图像中提取ROI区域（无标签版本）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基本使用
  python yolo_roi_image_extractor.py --input_dir ./images --output_dir ./roi_images --model_path ./weights/best.pt
  
  # 调整ROI检测阈值
  python yolo_roi_image_extractor.py --input_dir ./images --output_dir ./roi_images --model_path ./weights/best.pt --roi_conf 0.5 --roi_iou 0.7
  
  # 增加ROI区域padding（20%）
  python yolo_roi_image_extractor.py --input_dir ./images --output_dir ./roi_images --model_path ./weights/best.pt --padding 0.2
  
  # 递归处理子文件夹
  python yolo_roi_image_extractor.py --input_dir ./images --output_dir ./roi_images --model_path ./weights/best.pt --recursive
  
  # 保存为PNG格式
  python yolo_roi_image_extractor.py --input_dir ./images --output_dir ./roi_images --model_path ./weights/best.pt --format png

注意：
  - 未检测到ROI的图像会自动保存到输出目录下的 ./withoutROI/ 子文件夹中
  - JSON文件中会记录所有未检测到ROI的图像信息
        """
    )

    parser.add_argument('--input_dir', type=str, required=True,
                       help='输入图像目录')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='输出ROI图像目录')
    parser.add_argument('--model_path', type=str, required=True,
                       help='YOLO模型权重路径（.pt文件）')
    parser.add_argument('--roi_conf', type=float, default=0.25,
                       help='ROI检测置信度阈值 (默认: 0.25)')
    parser.add_argument('--roi_iou', type=float, default=0.45,
                       help='ROI检测IOU阈值 (默认: 0.45)')
    parser.add_argument('--padding', type=float, default=0.1,
                       help='ROI区域padding比例 (默认: 0.1)')
    parser.add_argument('--format', type=str, choices=['jpg', 'jpeg', 'png'],
                       default='jpg',
                       help='输出图像格式 (默认: jpg)')
    parser.add_argument('--quality', type=int, default=95,
                       help='JPEG保存质量 1-100 (默认: 95)')
    parser.add_argument('--recursive', action='store_true',
                       help='递归处理子目录')

    args = parser.parse_args()

    # 创建ROI提取器
    extractor = YOLOROIImageExtractor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        roi_conf_threshold=args.roi_conf,
        roi_iou_threshold=args.roi_iou,
        padding_ratio=args.padding,
        save_format=args.format,
        jpeg_quality=args.quality
    )

    # 处理图像
    extractor.process_images(recursive=args.recursive)


if __name__ == '__main__':
    main()