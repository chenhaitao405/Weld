"""
脚本名称: seg2det.py
功能概述: YOLO数据集格式转换工具（分割→检测/分类）
详细说明:
    - 输入格式: YOLO分割格式数据集
    - 处理流程: 读取多边形标注 → 计算边界框 → 转换为检测格式或分类格式
    - 输出格式: YOLO检测格式或分类格式数据集
依赖模块: utils.label_processing, utils.dataset_management
使用示例:
    # 转换为检测格式
    python seg2det.py --input_dir ./seg_dataset --output_dir ./det_dataset --mode det

    # 转换为分类格式
    python seg2det.py --input_dir ./seg_dataset --output_dir ./cls_dataset --mode cls

    # 转换为检测格式但不复制图像
    python seg2det.py --input_dir ./seg_dataset --output_dir ./det_dataset --mode det --no_copy_images
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import (
    read_yolo_labels,
    save_yolo_labels,
    find_image_files,
    create_directory_structure,
    read_dataset_yaml,
    update_dataset_yaml
)
from utils.constants import IMAGE_EXTENSIONS


class YOLOFormatConverter:
    """YOLO格式转换器"""

    def __init__(self, input_dir: str, output_dir: str, mode: str = 'det'):
        """
        初始化转换器

        Args:
            input_dir: 输入数据集目录
            output_dir: 输出数据集目录
            mode: 转换模式 ('det' 或 'cls')
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.mode = mode

        # 验证输入目录
        if not self.input_dir.exists():
            raise ValueError(f"输入目录不存在: {input_dir}")

        # 统计信息
        self.total_converted = 0
        self.total_with_labels = 0
        self.total_without_labels = 0
        self.class_distribution = {}

        print(f"YOLO格式转换器初始化:")
        print(f"  - 输入目录: {input_dir}")
        print(f"  - 输出目录: {output_dir}")
        print(f"  - 转换模式: {mode}")

    def seg_to_det_line(self, seg_line: list) -> list:
        """
        将一行分割标注转换为检测标注

        Args:
            seg_line: [class_id, x1, y1, x2, y2, ...] 多边形标注

        Returns:
            [class_id, x_center, y_center, width, height] 检测框标注
        """
        if len(seg_line) < 7:  # 至少需要class_id + 3个点
            return None

        class_id = seg_line[0]
        points = seg_line[1:]

        # 提取x和y坐标
        x_coords = []
        y_coords = []
        for i in range(0, len(points), 2):
            if i + 1 < len(points):
                x_coords.append(points[i])
                y_coords.append(points[i + 1])

        if not x_coords or not y_coords:
            return None

        # 计算边界框
        x_min = min(x_coords)
        x_max = max(x_coords)
        y_min = min(y_coords)
        y_max = max(y_coords)

        # 计算中心点和宽高
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min

        # 确保值在[0, 1]范围内
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width = max(0, min(1, width))
        height = max(0, min(1, height))

        return [class_id, x_center, y_center, width, height]

    def get_primary_class(self, labels: list) -> int:
        """
        获取主要类别（出现次数最多的类别）

        Args:
            labels: 标签列表

        Returns:
            主要类别ID，如果没有标签返回-1
        """
        if not labels:
            return -1

        # 统计每个类别出现的次数
        class_counts = {}
        for label in labels:
            if len(label) > 0:
                class_id = int(label[0])
                class_counts[class_id] = class_counts.get(class_id, 0) + 1

        if not class_counts:
            return -1

        # 返回出现次数最多的类别
        return max(class_counts, key=class_counts.get)

    def convert_to_det(self, copy_images: bool = True):
        """转换为检测格式"""
        print("开始转换为检测格式...")

        # 创建输出目录结构
        create_directory_structure(self.output_dir)

        # 处理图像
        if copy_images:
            print("复制图像文件...")
            self._copy_images()

        # 处理标签
        print("转换标签文件...")
        self._convert_labels_to_det()

        # 复制并更新dataset.yaml
        self._copy_and_update_yaml()

        # 打印统计信息
        self._print_statistics()

    def convert_to_cls(self):
        """转换为分类格式"""
        print("开始转换为分类格式...")

        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 查找输入目录结构
        input_images_dir = self.input_dir / 'images'
        input_labels_dir = self.input_dir / 'labels'

        if not input_images_dir.exists():
            raise ValueError(f"未找到images目录: {input_images_dir}")

        # 获取所有split
        splits = [d.name for d in input_images_dir.iterdir() if d.is_dir()]

        print(f"找到splits: {splits}")

        # 处理每个split
        for split in splits:
            print(f"\n处理{split}集...")
            self._process_split_to_cls(split)

        # 打印统计信息
        self._print_statistics()

    def _copy_images(self):
        """复制图像目录"""
        input_images_dir = self.input_dir / 'images'
        output_images_dir = self.output_dir / 'images'

        if input_images_dir.exists():
            shutil.copytree(input_images_dir, output_images_dir, dirs_exist_ok=True)
            print("图像复制完成")
        else:
            print(f"警告: 未找到图像目录 {input_images_dir}")

    def _convert_labels_to_det(self):
        """转换标签为检测格式"""
        input_labels_dir = self.input_dir / 'labels'
        output_labels_dir = self.output_dir / 'labels'

        if not input_labels_dir.exists():
            print(f"警告: 未找到标签目录 {input_labels_dir}")
            return

        # 获取所有split
        splits = [d.name for d in input_labels_dir.iterdir() if d.is_dir()]

        for split in splits:
            input_split_dir = input_labels_dir / split
            output_split_dir = output_labels_dir / split
            output_split_dir.mkdir(parents=True, exist_ok=True)

            # 获取所有标签文件
            txt_files = list(input_split_dir.glob('*.txt'))

            print(f"处理{split}集: {len(txt_files)}个文件")

            for txt_file in tqdm(txt_files, desc=f"转换{split}"):
                # 读取分割标签
                seg_labels = read_yolo_labels(str(txt_file), mode='seg')

                # 转换为检测标签
                det_labels = []
                for seg_label in seg_labels:
                    det_label = self.seg_to_det_line(seg_label)
                    if det_label:
                        det_labels.append(det_label)

                # 保存检测标签
                output_label_path = output_split_dir / txt_file.name
                save_yolo_labels(det_labels, str(output_label_path), mode='det')

                # 更新统计
                self.total_converted += 1
                if det_labels:
                    self.total_with_labels += 1
                else:
                    self.total_without_labels += 1

    def _process_split_to_cls(self, split: str):
        """处理单个split转换为分类格式"""
        split_images_dir = self.input_dir / 'images' / split
        split_labels_dir = self.input_dir / 'labels' / split
        split_output_dir = self.output_dir / split

        # 获取所有图像文件
        image_files = find_image_files(str(split_images_dir))

        print(f"  找到{len(image_files)}个图像")

        for image_file in tqdm(image_files, desc=f"处理{split}"):
            # 查找对应的标签文件
            label_file = split_labels_dir / f"{image_file.stem}.txt"

            # 判断图像属于哪个类别
            class_folder = "none"
            if label_file.exists():
                labels = read_yolo_labels(str(label_file), mode='seg')
                primary_class = self.get_primary_class(labels)

                if primary_class >= 0:
                    class_folder = f"class_{primary_class}"
                    self.total_with_labels += 1

                    # 更新类别分布统计
                    if split not in self.class_distribution:
                        self.class_distribution[split] = {}
                    self.class_distribution[split][class_folder] = \
                        self.class_distribution[split].get(class_folder, 0) + 1
                else:
                    self.total_without_labels += 1
            else:
                self.total_without_labels += 1

            # 创建目标文件夹并复制图像
            target_dir = split_output_dir / class_folder
            target_dir.mkdir(parents=True, exist_ok=True)

            target_path = target_dir / image_file.name
            shutil.copy2(image_file, target_path)

            self.total_converted += 1

    def _copy_and_update_yaml(self):
        """复制并更新dataset.yaml文件"""
        input_yaml = self.input_dir / 'dataset.yaml'
        output_yaml = self.output_dir / 'dataset.yaml'

        if input_yaml.exists():
            # 读取原始yaml
            yaml_data = read_dataset_yaml(str(input_yaml))

            # 更新路径
            yaml_data['train'] = str(self.output_dir / 'images' / 'train')
            yaml_data['val'] = str(self.output_dir / 'images' / 'val')

            # 添加转换信息
            yaml_data['conversion_info'] = {
                'source_format': 'segmentation',
                'target_format': 'detection',
                'converter': 'seg2det.py'
            }

            # 保存更新后的yaml
            update_dataset_yaml(str(output_yaml), yaml_data)

            print(f"dataset.yaml已保存到: {output_yaml}")
        else:
            print(f"警告: 未找到dataset.yaml文件")

    def _print_statistics(self):
        """打印统计信息"""
        print(f"\n{'=' * 50}")
        print(f"✅ 转换完成！")
        print(f"📊 统计信息:")
        print(f"  - 总文件数: {self.total_converted}")
        print(f"  - 有标签文件: {self.total_with_labels}")
        print(f"  - 无标签文件: {self.total_without_labels}")
        print(f"  - 输出目录: {self.output_dir}")

        if self.class_distribution:
            print(f"\n📈 类别分布:")
            for split, classes in self.class_distribution.items():
                print(f"  {split}:")
                for class_name, count in sorted(classes.items()):
                    print(f"    - {class_name}: {count}个图像")


def main():
    parser = argparse.ArgumentParser(
        description='YOLO数据集格式转换工具（分割/检测/分类）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 转换为检测格式
  python seg2det.py --input_dir ./seg_dataset --output_dir ./det_dataset --mode det

  # 转换为分类格式
  python seg2det.py --input_dir ./seg_dataset --output_dir ./cls_dataset --mode cls

  # 转换为检测格式但不复制图像（节省空间）
  python seg2det.py --input_dir ./seg_dataset --output_dir ./det_dataset --mode det --no_copy_images
        """
    )

    parser.add_argument('--input_dir', type=str, required=True,
                        help='输入数据集目录')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出数据集目录')
    parser.add_argument('--mode', type=str, choices=['det', 'cls'], default='det',
                        help='转换模式: "det"(检测) 或 "cls"(分类) (默认: det)')
    parser.add_argument('--no_copy_images', action='store_true',
                        help='不复制图像到输出目录 (仅对det模式有效)')

    args = parser.parse_args()

    # 创建转换器
    converter = YOLOFormatConverter(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        mode=args.mode
    )

    # 根据模式执行转换
    if args.mode == 'cls':
        converter.convert_to_cls()
    else:  # det mode
        converter.convert_to_det(copy_images=not args.no_copy_images)


if __name__ == "__main__":
    main()