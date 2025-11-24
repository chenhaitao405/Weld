import os
import sys
import argparse
import shutil
import math
import json
from collections import OrderedDict
from pathlib import Path
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 获取当前脚本所在目录的父目录（dataprocess 目录）
current_dir = os.path.dirname(os.path.abspath(__file__))
dataprocess_dir = os.path.dirname(current_dir)
sys.path.append(dataprocess_dir)

from utils import (
    read_labelme_json,
    save_yolo_labels,
    train_val_split,
    create_dataset_yaml,
    find_image_files
)
from utils.constants import IMAGE_EXTENSIONS


class Labelme2YOLO:
    """LabelMe到YOLO格式转换器 - 完整修复版"""

    def __init__(self, json_dir, to_seg=False, filter_label=None,
                 unify_to_crack=False, output_dir=None, image_dir=None,
                 predefined_label_map=None):
        """
        初始化转换器

        Args:
            json_dir: JSON文件目录
            to_seg: 是否转换为分割格式
            filter_label: 要过滤的标签
            unify_to_crack: 是否统一为crack标签
            output_dir: 输出目录
            image_dir: 图像文件目录（如果为None，则使用json_dir）
            predefined_label_map: 预定义的标签映射字典
        """
        self._json_dir = json_dir
        self._image_dir = image_dir if image_dir else json_dir
        self._to_seg = to_seg
        self._filter_label = filter_label
        self._unify_to_crack = unify_to_crack
        self._predefined_label_map = predefined_label_map

        # 处理标签映射的优先级
        if predefined_label_map:
            # 如果提供了预定义映射，使用它（批处理模式）
            self._label_id_map = OrderedDict(predefined_label_map)
            # 检查是否是统一为crack的映射
            if len(self._label_id_map) == 1 and 'crack' in self._label_id_map:
                self._unify_to_crack = True
                print(f"\n检测到预定义映射只有'crack'，启用unify_to_crack模式")
            print(f"\n使用预定义的标签映射（{len(self._label_id_map)} 个标签）")
        elif unify_to_crack:
            # 如果启用了unify_to_crack，创建单一crack映射
            self._label_id_map = OrderedDict([('crack', 0)])
            print(f"\n启用unify_to_crack，所有标签统一为'crack'")
        else:
            # 自动扫描获取标签映射
            self._label_id_map = self._get_label_id_map(self._json_dir)

        # 设置输出路径
        if output_dir:
            self._save_path_pfx = output_dir
        else:
            suffix = 'YOLODataset_seg' if to_seg else 'YOLODataset'
            self._save_path_pfx = os.path.join(self._json_dir, suffix)

        # 创建输出目录
        Path(self._save_path_pfx).mkdir(parents=True, exist_ok=True)

        print(f"\nLabelme2YOLO converter initialized:")
        print(f"  - Input JSON directory: {json_dir}")
        print(f"  - Input image directory: {self._image_dir}")
        print(f"  - Output directory: {self._save_path_pfx}")
        print(f"  - Mode: {'Segmentation' if to_seg else 'Detection'}")
        if filter_label:
            print(f"  - Filtering label: {filter_label}")
        if self._unify_to_crack:
            print(f"  - All labels unified to 'crack'")
        print(f"  - Label mapping: {dict(self._label_id_map)}")

    def _make_train_val_dir(self):
        """创建训练集和验证集目录"""
        self._label_dir_path = os.path.join(self._save_path_pfx, 'labels/')
        self._image_dir_path = os.path.join(self._save_path_pfx, 'images/')

        for yolo_path in [
            os.path.join(self._label_dir_path, 'train/'),
            os.path.join(self._label_dir_path, 'val/'),
            os.path.join(self._image_dir_path, 'train/'),
            os.path.join(self._image_dir_path, 'val/')
        ]:
            Path(yolo_path).mkdir(parents=True, exist_ok=True)

    def _get_label_id_map(self, json_dir):
        """获取标签ID映射 - 自动扫描模式"""
        # 注意：这个方法只在没有预定义映射且unify_to_crack=False时调用

        label_set = set()
        json_files_processed = 0
        label_statistics = {}

        print("\n正在扫描标签...")

        for file_name in os.listdir(json_dir):
            if file_name.endswith('.json'):
                json_path = os.path.join(json_dir, file_name)
                try:
                    data = read_labelme_json(json_path)
                    json_files_processed += 1

                    for shape in data.get('shapes', []):
                        label = shape.get('label', '').strip()

                        if label and label != self._filter_label:
                            label_set.add(label)
                            label_statistics[label] = label_statistics.get(label, 0) + 1

                except Exception as e:
                    print(f"  警告：读取文件 {file_name} 失败: {e}")
                    continue

        print(f"  - 处理了 {json_files_processed} 个JSON文件")
        print(f"  - 发现了 {len(label_set)} 个不同的标签")

        # 显示标签统计
        if label_statistics:
            print("\n标签出现频次:")
            for label, count in sorted(label_statistics.items(), key=lambda x: x[1], reverse=True):
                print(f"  - {label}: {count} 次")

        # 创建有序的标签ID映射
        return OrderedDict([(label, label_id)
                            for label_id, label in enumerate(sorted(label_set))])

    def _get_yolo_object_list(self, json_data):
        """从JSON数据提取YOLO格式的标注"""
        yolo_obj_list = []

        img_h = json_data.get('imageHeight', 0)
        img_w = json_data.get('imageWidth', 0)

        if img_h <= 0 or img_w <= 0:
            print(f"  警告：图像尺寸无效 (w={img_w}, h={img_h})")
            return yolo_obj_list

        skipped_labels = set()  # 记录跳过的标签
        unified_count = 0  # 统计统一为crack的数量

        for shape in json_data.get('shapes', []):
            original_label = shape.get('label', '').strip()
            label = original_label  # 保存原始标签用于日志

            # 过滤指定标签
            if self._filter_label and label == self._filter_label:
                continue

            # 检查points
            if 'points' not in shape or len(shape['points']) < 2:
                continue

            # 统一标签为crack（如果启用）
            if self._unify_to_crack:
                if label != 'crack':  # 记录统一的数量
                    unified_count += 1
                label = 'crack'

            # 获取标签ID
            if label not in self._label_id_map:
                if not self._unify_to_crack:  # 只在非统一模式下警告
                    skipped_labels.add(original_label)
                continue

            label_id = self._label_id_map[label]

            # 根据形状类型处理
            if shape['shape_type'] == 'circle':
                yolo_obj = self._get_circle_shape_yolo_object(shape, img_h, img_w, label_id)
            elif shape['shape_type'] == 'rectangle':
                yolo_obj = self._get_rectangle_shape_yolo_object(shape, img_h, img_w, label_id)
            else:
                if len(shape['points']) >= 3 or not self._to_seg:
                    yolo_obj = self._get_other_shape_yolo_object(shape, img_h, img_w, label_id)
                else:
                    continue

            if yolo_obj:
                yolo_obj_list.append(yolo_obj)

        # 输出统计信息
        if self._unify_to_crack and unified_count > 0:
            pass  # 静默处理，避免每个文件都输出

        # 如果有跳过的标签，发出警告（只在非统一模式下）
        if skipped_labels and not self._unify_to_crack:
            print(f"  警告：以下标签不在映射中被跳过: {skipped_labels}")

        return yolo_obj_list

    def _get_rectangle_shape_yolo_object(self, shape, img_h, img_w, label_id):
        """处理矩形标注"""
        points = shape['points']
        if len(points) != 2:
            return None

        (x1, y1), (x2, y2) = points
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)

        # 确保顺序正确
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        if self._to_seg:
            # 分割模式：转换为多边形
            retval = [label_id]
            points_norm = [
                [x1 / img_w, y1 / img_h],
                [x2 / img_w, y1 / img_h],
                [x2 / img_w, y2 / img_h],
                [x1 / img_w, y2 / img_h]
            ]
            for p in points_norm:
                retval.extend([round(p[0], 6), round(p[1], 6)])
            return retval
        else:
            # 检测模式：中心点格式
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            xc = x1 + w / 2.0
            yc = y1 + h / 2.0

            xc_n = round(xc / float(img_w), 6)
            yc_n = round(yc / float(img_h), 6)
            w_n = round(w / float(img_w), 6)
            h_n = round(h / float(img_h), 6)

            return [label_id, xc_n, yc_n, w_n, h_n]

    def _get_circle_shape_yolo_object(self, shape, img_h, img_w, label_id):
        """处理圆形标注"""
        obj_center_x, obj_center_y = shape['points'][0]

        radius = math.sqrt(
            (obj_center_x - shape['points'][1][0]) ** 2 +
            (obj_center_y - shape['points'][1][1]) ** 2
        )

        if self._to_seg:
            # 分割模式：将圆形转换为多边形
            retval = [label_id]
            n_points = max(8, int(radius / 10))

            for i in range(n_points):
                angle = 2 * math.pi * i / n_points
                x = obj_center_x + radius * math.cos(angle)
                y = obj_center_y - radius * math.sin(angle)
                retval.extend([round(x / img_w, 6), round(y / img_h, 6)])

            return retval
        else:
            # 检测模式：转换为边界框
            obj_w = 2 * radius
            obj_h = 2 * radius

            yolo_center_x = round(float(obj_center_x / img_w), 6)
            yolo_center_y = round(float(obj_center_y / img_h), 6)
            yolo_w = round(float(obj_w / img_w), 6)
            yolo_h = round(float(obj_h / img_h), 6)

            return [label_id, yolo_center_x, yolo_center_y, yolo_w, yolo_h]

    def _get_other_shape_yolo_object(self, shape, img_h, img_w, label_id):
        """处理其他形状的标注"""
        if self._to_seg:
            # 分割模式：直接使用多边形点
            retval = [label_id]
            for point in shape['points']:
                x_norm = round(float(point[0]) / img_w, 6)
                y_norm = round(float(point[1]) / img_h, 6)
                retval.extend([x_norm, y_norm])
            return retval
        else:
            # 检测模式：计算边界框
            points = shape['points']
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]

            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            obj_w = x_max - x_min
            obj_h = y_max - y_min

            yolo_center_x = round((x_min + obj_w / 2.0) / img_w, 6)
            yolo_center_y = round((y_min + obj_h / 2.0) / img_h, 6)
            yolo_w = round(obj_w / img_w, 6)
            yolo_h = round(obj_h / img_h, 6)

            return [label_id, yolo_center_x, yolo_center_y, yolo_w, yolo_h]

    def _save_yolo_image(self, json_name, image_dir_path, target_dir):
        """保存图像到YOLO数据集"""
        json_name_without_ext = Path(json_name).stem

        # 从指定的图像目录查找对应的图像文件
        src_img_path = None
        for ext in IMAGE_EXTENSIONS:
            potential_path = Path(self._image_dir) / f"{json_name_without_ext}{ext}"
            if potential_path.exists():
                src_img_path = potential_path
                break

        if src_img_path is None:
            print(f"  警告: 未找到图像文件: {json_name_without_ext}")
            return None

        # 复制图像
        dst_img_path = Path(image_dir_path) / target_dir / src_img_path.name
        shutil.copy2(src_img_path, dst_img_path)

        return str(dst_img_path)

    def convert(self, val_size):
        """执行转换"""
        # 获取所有JSON文件
        json_names = [f for f in os.listdir(self._json_dir)
                      if f.endswith('.json')]

        if not json_names:
            print("未找到JSON文件！")
            return

        print(f"\n找到 {len(json_names)} 个JSON文件")

        # 划分训练集和验证集
        train_json_names, val_json_names = train_val_split(json_names, val_size)

        print(f"训练集: {len(train_json_names)} 个文件")
        print(f"验证集: {len(val_json_names)} 个文件")

        # 创建目录结构
        self._make_train_val_dir()

        # 记录处理统计
        stats = {
            'train': {'success': 0, 'failed': 0, 'no_labels': 0},
            'val': {'success': 0, 'failed': 0, 'no_labels': 0}
        }

        # 如果启用了unify_to_crack，记录原始标签
        if self._unify_to_crack:
            original_labels = set()

        # 转换文件
        for target_dir, json_names_list in zip(
                ['train/', 'val/'],
                [train_json_names, val_json_names]
        ):
            split_name = target_dir.replace('/', '')
            print(f"\n处理{split_name}集...")

            for json_name in tqdm(json_names_list):
                json_path = os.path.join(self._json_dir, json_name)

                try:
                    json_data = read_labelme_json(json_path)

                    # 如果启用了unify_to_crack，收集原始标签
                    if self._unify_to_crack:
                        for shape in json_data.get('shapes', []):
                            label = shape.get('label', '').strip()
                            if label:
                                original_labels.add(label)

                    # 保存图像
                    img_path = self._save_yolo_image(
                        json_name,
                        self._image_dir_path, target_dir
                    )

                    if img_path:
                        # 获取YOLO标注
                        yolo_obj_list = self._get_yolo_object_list(json_data)

                        if yolo_obj_list:
                            # 保存标注
                            label_path = os.path.join(
                                self._label_dir_path, target_dir,
                                Path(json_name).stem + '.txt'
                            )
                            save_yolo_labels(
                                yolo_obj_list, label_path,
                                'seg' if self._to_seg else 'det'
                            )
                            stats[split_name]['success'] += 1
                        else:
                            stats[split_name]['no_labels'] += 1
                    else:
                        stats[split_name]['failed'] += 1

                except Exception as e:
                    print(f"\n  错误处理 {json_name}: {e}")
                    stats[split_name]['failed'] += 1

        # 显示处理统计
        print("\n处理统计:")
        for split in ['train', 'val']:
            s = stats[split]
            total = s['success'] + s['failed'] + s['no_labels']
            print(f"  {split}集: 总计 {total} 个文件")
            print(f"    - 成功: {s['success']}")
            print(f"    - 无标注: {s['no_labels']}")
            print(f"    - 失败: {s['failed']}")

        # 如果启用了unify_to_crack，显示原始标签信息
        if self._unify_to_crack and original_labels:
            print(f"\n原始标签（已统一为'crack'）: {sorted(original_labels)}")

        # 生成dataset.yaml
        print('\n生成dataset.yaml文件...')
        self._save_dataset_yaml()

        print(f'\n转换完成！输出目录: {self._save_path_pfx}')

    def _save_dataset_yaml(self):
        """生成dataset.yaml文件"""
        train_path = str(Path(self._image_dir_path) / 'train')
        val_path = str(Path(self._image_dir_path) / 'val')

        # 创建yaml内容
        yaml_content = f"""# Ultralytics YOLO 🚀, AGPL-3.0 license
# Dataset configuration

# Dataset paths
path: {self._save_path_pfx}  # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/val  # val images (relative to 'path')

# Classes
nc: {len(self._label_id_map)}  # number of classes
names: {list(self._label_id_map.keys())}  # class names

# Label ID mapping
label_id_map: {dict(self._label_id_map)}
"""

        # 如果启用了unify_to_crack，添加注释
        if self._unify_to_crack:
            yaml_content += "\n# Note: All labels have been unified to 'crack'\n"

        # 写入文件
        yaml_path = os.path.join(self._save_path_pfx, 'dataset.yaml')
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(yaml_content)

        print(f'类别映射: {dict(self._label_id_map)}')


def main():
    parser = argparse.ArgumentParser(
        description='将LabelMe JSON格式转换为YOLO格式（完整修复版）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基本转换（检测格式）
  python labelme2yolo.py --json_dir ./labelme_data

  # 转换为分割格式
  python labelme2yolo.py --json_dir ./labelme_data --seg

  # 设置验证集比例
  python labelme2yolo.py --json_dir ./labelme_data --val_size 0.3

  # 过滤特定标签
  python labelme2yolo.py --json_dir ./labelme_data --filter_label "焊缝"

  # 统一所有标签为crack（重要功能）
  python labelme2yolo.py --json_dir ./labelme_data --unify_to_crack

  # 指定输出目录
  python labelme2yolo.py --json_dir ./labelme_data --output_dir ./yolo_output

  # 指定图像文件目录（与JSON目录分离）
  python labelme2yolo.py --json_dir ./labelme_data --image_dir ./images

  # 使用预定义的标签映射（JSON格式，用于批处理）
  python labelme2yolo.py --json_dir ./labelme_data --label_map '{"crack": 0, "scratch": 1}'

注意：
  - --unify_to_crack 会将所有标签统一为'crack'，适用于二分类任务
  - --label_map 主要用于批处理多个数据集时保持标签一致性
  - 两个参数不应同时使用
        """
    )

    parser.add_argument('--json_dir', type=str, required=True,
                        help='LabelMe JSON文件目录')
    parser.add_argument('--image_dir', type=str, default=None,
                        help='图像文件目录 (默认: 与json_dir相同)')
    parser.add_argument('--val_size', type=float, default=0.1,
                        help='验证集比例 (默认: 0.1)')
    parser.add_argument('--seg', action='store_true',
                        help='转换为YOLOv5分割格式')
    parser.add_argument('--filter_label', type=str, default=None,
                        help='要过滤的标签名称')
    parser.add_argument('--unify_to_crack', action='store_true',
                        help='将所有标签统一为"crack"（用于二分类）')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录 (默认: json_dir/YOLODataset[_seg])')
    parser.add_argument('--label_map', type=str, default=None,
                        help='预定义的标签映射（JSON格式字符串，批处理用）')

    args = parser.parse_args()

    # 检查参数冲突
    if args.label_map and args.unify_to_crack:
        print("警告：同时指定了 --label_map 和 --unify_to_crack")
        print("      将优先使用 --label_map")

    # 解析标签映射
    predefined_label_map = None
    if args.label_map:
        try:
            predefined_label_map = json.loads(args.label_map)
            print(f"使用预定义标签映射: {predefined_label_map}")
        except json.JSONDecodeError as e:
            print(f"错误：无法解析标签映射JSON: {e}")
            sys.exit(1)

    # 创建转换器
    converter = Labelme2YOLO(
        args.json_dir,
        to_seg=args.seg,
        filter_label=args.filter_label,
        unify_to_crack=args.unify_to_crack,
        output_dir=args.output_dir,
        image_dir=args.image_dir,
        predefined_label_map=predefined_label_map
    )

    # 执行转换
    converter.convert(val_size=args.val_size)


if __name__ == '__main__':
    main()