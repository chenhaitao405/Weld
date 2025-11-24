"""
数据集管理工具模块
提供数据集创建、平衡、分割等功能
"""

import os
import random
import shutil
import yaml
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from sklearn.model_selection import train_test_split


def create_directory_structure(base_dir: str, include_test: bool = False):
    """
    创建标准的数据集目录结构

    Args:
        base_dir: 基础目录
        include_test: 是否包含test集
    """
    base_path = Path(base_dir)

    dirs = [
        base_path / 'images' / 'train',
        base_path / 'images' / 'val',
        base_path / 'labels' / 'train',
        base_path / 'labels' / 'val'
    ]

    if include_test:
        dirs.extend([
            base_path / 'images' / 'test',
            base_path / 'labels' / 'test'
        ])

    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)


def find_image_files(directory: str, extensions: List[str] = None) -> List[Path]:
    """
    查找目录中的所有图像文件

    Args:
        directory: 目录路径
        extensions: 图像扩展名列表

    Returns:
        图像文件路径列表
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']

    dir_path = Path(directory)
    image_files = []

    for ext in extensions:
        image_files.extend(dir_path.glob(f'*{ext}'))
        image_files.extend(dir_path.glob(f'*{ext.upper()}'))

    return sorted(image_files)


def find_label_files(directory: str, extension: str = '.txt') -> List[Path]:
    """
    查找目录中的所有标签文件

    Args:
        directory: 目录路径
        extension: 标签文件扩展名

    Returns:
        标签文件路径列表
    """
    dir_path = Path(directory)
    return sorted(dir_path.glob(f'*{extension}'))


def match_images_labels(image_dir: str, label_dir: str,
                        image_ext: List[str] = None,
                        label_ext: str = '.txt') -> List[Tuple[Path, Path]]:
    """
    匹配图像和标签文件

    Args:
        image_dir: 图像目录
        label_dir: 标签目录
        image_ext: 图像扩展名列表
        label_ext: 标签扩展名

    Returns:
        匹配的(图像路径, 标签路径)列表
    """
    image_files = find_image_files(image_dir, image_ext)
    matched_pairs = []

    label_path = Path(label_dir)
    for image_file in image_files:
        label_file = label_path / f"{image_file.stem}{label_ext}"
        if label_file.exists():
            matched_pairs.append((image_file, label_file))

    return matched_pairs


def train_val_split(data_list: List, val_size: float = 0.2,
                    random_seed: int = 42) -> Tuple[List, List]:
    """
    划分训练集和验证集

    Args:
        data_list: 数据列表
        val_size: 验证集比例
        random_seed: 随机种子

    Returns:
        (train_list, val_list)
    """
    train_list, val_list = train_test_split(
        data_list, test_size=val_size, random_state=random_seed
    )
    return train_list, val_list


def balance_dataset(label_dir: str, image_dir: str,
                    target_ratio: float = 1.0,
                    label_ext: str = '.txt',
                    image_ext: str = '.jpg') -> Dict:
    """
    平衡数据集（调整有标签和无标签样本的比例）

    Args:
        label_dir: 标签目录
        image_dir: 图像目录
        target_ratio: 目标比例（无标签/有标签）
        label_ext: 标签文件扩展名
        image_ext: 图像文件扩展名

    Returns:
        统计信息字典
    """
    label_path = Path(label_dir)
    image_path = Path(image_dir)

    # 统计有标签和无标签的样本
    labeled_samples = []
    unlabeled_samples = []

    for label_file in label_path.rglob(f'*{label_ext}'):
        with open(label_file, 'r') as f:
            content = f.read().strip()

        relative_path = label_file.relative_to(label_path)
        image_file = image_path / relative_path.with_suffix(image_ext)

        if content:  # 有标签
            labeled_samples.append((label_file, image_file))
        else:  # 无标签
            unlabeled_samples.append((label_file, image_file))

    # 计算需要保留的无标签样本数量
    target_unlabeled = int(len(labeled_samples) * target_ratio)

    # 平衡数据集
    removed_count = 0
    if target_unlabeled < len(unlabeled_samples):
        random.seed(42)
        samples_to_keep = random.sample(unlabeled_samples, target_unlabeled)
        samples_to_remove = [s for s in unlabeled_samples if s not in samples_to_keep]

        for label_file, image_file in samples_to_remove:
            if label_file.exists():
                label_file.unlink()
            if image_file.exists():
                image_file.unlink()
            removed_count += 1

    return {
        'labeled_count': len(labeled_samples),
        'unlabeled_count': len(unlabeled_samples),
        'kept_unlabeled': min(target_unlabeled, len(unlabeled_samples)),
        'removed_count': removed_count
    }


def create_dataset_yaml(output_path: str,
                        train_path: str,
                        val_path: str,
                        num_classes: int,
                        class_names: List[str],
                        test_path: str = None,
                        **kwargs):
    """
    创建YOLO格式的dataset.yaml文件

    Args:
        output_path: 输出路径
        train_path: 训练集路径
        val_path: 验证集路径
        num_classes: 类别数量
        class_names: 类别名称列表
        test_path: 测试集路径（可选）
        **kwargs: 其他配置参数
    """
    yaml_content = {
        'train': str(Path(train_path).absolute()),
        'val': str(Path(val_path).absolute()),
        'nc': num_classes,
        'names': class_names
    }

    if test_path:
        yaml_content['test'] = str(Path(test_path).absolute())

    # 添加其他配置参数
    yaml_content.update(kwargs)

    # 保存YAML文件
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)


import yaml
from typing import Dict

def update_dataset_yaml(yaml_path: str, updates: Dict):
    """
    更新或创建dataset.yaml文件

    Args:
        yaml_path: YAML文件路径
        updates: 要更新的字段字典
    """
    # 尝试读取现有YAML，如果不存在则创建新字典
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            yaml_content = yaml.safe_load(f) or {}  # 处理空文件情况
    except FileNotFoundError:
        # 文件不存在时初始化空字典
        yaml_content = {}

    # 更新字段（新字段会覆盖旧字段，不存在的字段会新增）
    yaml_content.update(updates)

    # 确保目录存在
    import os
    os.makedirs(os.path.dirname(yaml_path), exist_ok=True)

    # 保存更新后的YAML
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(
            yaml_content,
            f,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False  # 保持字段顺序
        )



def read_dataset_yaml(yaml_path: str) -> Dict:
    """
    读取dataset.yaml文件

    Args:
        yaml_path: YAML文件路径

    Returns:
        配置字典
    """
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)