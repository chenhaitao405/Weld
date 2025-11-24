
# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import sys
import json
from typing import List, Dict, Set
from collections import OrderedDict
from pathlib import Path
import platform
from tqdm import tqdm
import shutil

# ========================= 配置区域 =========================
# 根据操作系统自动选择路径
if platform.system() == "Windows":
    BASE_PATH = r"C:\Users\CHT\Desktop\datasets1117\labeled"
    JSON_BASE_PATH = r"C:\Users\CHT\Desktop\datasets1117\adjust"
elif platform.system() == "Linux":
    BASE_PATH = "/home/lenovo/code/CHT/datasets/Xray/self/1120/labeled"
    JSON_BASE_PATH = "/home/lenovo/code/CHT/datasets/Xray/self/1120/labeled"  # 修复引号缺失问题
else:
    # 其他系统（如macOS）可根据需要添加配置，这里抛出异常提醒
    raise EnvironmentError(
        f"不支持的操作系统：{platform.system()}\n"
        "请在配置区域添加对应系统的路径配置"
    )

DATASETS = [
    "D1",
    "D2",
    "D3",
    "D4",
    "img20250608",
    "img20250609"
]
OUTPUT_BASE_DIR = "unifyCrack"
OUTPUT_CONFIG = {
    "yolo_dir": os.path.join(BASE_PATH, OUTPUT_BASE_DIR,"yolo"),
    "roi_dir": os.path.join(BASE_PATH, OUTPUT_BASE_DIR,"convert"),
    "patch_dir": os.path.join(BASE_PATH,OUTPUT_BASE_DIR, "patch")
}
FIXED_PARAMS = {
    "labelme2yolo": {
        "seg": True,
        "unify_to_crack": True,  # 如果为True，所有标签都会被统一为crack
        "script_path": "convert/labelme2yolo.py"
    },
    "yolo_roi_extractor": {
        "model_path": "/home/lenovo/code/CHT/detect/ultralytics-main/runs/detect/11m_pretrain/weights/best.pt",
        "roi_conf": 0.25,
        "roi_iou": 0.45,
        "padding": 0.1,
        "mode": "seg",
        "script_path": "convert/pj/yolo_roi_extractor.py"
    },
    "patchandenhance": {
        "overlap": 0.7,
        "enhance_mode": "windowing",
        "no_slice":True,
        "window_size": [640, 640],
        "label_mode": "seg",
        "script_path": "convert/pj/patchandenhance.py"
    }
}

# ===========================================================================

def collect_all_labels(datasets: List[str], json_base_path: str,
                       unify_to_crack: bool = False) -> OrderedDict:
    """
    收集所有数据集的标签，建立统一的标签映射
    """
    # 如果启用了unify_to_crack，直接返回crack映射
    if unify_to_crack:
        print("\n📊 启用了 unify_to_crack，所有标签将统一为 'crack'")
        label_map = OrderedDict([('crack', 0)])
        print(f"📋 统一标签映射：{dict(label_map)}")
        return label_map

    print("\n📊 收集所有数据集的标签...")
    all_labels = set()
    dataset_labels = {}

    for dataset in datasets:
        json_dir = os.path.join(json_base_path, dataset, "label")
        if not os.path.exists(json_dir):
            print(f"  ⚠️ 跳过 {dataset}：标注目录不存在 {json_dir}")
            continue

        dataset_labels[dataset] = set()

        # 扫描该数据集的所有JSON文件
        for json_file in os.listdir(json_dir):
            if not json_file.endswith('.json'):
                continue

            json_path = os.path.join(json_dir, json_file)
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                for shape in data.get('shapes', []):
                    label = shape.get('label', '').strip()
                    if label:
                        dataset_labels[dataset].add(label)
                        all_labels.add(label)

            except Exception as e:
                print(f"  ⚠️ 读取文件失败 {json_file}: {e}")

        if dataset_labels[dataset]:
            print(f"  ✓ {dataset}: 发现 {len(dataset_labels[dataset])} 个标签")

    # 创建统一的标签映射
    sorted_labels = sorted(all_labels)
    label_map = OrderedDict([(label, idx) for idx, label in enumerate(sorted_labels)])

    print(f"\n📋 统一标签映射（共 {len(label_map)} 个标签）：")
    for label, idx in label_map.items():
        # 找出哪些数据集包含这个标签
        datasets_with_label = [d for d, labels in dataset_labels.items() if label in labels]
        print(f"  {idx}: {label} (出现在: {', '.join(datasets_with_label)})")

    return label_map


def create_dataset_yaml(output_dir: str, label_map: OrderedDict):
    """创建统一的dataset.yaml文件"""
    yaml_path = os.path.join(output_dir, "dataset.yaml")

    content = f"""# Ultralytics YOLO 🚀, AGPL-3.0 license
# 统一数据集配置文件

# 数据集路径
path: {output_dir}  # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/val  # val images (relative to 'path')

# 类别
nc: {len(label_map)}  # number of classes
names: {list(label_map.keys())}  # class names

# 标签ID映射
label_id_map: {dict(label_map)}
"""

    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"\n✅ 创建统一的 dataset.yaml: {yaml_path}")


def process_single_json(json_path: str, image_dir: str, label_map: OrderedDict,
                        unify_to_crack: bool, to_seg: bool) -> tuple:
    """
    处理单个JSON文件，返回YOLO格式的标注

    Returns:
        (yolo_objects, image_path, img_width, img_height)
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
    except Exception as e:
        print(f"  错误：无法读取 {json_path}: {e}")
        return None, None, 0, 0

    img_h = json_data.get('imageHeight', 0)
    img_w = json_data.get('imageWidth', 0)

    if img_h <= 0 or img_w <= 0:
        return None, None, 0, 0

    # 查找对应的图像文件
    json_name = Path(json_path).stem
    image_path = None
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
        potential_path = Path(image_dir) / f"{json_name}{ext}"
        if potential_path.exists():
            image_path = str(potential_path)
            break

    if not image_path:
        return None, None, img_w, img_h

    # 提取标注
    yolo_objects = []
    for shape in json_data.get('shapes', []):
        label = shape.get('label', '').strip()

        if not label or 'points' not in shape or len(shape['points']) < 2:
            continue

        # 统一标签为crack（如果启用）
        if unify_to_crack:
            label = 'crack'

        # 获取标签ID
        if label not in label_map:
            continue

        label_id = label_map[label]

        # 处理不同形状类型
        points = shape['points']

        if to_seg:
            # 分割模式：保存多边形点
            yolo_obj = [label_id]
            for point in points:
                x_norm = round(float(point[0]) / img_w, 6)
                y_norm = round(float(point[1]) / img_h, 6)
                yolo_obj.extend([x_norm, y_norm])
        else:
            # 检测模式：转换为边界框
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]

            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            obj_w = x_max - x_min
            obj_h = y_max - y_min

            xc = (x_min + x_max) / 2.0
            yc = (y_min + y_max) / 2.0

            yolo_obj = [
                label_id,
                round(xc / img_w, 6),
                round(yc / img_h, 6),
                round(obj_w / img_w, 6),
                round(obj_h / img_h, 6)
            ]

        yolo_objects.append(yolo_obj)

    return yolo_objects, image_path, img_w, img_h


def process_all_datasets_directly(datasets: List[str], base_path: str,
                                  json_base_path: str, output_dir: str,
                                  label_map: OrderedDict, params: dict):
    """
    直接处理所有数据集到主目录，使用统一的标签映射
    """
    # 创建输出目录结构
    os.makedirs(os.path.join(output_dir, "images/train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images/val"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels/train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels/val"), exist_ok=True)

    # 先创建统一的dataset.yaml
    create_dataset_yaml(output_dir, label_map)

    # 收集所有要处理的文件
    all_files = []

    for dataset in datasets:
        image_dir = os.path.join(base_path, dataset)
        json_dir = os.path.join(json_base_path, dataset, "label")

        if not os.path.exists(image_dir) or not os.path.exists(json_dir):
            print(f"⚠️ 跳过 {dataset}：路径不存在")
            continue

        # 获取该数据集的所有JSON文件
        json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]

        for json_file in json_files:
            all_files.append({
                'dataset': dataset,
                'json_path': os.path.join(json_dir, json_file),
                'image_dir': image_dir,
                'json_name': Path(json_file).stem
            })

    print(f"\n📝 处理 {len(all_files)} 个标注文件...")

    # 随机划分训练集和验证集
    import random
    random.shuffle(all_files)

    val_size = params.get('val_size', 0.1)
    val_count = int(len(all_files) * val_size)

    val_files = all_files[:val_count]
    train_files = all_files[val_count:]

    print(f"  训练集: {len(train_files)} 个文件")
    print(f"  验证集: {len(val_files)} 个文件")

    # 处理训练集和验证集
    for split_name, file_list in [('train', train_files), ('val', val_files)]:
        print(f"\n处理 {split_name} 集...")

        success_count = 0
        no_label_count = 0
        fail_count = 0

        for file_info in tqdm(file_list, desc=f"处理{split_name}"):
            dataset = file_info['dataset']
            json_path = file_info['json_path']
            image_dir = file_info['image_dir']
            json_name = file_info['json_name']

            # 处理JSON获取标注
            yolo_objects, image_path, img_w, img_h = process_single_json(
                json_path, image_dir, label_map,
                params.get('unify_to_crack', False),
                params.get('seg', False)
            )

            if not image_path:
                fail_count += 1
                continue

            # 复制图像（添加数据集前缀）
            src_image = Path(image_path)
            dst_image_name = f"{dataset}_{src_image.name}"
            dst_image_path = Path(output_dir) / "images" / split_name / dst_image_name
            shutil.copy2(src_image, dst_image_path)

            # 保存标注文件
            if yolo_objects:
                label_name = f"{dataset}_{json_name}.txt"
                label_path = Path(output_dir) / "labels" / split_name / label_name

                with open(label_path, 'w') as f:
                    for obj in yolo_objects:
                        line = ' '.join(map(str, obj))
                        f.write(line + '\n')

                success_count += 1
            else:
                no_label_count += 1

        print(f"  {split_name}集统计: 成功 {success_count}, 无标注 {no_label_count}, 失败 {fail_count}")

    print(f"\n✅ 所有数据集处理完成！")


def run_command(command: List[str], step_name: str):
    """执行命令"""
    print(f"\n{'=' * 80}")
    print(f"📌 正在执行【{step_name}】")
    print(f"命令：{' '.join(command)}")
    print(f"{'=' * 80}")

    try:
        subprocess.run(
            command,
            check=True,
            stdout=None,
            stderr=None,
            text=True,
            env=os.environ
        )
        print(f"\n✅ 【{step_name}】执行成功！")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 【{step_name}】执行失败！错误码：{e.returncode}")
        sys.exit(1)


def get_abs_path(relative_path: str) -> str:
    """获取脚本所在目录的绝对路径"""
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_script_dir, relative_path))


def process_roi_extractor(input_dir: str, output_dir: str):
    """执行 ROI 提取"""
    script_path = get_abs_path(FIXED_PARAMS["yolo_roi_extractor"]["script_path"])
    command = [
        sys.executable, script_path,
        "--input_dir", input_dir,
        "--output_dir", output_dir,
        "--model_path", FIXED_PARAMS["yolo_roi_extractor"]["model_path"],
        "--roi_conf", str(FIXED_PARAMS["yolo_roi_extractor"]["roi_conf"]),
        "--roi_iou", str(FIXED_PARAMS["yolo_roi_extractor"]["roi_iou"]),
        "--padding", str(FIXED_PARAMS["yolo_roi_extractor"]["padding"]),
        "--mode", FIXED_PARAMS["yolo_roi_extractor"]["mode"]
    ]

    run_command(command, "YOLO ROI提取")


def process_patch_enhance(input_dir: str, output_dir: str):
    """执行图像裁剪增强"""
    script_path = get_abs_path(FIXED_PARAMS["patchandenhance"]["script_path"])
    command = [
        sys.executable, script_path,
        "--input_dir", input_dir,
        "--output_dir", output_dir,
        "--overlap", str(FIXED_PARAMS["patchandenhance"]["overlap"]),
        "--enhance_mode", FIXED_PARAMS["patchandenhance"]["enhance_mode"],
        "--window_size",
        str(FIXED_PARAMS["patchandenhance"]["window_size"][0]),
        str(FIXED_PARAMS["patchandenhance"]["window_size"][1]),
        "--label_mode", FIXED_PARAMS["patchandenhance"]["label_mode"]
    ]

    if FIXED_PARAMS["patchandenhance"]["no_slice"]:
        command.append("--no_slice")

    run_command(command, "图像裁剪与增强")


def main():
    print("🚀 数据处理流水线启动（简化版）！")
    print(f"基础路径：{BASE_PATH}")
    print(f"待处理数据集：{DATASETS}")

    # 获取 unify_to_crack 设置
    unify_to_crack = FIXED_PARAMS["labelme2yolo"].get("unify_to_crack", False)
    if unify_to_crack:
        print("\n⚠️ 注意：已启用 unify_to_crack，所有标签将被统一为 'crack'")

    # 第一步：收集所有标签，建立统一映射
    label_map = collect_all_labels(DATASETS, JSON_BASE_PATH, unify_to_crack)

    if not label_map:
        print("❌ 错误：未找到任何标签！")
        sys.exit(1)

    # 第二步：直接处理所有数据集到主目录
    print("\n" + "=" * 100)
    print("📝 批量处理 Labelme 数据（使用统一标签映射）")
    print("=" * 100)

    process_all_datasets_directly(
        DATASETS,
        BASE_PATH,
        JSON_BASE_PATH,
        OUTPUT_CONFIG["yolo_dir"],
        label_map,
        FIXED_PARAMS["labelme2yolo"]
    )

    # 第三步：执行 ROI 提取
    print("\n" + "=" * 100)
    print("📝 执行 YOLO ROI 区域提取")
    print("=" * 100)

    if not os.path.exists(OUTPUT_CONFIG["yolo_dir"]):
        print(f"❌ 错误：YOLO 数据集目录不存在 {OUTPUT_CONFIG['yolo_dir']}")
        sys.exit(1)

    process_roi_extractor(OUTPUT_CONFIG["yolo_dir"], OUTPUT_CONFIG["roi_dir"])

    # 第四步：执行图像裁剪增强
    print("\n" + "=" * 100)
    print("📝 执行图像裁剪与增强")
    print("=" * 100)

    if not os.path.exists(OUTPUT_CONFIG["roi_dir"]):
        print(f"❌ 错误：ROI 提取目录不存在 {OUTPUT_CONFIG['roi_dir']}")
        sys.exit(1)

    process_patch_enhance(OUTPUT_CONFIG["roi_dir"], OUTPUT_CONFIG["patch_dir"])

    print("\n" + "🎉" * 50)
    print("🎉 所有数据处理步骤执行完成！")
    print(f"📁 最终结果保存目录：{OUTPUT_CONFIG['patch_dir']}")
    print("🎉" * 50)


if __name__ == "__main__":
    main()