
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
    MODEL_PATH = "E:\CODE\weldDataProcess\model\weldDetect.pt"
elif platform.system() == "Linux":
    BASE_PATH = "/home/lenovo/code/CHT/datasets/Xray/self/1120/labeled"
    JSON_BASE_PATH = "/home/lenovo/code/CHT/datasets/Xray/self/1120/labeled"  # 修复引号缺失问题
    MODEL_PATH = "/home/lenovo/code/CHT/detect/ultralytics-main/runs/detect/11m_pretrain/weights/best.pt"
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
OUTPUT_BASE_DIR = "processed"
OUTPUT_CONFIG = {
    "yolo_dir": os.path.join(BASE_PATH, OUTPUT_BASE_DIR,"yolo"),
    "roi_dir": os.path.join(BASE_PATH, OUTPUT_BASE_DIR,"convert"),
    "patch_dir": os.path.join(BASE_PATH,OUTPUT_BASE_DIR, "patch"),
    "cls_dir": os.path.join(BASE_PATH, OUTPUT_BASE_DIR, "cls")
}
FIXED_PARAMS = {
    "labelme2yolo": {
        "seg": True,
        "unify_to_crack": True,  # 如果为True，所有标签都会被统一为crack
        "script_path": "convert/labelme2yolo.py"
    },
    "yolo_roi_extractor": {
        "model_path": MODEL_PATH,
        "roi_conf": 0.25,
        "roi_iou": 0.45,
        "padding": 0.1,
        "mode": "seg",
        "script_path": "convert/pj/yolo_roi_extractor.py"
    },
    "patchandenhance": {
        "overlap": 0.7,
        "enhance_mode": "windowing",
        "no_slice":False,
        "window_size": [640, 640],
        "label_mode": "seg",
        "script_path": "convert/pj/patchandenhance.py"
    },
    "seg2det":{
        "mode": "cls",
        "script_path": "convert/pj/seg2det.py"
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


def process_labelme2yolo_unified(datasets: List[str], base_path: str,
                                  json_base_path: str, output_dir: str,
                                  ):
    """
    直接处理所有数据集到主目录，使用统一的标签映射
    """

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
    script_path = get_abs_path(FIXED_PARAMS["labelme2yolo"]["script_path"])

    for dataset in datasets:
        image_dir = os.path.join(base_path, dataset)
        json_dir = os.path.join(json_base_path, dataset, "label")

        if not os.path.exists(image_dir) or not os.path.exists(json_dir):
            print(f"⚠️ 跳过 {dataset}：路径不存在")
            continue

        print(f"\n处理数据集: {dataset}")

        command = [
            sys.executable, script_path,
            "--json_dir", json_dir,
            "--image_dir", image_dir,
            "--output_dir", output_dir,
            "--label_map", json.dumps(dict(label_map))  # 传递统一的标签映射
        ]

        if FIXED_PARAMS["labelme2yolo"]["seg"]:
            command.append("--seg")

        # 执行转换
        run_command(command, f"Labelme转YOLO - {dataset}")



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

def seg2det(input_dir: str, output_dir: str):
    """执行训练任务转换"""
    script_path = get_abs_path(FIXED_PARAMS["seg2det"]["script_path"])
    command = [
        sys.executable, script_path,
        "--input_dir", input_dir,
        "--output_dir", output_dir,
        "--mode", str(FIXED_PARAMS["seg2det"]["mode"]),
    ]

    if FIXED_PARAMS["patchandenhance"]["no_slice"]:
        command.append("--no_slice")

    run_command(command, "图像裁剪与增强")


def main():
    print("🚀 数据处理流水线启动（简化版）！")
    print(f"基础路径：{BASE_PATH}")
    print(f"待处理数据集：{DATASETS}")


    print("\n" + "=" * 100)
    print("📝 批量处理 Labelme 数据（使用统一标签映射）")
    print("=" * 100)
    # 第一步：labelme标签转换
    process_labelme2yolo_unified(
        DATASETS,
        BASE_PATH,
        JSON_BASE_PATH,
        OUTPUT_CONFIG["yolo_dir"],
    )

    # 第二步：执行 ROI 提取
    print("\n" + "=" * 100)
    print("📝 执行 YOLO ROI 区域提取")
    print("=" * 100)

    if not os.path.exists(OUTPUT_CONFIG["yolo_dir"]):
        print(f"❌ 错误：YOLO 数据集目录不存在 {OUTPUT_CONFIG['yolo_dir']}")
        sys.exit(1)

    process_roi_extractor(OUTPUT_CONFIG["yolo_dir"], OUTPUT_CONFIG["roi_dir"])

    # 第三步：执行图像裁剪增强
    print("\n" + "=" * 100)
    print("📝 执行图像裁剪与增强")
    print("=" * 100)

    if not os.path.exists(OUTPUT_CONFIG["roi_dir"]):
        print(f"❌ 错误：ROI 提取目录不存在 {OUTPUT_CONFIG['roi_dir']}")
        sys.exit(1)

    process_patch_enhance(OUTPUT_CONFIG["roi_dir"], OUTPUT_CONFIG["patch_dir"])

    # 第四步：执行训练任务转换
    print("\n" + "=" * 100)
    print("📝 执行训练任务转换")
    print("=" * 100)

    if not os.path.exists(OUTPUT_CONFIG["patch_dir"]):
        print(f"❌ 错误：ROI 提取目录不存在 {OUTPUT_CONFIG['patch_dir']}")
        sys.exit(1)

    seg2det(OUTPUT_CONFIG["patch_dir"], OUTPUT_CONFIG["cls_dir"])

    print("\n" + "🎉" * 50)
    print("🎉 所有数据处理步骤执行完成！")
    print(f"📁 最终结果保存目录：{OUTPUT_CONFIG['patch_dir']}")
    print("🎉" * 50)


if __name__ == "__main__":
    main()