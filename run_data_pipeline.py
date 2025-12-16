#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import sys
import json
from typing import List, Dict
from collections import OrderedDict
from pathlib import Path
import platform
from tqdm import tqdm
import shutil
import argparse
import copy
import yaml

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(CURRENT_DIR, "configs", "pipeline_profiles.yaml")

CONFIG_PATH = DEFAULT_CONFIG_PATH
ACTIVE_PROFILE_NAME = None
BASE_PATH = ""
JSON_BASE_PATH = ""
OUTPUT_BASE_DIR = ""
REFERENCE_LABEL_MAP_PATH = "/datasets/PAR/Xray/self/1120/labeled/roi2_merge/yolo/dataset.yaml"
DATASETS: List[str] = []
OUTPUT_CONFIG: Dict[str, str] = {}
FIXED_PARAMS: Dict[str, Dict] = {}
PARAM_LOG_PATH = ""
PARAM_LOG: Dict = {}

def _ensure_log_dir():
    if not PARAM_LOG_PATH:
        return
    os.makedirs(os.path.dirname(PARAM_LOG_PATH), exist_ok=True)

def save_param_log():
    """持久化流水线参数记录"""
    if not PARAM_LOG_PATH:
        return
    _ensure_log_dir()
    with open(PARAM_LOG_PATH, 'w', encoding='utf-8') as f:
        json.dump(PARAM_LOG, f, ensure_ascii=False, indent=2)

def log_command(step_name: str, command: List[str], param_key: str = None,
                extra_info: Dict = None):
    """记录脚本调用及其输入参数"""
    arguments = command[2:] if len(command) > 2 else []
    params = {}
    if param_key and param_key in FIXED_PARAMS:
        params = copy.deepcopy(FIXED_PARAMS[param_key])
        params.pop("script_path", None)

    entry = {
        "step": step_name,
        "arguments": arguments,
    }

    if params:
        entry["params"] = params

    if extra_info:
        entry["extra"] = extra_info

    PARAM_LOG["commands"].append(entry)
    save_param_log()

# 定义步骤信息
STEP_INFO = {
    '1': {
        'name': 'Labelme转YOLO',
        'func': 'step1_labelme2yolo',
        'input': None,
        'output': 'yolo_dir'
    },
    '2': {
        'name': 'YOLO ROI提取',
        'func': 'step2_roi_extractor',
        'input': 'yolo_dir',
        'output': 'roi_dir'
    },
    '3': {
        'name': 'YOLO竖图旋转',
        'func': 'step3_rotate_yolo',
        'input': 'roi_dir',
        'output': 'roi_rotate'
    },
    '4': {
        'name': '图像裁剪与增强',
        'func': 'step4_patch_enhance',
        'input': 'roi_rotate',
        'output': 'patch_dir'
    },
    '5': {
        'name': '训练任务转换',
        'func': 'step5_seg2det',
        'input': 'patch_dir',
        'output': 'cls_dir'
    },
    '6': {
        'name': 'YOLO转COCO',
        'func': 'step6_yolo2coco',
        'input': 'cls_dir',
        'output': 'coco_dir'
    },
    '7': {
        'name': 'COCO数据集合并',
        'func': 'step7_merge_coco',
        'input': 'coco_dir',
        'output': 'merged_coco_dir'
    }
}

def resolve_path(path_value: str, base_dir: str = None) -> str:
    """将路径解析为绝对路径，必要时相对 base_dir."""
    if path_value is None:
        return None

    expanded = os.path.expanduser(str(path_value))
    if os.path.isabs(expanded):
        return os.path.abspath(expanded)

    if base_dir:
        return os.path.abspath(os.path.join(base_dir, expanded))

    return os.path.abspath(expanded)


def load_pipeline_profile(config_path: str, requested_profile: str = None) -> str:
    """读取配置文件并应用指定 profile."""
    config_path = resolve_path(config_path or DEFAULT_CONFIG_PATH)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在：{config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f) or {}

    if not isinstance(config_data, dict):
        raise ValueError("配置文件格式错误，期望为字典结构")

    profiles = config_data.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("配置文件缺少 profiles 定义")

    profile_name = requested_profile or config_data.get("default_profile")
    if not profile_name:
        current_platform = platform.system()
        for name, profile_data in profiles.items():
            if profile_data.get("platform") == current_platform:
                profile_name = name
                break

    if not profile_name:
        profile_name = next(iter(profiles.keys()))

    if profile_name not in profiles:
        raise KeyError(f"配置文件中不存在 profile: {profile_name}")

    apply_profile(config_path, profile_name, profiles[profile_name])
    return profile_name


def apply_profile(config_path: str, profile_name: str, profile_data: Dict):
    """根据 profile 设置全局路径和参数."""
    global CONFIG_PATH, ACTIVE_PROFILE_NAME, BASE_PATH, JSON_BASE_PATH
    global OUTPUT_BASE_DIR, DATASETS, OUTPUT_CONFIG, FIXED_PARAMS
    global PARAM_LOG_PATH, PARAM_LOG
    global REFERENCE_LABEL_MAP_PATH

    paths_section = profile_data.get("paths") or {}
    base_path_raw = paths_section.get("base_path")
    if not base_path_raw:
        raise ValueError(f"profile {profile_name} 缺少 paths.base_path")
    json_base_raw = paths_section.get("json_base_path")
    if not json_base_raw:
        raise ValueError(f"profile {profile_name} 缺少 paths.json_base_path")

    output_base_raw = paths_section.get("output_base_dir") or "pipeline_outputs"
    labelme_params = (profile_data.get("params") or {}).get("labelme2yolo", {})
    reference_label_map_raw = paths_section.get("reference_label_map_path")
    if not reference_label_map_raw and not labelme_params.get("unify_to_crack"):
        raise ValueError(f"profile {profile_name} 缺少 paths.reference_label_map_path")

    CONFIG_PATH = config_path
    ACTIVE_PROFILE_NAME = profile_name
    BASE_PATH = resolve_path(base_path_raw)
    JSON_BASE_PATH = resolve_path(json_base_raw)
    OUTPUT_BASE_DIR = resolve_path(output_base_raw, BASE_PATH)
    REFERENCE_LABEL_MAP_PATH = resolve_path(reference_label_map_raw, BASE_PATH) if reference_label_map_raw else ""

    datasets = profile_data.get("datasets") or []
    if not isinstance(datasets, list):
        raise ValueError(f"profile {profile_name} 的 datasets 必须是列表")
    DATASETS = list(datasets)

    outputs_section = profile_data.get("outputs") or {}
    if not isinstance(outputs_section, dict) or not outputs_section:
        raise ValueError(f"profile {profile_name} 缺少 outputs 定义")
    resolved_outputs: Dict[str, str] = {}
    for key, value in outputs_section.items():
        if value is None:
            raise ValueError(f"profile {profile_name} 中 outputs.{key} 为空")
        resolved_outputs[key] = resolve_path(value, OUTPUT_BASE_DIR)
    OUTPUT_CONFIG = resolved_outputs

    FIXED_PARAMS = copy.deepcopy(profile_data.get("params") or {})

    param_log_raw = profile_data.get("param_log_path")
    PARAM_LOG_PATH = resolve_path(param_log_raw, OUTPUT_BASE_DIR) if param_log_raw else os.path.join(OUTPUT_BASE_DIR, "pipeline_params.json")

    required_outputs = {info["output"] for info in STEP_INFO.values() if info.get("output")}
    missing_outputs = sorted(key for key in required_outputs if key not in OUTPUT_CONFIG)
    if missing_outputs:
        raise ValueError(f"profile {profile_name} 缺少以下输出目录配置：{', '.join(missing_outputs)}")

    PARAM_LOG = {
        "config_path": CONFIG_PATH,
        "config_profile": ACTIVE_PROFILE_NAME,
        "base_path": BASE_PATH,
        "json_base_path": JSON_BASE_PATH,
        "reference_label_map_path": REFERENCE_LABEL_MAP_PATH,
        "datasets": list(DATASETS),
        "output_base_dir": OUTPUT_BASE_DIR,
        "selected_steps": [],
        "commands": []
    }

# ===========================================================================

def load_label_map_from_yaml(yaml_path: str) -> OrderedDict:
    """从 dataset.yaml 读取 label_id_map。"""
    if not yaml_path:
        raise ValueError("缺少参考 dataset.yaml 路径")

    yaml_file = Path(yaml_path)
    if not yaml_file.exists():
        raise FileNotFoundError(f"参考 dataset.yaml 不存在: {yaml_file}")

    try:
        with yaml_file.open("r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f)
    except yaml.YAMLError as err:
        raise RuntimeError(f"解析 {yaml_file} 失败: {err}") from err

    label_map_raw = yaml_data.get("label_id_map") if yaml_data else None
    if not isinstance(label_map_raw, dict):
        raise ValueError(f"{yaml_file} 缺少有效的 label_id_map")

    ordered_pairs = sorted(label_map_raw.items(), key=lambda item: item[1])
    return OrderedDict(ordered_pairs)

# ===========================================================================

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='数据处理流水线控制脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法：
  python %(prog)s --steps 1234567  # 运行所有7个步骤
  python %(prog)s --steps 1234     # 只运行前4个步骤
  python %(prog)s --steps 2345     # 只运行步骤2、3、4、5
  python %(prog)s --steps 135      # 只运行步骤1、3、5
  python %(prog)s --steps 6        # 只运行YOLO→COCO
  
步骤说明：
  1: Labelme转YOLO格式
  2: YOLO ROI区域提取
  3: YOLO竖图旋转
  4: 图像裁剪与增强
  5: 训练任务转换（seg转det/cls）
  6: YOLO→COCO 转换
  7: COCO 数据集合并
        """
    )
    
    parser.add_argument(
        '--steps',
        type=str,
        default='1234567',
        help='要执行的步骤编号，如 "1234567" 执行全部，"1234" 执行前四步 (默认: 1234567)'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='强制执行步骤，即使前置依赖的输出目录不存在'
    )

    parser.add_argument(
        '--config-path',
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help=f'配置文件路径 (默认: {DEFAULT_CONFIG_PATH})'
    )

    parser.add_argument(
        '--profile',
        type=str,
        default=None,
        help='配置文件中要使用的 profile 名称（默认使用 default_profile 或操作系统匹配项）'
    )

    return parser.parse_args()

def validate_steps(steps_str: str) -> List[str]:
    """验证并返回要执行的步骤列表"""
    valid_steps = set('1234567')
    steps = []
    
    for char in steps_str:
        if char in valid_steps:
            if char not in steps:  # 避免重复
                steps.append(char)
        else:
            print(f"⚠️ 警告：忽略无效的步骤编号 '{char}'")
    
    if not steps:
        print("❌ 错误：没有有效的步骤可执行！")
        sys.exit(1)
    
    return steps

def collect_all_labels(datasets: List[str], json_base_path: str,
                       unify_to_crack: bool = False,
                       reference_label_map_path: str = None) -> OrderedDict:
    """
    收集所有数据集的标签，建立统一的标签映射
    """
    # 如果启用了unify_to_crack，直接返回crack映射
    if unify_to_crack:
        print("\n📊 启用了 unify_to_crack，所有标签将统一为 'crack'")
        label_map = OrderedDict([('crack', 0)])
        print(f"📋 统一标签映射：{dict(label_map)}")
        return label_map

    if reference_label_map_path:
        print("\n📊 从参考 dataset.yaml 读取标签映射...")
        label_map = load_label_map_from_yaml(reference_label_map_path)
        print(f"📋 引用 {reference_label_map_path} 中的 label_id_map：")
        for label, idx in label_map.items():
            print(f"  {idx}: {label}")
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
    label_map = collect_all_labels(
        datasets,
        json_base_path,
        unify_to_crack,
        REFERENCE_LABEL_MAP_PATH
    )

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
        run_command(
            command,
            f"Labelme转YOLO - {dataset}",
            param_key="labelme2yolo",
            extra_info={"dataset": dataset}
        )

def run_command(command: List[str], step_name: str, param_key: str = None,
                extra_info: Dict = None):
    """执行命令"""
    log_command(step_name, command, param_key, extra_info)
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

    run_command(command, "YOLO ROI提取", param_key="yolo_roi_extractor")

def process_rotate_yolo(input_dir: str, output_dir: str):
    """执行YOLO竖图旋转标准化"""
    script_path = get_abs_path(FIXED_PARAMS["rotate_yolo"]["script_path"])
    command = [
        sys.executable, script_path,
        "--input", input_dir,
        "--output", output_dir
    ]

    run_command(command, "YOLO竖图旋转", param_key="rotate_yolo")

def process_patch_enhance(input_dir: str, output_dir: str):
    """执行图像裁剪增强"""
    script_path = get_abs_path(FIXED_PARAMS["patchandenhance"]["script_path"])
    patch_cfg = FIXED_PARAMS["patchandenhance"]
    slice_mode = patch_cfg.get("slice_mode")
    if slice_mode is None:
        slice_mode = 1 if patch_cfg.get("no_slice") else 2

    command = [
        sys.executable, script_path,
        "--input_dir", input_dir,
        "--output_dir", output_dir,
        "--enhance_mode", patch_cfg["enhance_mode"],
        "--label_mode", patch_cfg["label_mode"]
    ]

    if slice_mode == 2:
        command.extend([
            "--overlap", str(patch_cfg["overlap"]),
            "--window_size",
            str(patch_cfg["window_size"][0]),
            str(patch_cfg["window_size"][1])
        ])

    command.extend(["--slice_mode", str(slice_mode)])

    run_command(command, "图像裁剪与增强", param_key="patchandenhance")

def seg2det(input_dir: str, output_dir: str):
    """执行训练任务转换"""
    seg_cfg = FIXED_PARAMS["seg2det"]
    script_path = get_abs_path(seg_cfg["script_path"])
    command = [
        sys.executable, script_path,
        "--input_dir", input_dir,
        "--output_dir", output_dir,
        "--mode", str(seg_cfg["mode"]),
    ]
    if seg_cfg.get("balance_data"):
        command.append("--balance_data")
        balance_ratio = seg_cfg.get("balance_ratio")
        if balance_ratio is not None:
            command.extend(["--balance_ratio", str(balance_ratio)])

    run_command(command, "训练任务转换", param_key="seg2det")


def process_yolo2coco(input_dir: str, output_dir: str):
    """执行 YOLO→COCO 转换"""
    yolo2coco_cfg = FIXED_PARAMS.get("yolo2coco")
    if not yolo2coco_cfg:
        raise KeyError("配置缺少 params.yolo2coco")

    script_path = get_abs_path(yolo2coco_cfg["script_path"])
    command = [
        sys.executable, script_path,
        "--input_dir", input_dir,
        "--output_dir", output_dir
    ]

    task = yolo2coco_cfg.get("task")
    if task:
        command.extend(["--task", str(task)])
    if yolo2coco_cfg.get("test_split_ratio") is not None:
        command.extend(["--test_split_ratio", str(yolo2coco_cfg["test_split_ratio"])])
    if yolo2coco_cfg.get("split_seed") is not None:
        command.extend(["--split_seed", str(yolo2coco_cfg["split_seed"])])

    run_command(command, "YOLO转COCO", param_key="yolo2coco")


def process_merge_coco(dataset_a_dir: str, output_dir: str):
    """执行 COCO 数据集合并"""
    merge_cfg = FIXED_PARAMS.get("merge_coco")
    if not merge_cfg:
        raise KeyError("配置缺少 params.merge_coco")

    dataset_b_raw = merge_cfg.get("dataset_b")
    if not dataset_b_raw:
        raise ValueError("merge_coco.dataset_b 未配置，请在 YAML 中指定")

    dataset_b_path = resolve_path(dataset_b_raw, BASE_PATH)
    script_path = get_abs_path(merge_cfg["script_path"])
    command = [
        sys.executable, script_path,
        "--dataset-a", dataset_a_dir,
        "--dataset-b", dataset_b_path,
        "--output-dir", output_dir
    ]

    splits = merge_cfg.get("splits")
    if splits:
        command.extend(["--splits"] + [str(split) for split in splits])

    if merge_cfg.get("prefix_a"):
        command.extend(["--prefix-a", str(merge_cfg["prefix_a"])])
    if merge_cfg.get("prefix_b"):
        command.extend(["--prefix-b", str(merge_cfg["prefix_b"])])
    if merge_cfg.get("copy_images"):
        command.append("--copy-images")

    merge_ratio_config = merge_cfg.get("merge_ratio")
    logged_merge_ratio = None
    if isinstance(merge_ratio_config, (list, tuple)):
        ratio_values = [str(value) for value in merge_ratio_config if value is not None]
        if ratio_values:
            command.extend(["--merge-ratio"] + ratio_values)
            logged_merge_ratio = list(merge_ratio_config)
    elif merge_ratio_config is not None:
        command.extend(["--merge-ratio", str(merge_ratio_config)])
        logged_merge_ratio = merge_ratio_config

    run_command(
        command,
        "合并COCO数据集",
        param_key="merge_coco",
        extra_info={
            "dataset_b": str(dataset_b_path),
            "merge_ratio": logged_merge_ratio if logged_merge_ratio is not None else "default"
        }
    )

# =================== 步骤执行函数 ===================

def step1_labelme2yolo():
    """步骤1: Labelme转YOLO格式"""
    print("\n" + "=" * 100)
    print("📝 步骤1: 批量处理 Labelme 数据（使用统一标签映射）")
    print("=" * 100)
    
    process_labelme2yolo_unified(
        DATASETS,
        BASE_PATH,
        JSON_BASE_PATH,
        OUTPUT_CONFIG["yolo_dir"],
    )

def step2_roi_extractor():
    """步骤2: YOLO ROI提取"""
    print("\n" + "=" * 100)
    print("📝 步骤2: 执行 YOLO ROI 区域提取")
    print("=" * 100)
    
    if not os.path.exists(OUTPUT_CONFIG["yolo_dir"]):
        print(f"⚠️ 警告：YOLO 数据集目录不存在 {OUTPUT_CONFIG['yolo_dir']}")
        print("  提示：可能需要先执行步骤1")
    
    process_roi_extractor(OUTPUT_CONFIG["yolo_dir"], OUTPUT_CONFIG["roi_dir"])

def step3_rotate_yolo():
    """步骤3: YOLO竖图旋转"""
    print("\n" + "=" * 100)
    print("📝 步骤3: 执行竖图旋转归一")
    print("=" * 100)
    
    if not os.path.exists(OUTPUT_CONFIG["roi_dir"]):
        print(f"⚠️ 警告：ROI 提取目录不存在 {OUTPUT_CONFIG['roi_dir']}")
        print("  提示：可能需要先执行步骤2")
    
    process_rotate_yolo(OUTPUT_CONFIG["roi_dir"], OUTPUT_CONFIG["roi_rotate"])

def step4_patch_enhance():
    """步骤4: 图像裁剪与增强"""
    print("\n" + "=" * 100)
    print("📝 步骤4: 执行图像裁剪与增强")
    print("=" * 100)
    
    if not os.path.exists(OUTPUT_CONFIG["roi_rotate"]):
        print(f"⚠️ 警告：ROI 旋转目录不存在 {OUTPUT_CONFIG['roi_rotate']}")
        print("  提示：可能需要先执行步骤3")
    
    process_patch_enhance(OUTPUT_CONFIG["roi_rotate"], OUTPUT_CONFIG["patch_dir"])

def step5_seg2det():
    """步骤5: 训练任务转换"""
    print("\n" + "=" * 100)
    print("📝 步骤5: 执行训练任务转换")
    print("=" * 100)
    
    if not os.path.exists(OUTPUT_CONFIG["patch_dir"]):
        print(f"⚠️ 警告：patch 目录不存在 {OUTPUT_CONFIG['patch_dir']}")
        print("  提示：可能需要先执行步骤4")
    
    seg2det(OUTPUT_CONFIG["patch_dir"], OUTPUT_CONFIG["cls_dir"])


def step6_yolo2coco():
    """步骤6: YOLO→COCO 转换"""
    print("\n" + "=" * 100)
    print("📝 步骤6: YOLO→COCO 转换")
    print("=" * 100)

    if not os.path.exists(OUTPUT_CONFIG["cls_dir"]):
        print(f"⚠️ 警告：det 数据目录不存在 {OUTPUT_CONFIG['cls_dir']}")
        print("  提示：可能需要先执行步骤5")

    process_yolo2coco(OUTPUT_CONFIG["cls_dir"], OUTPUT_CONFIG["coco_dir"])


def step7_merge_coco():
    """步骤7: 合并 COCO 数据集"""
    print("\n" + "=" * 100)
    print("📝 步骤7: 合并 COCO 数据集")
    print("=" * 100)

    if not os.path.exists(OUTPUT_CONFIG["coco_dir"]):
        print(f"⚠️ 警告：COCO 转换输出不存在 {OUTPUT_CONFIG['coco_dir']}")
        print("  提示：可能需要先执行步骤6")

    process_merge_coco(OUTPUT_CONFIG["coco_dir"], OUTPUT_CONFIG["merged_coco_dir"])

def main():
    # 解析命令行参数
    args = parse_arguments()

    try:
        active_profile = load_pipeline_profile(args.config_path, args.profile)
    except Exception as exc:
        print(f"❌ 配置加载失败：{exc}")
        sys.exit(1)
    
    print("🚀 数据处理流水线启动（可控版本）！")
    print(f"配置文件：{CONFIG_PATH}")
    print(f"使用的profile：{active_profile}")
    print(f"基础路径：{BASE_PATH}")
    print(f"待处理数据集：{DATASETS}")
    
    # 验证步骤
    steps = validate_steps(args.steps)
    PARAM_LOG["selected_steps"] = list(steps)
    save_param_log()
    
    print(f"\n📌 将要执行的步骤：{' '.join(steps)}")
    for step in steps:
        print(f"  {step}: {STEP_INFO[step]['name']}")
    
    
    print("\n" + "=" * 100)
    print("开始执行选定的步骤")
    print("=" * 100)
    
    # 执行选定的步骤
    for step in steps:
        step_func_name = STEP_INFO[step]['func']
        step_func = globals()[step_func_name]
        
        try:
            step_func()
        except Exception as e:
            print(f"\n❌ 步骤{step}执行失败：{e}")
            if not args.force:
                print("终止执行（使用 --force 可以继续执行后续步骤）")
                sys.exit(1)
            else:
                print("使用了 --force 参数，继续执行后续步骤")
    
    # 完成信息
    print("\n" + "🎉" * 50)
    print("🎉 所选步骤执行完成！")
    print(f"📁 执行的步骤：{' '.join(steps)}")
    
    # 显示各步骤的输出目录
    for step in steps:
        output_key = STEP_INFO[step]['output']
        if output_key:
            output_dir = OUTPUT_CONFIG[output_key]
            print(f"  步骤{step}输出：{output_dir}")
    
    print("🎉" * 50)

if __name__ == "__main__":
    main()
