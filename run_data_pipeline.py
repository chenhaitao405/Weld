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
DEFAULT_CONFIG_PATH = os.path.join(CURRENT_DIR, "configs", "pipeline_profiles_new.yaml")
DEFAULT_PARAMS_PATH = os.path.join(CURRENT_DIR, "params.yaml")

CONFIG_PATH = DEFAULT_CONFIG_PATH
ACTIVE_PROFILE_NAME = None
BASE_PATH = ""
JSON_BASE_PATH = ""
OUTPUT_BASE_DIR = ""
REFERENCE_LABEL_MAP_PATH = "/datasets/PAR/Xray/self/1120/labeled/roi2_merge/yolo/dataset.yaml"
DATASETS: List[str] = []
DATASET_PAIRS: List[Dict[str, str]] = []
DATASET_ITEMS: List[Dict[str, str]] = []
OUTPUT_CONFIG: Dict[str, str] = {}
INTERMEDIATE_OUTPUTS: Dict[str, str] = {}
OUTPUTS_SECTION_RAW: Dict[str, str] = {}
PARAM_LOG_PATH_RAW = ""
FIXED_PARAMS: Dict[str, Dict] = {}
PARAM_LOG_PATH = ""
PARAM_LOG: Dict = {}
PIPELINE_SEED = None
PENDING_DATASET_PAIRS: List[Dict[str, str]] = []

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

# 定义步骤信息（合并步骤：23 与 456）
STEP_INFO = {
    '1': {
        'name': 'Labelme转YOLO',
        'func': 'step1_labelme2yolo',
        'input': None,
        'output': 'yolo_dir'
    },
    '23': {
        'name': 'YOLO ROI提取 + 竖图旋转',
        'func': 'step23_roi_rotate',
        'input': 'yolo_dir',
        'output': 'roi_rotate'
    },
    '456': {
        'name': '裁剪增强 + seg2det + YOLO转COCO',
        'func': 'step456_patch_seg2det_coco',
        'input': 'roi_rotate',
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


def _deep_update(base: Dict, updates: Dict) -> Dict:
    """递归合并字典，updates 覆盖 base。"""
    merged = copy.deepcopy(base)
    for key, value in (updates or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_params_config(params_path: str) -> Dict:
    """读取 params.yaml（可选）"""
    if not params_path:
        return {}
    params_path = resolve_path(params_path, os.getcwd())
    if not os.path.exists(params_path):
        print(f"⚠️ 警告：未找到 params 配置文件 {params_path}，将使用 profile 中的参数")
        return {}
    with open(params_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        print(f"⚠️ 警告：params 配置格式异常，期望字典结构：{params_path}")
        return {}
    data["__params_path__"] = params_path
    return data


def extract_preprocess_overrides(params_data: Dict) -> Dict:
    """从 params.yaml 提取预处理参数覆盖项。"""
    preprocess_cfg = params_data.get("preprocess") or params_data.get("pipeline") or {}
    if not isinstance(preprocess_cfg, dict):
        return {}

    overrides = copy.deepcopy(preprocess_cfg.get("params") or {})
    step_keys = [
        "labelme2yolo",
        "yolo_roi_extractor",
        "rotate_yolo",
        "patchandenhance",
        "seg2det",
        "yolo2coco",
        "merge_coco",
    ]
    for key in step_keys:
        if key in preprocess_cfg:
            value = preprocess_cfg.get(key)
            if value is not None:
                overrides[key] = copy.deepcopy(value)

    preprocess_cfg = copy.deepcopy(preprocess_cfg)
    preprocess_cfg["__overrides__"] = overrides
    return preprocess_cfg


def _normalize_dataset_pairs(raw_pairs: List[Dict[str, str]], base_dir: str) -> List[Dict[str, str]]:
    """规范化 dataset_pairs 配置并解析路径."""
    normalized: List[Dict[str, str]] = []
    for idx, pair in enumerate(raw_pairs):
        if not isinstance(pair, dict):
            raise ValueError(f"dataset_pairs[{idx}] 必须是字典")

        image_root_raw = pair.get("image_root") or pair.get("images") or pair.get("image_dir")
        label_root_raw = pair.get("label_root") or pair.get("labels") or pair.get("label_dir")
        if not image_root_raw or not label_root_raw:
            raise ValueError(f"dataset_pairs[{idx}] 需要包含 image_root 与 label_root")

        normalized.append({
            "name": pair.get("name"),
            "image_root": resolve_path(image_root_raw, base_dir),
            "label_root": resolve_path(label_root_raw, base_dir),
        })
    return normalized


def _has_json_files(target_dir: str) -> bool:
    if not target_dir or not os.path.isdir(target_dir):
        return False
    ignore_names = {
        "val_manifest.json",
        "train_manifest.json",
        "manifest.json",
    }
    for name in os.listdir(target_dir):
        if name.endswith(".json"):
            if name in ignore_names or name.endswith("_manifest.json"):
                continue
            return True
    return False


def _resolve_label_dir(base_dir: str) -> str:
    """优先使用 base_dir/label 或 base_dir/labels，否则回退到 base_dir。"""
    if not base_dir or not os.path.isdir(base_dir):
        return ""
    label_dir = os.path.join(base_dir, "label")
    labels_dir = os.path.join(base_dir, "labels")
    if _has_json_files(label_dir):
        return label_dir
    if _has_json_files(labels_dir):
        return labels_dir
    if _has_json_files(base_dir):
        return base_dir
    if os.path.isdir(label_dir):
        return label_dir
    if os.path.isdir(labels_dir):
        return labels_dir
    return base_dir


def _discover_dataset_items_from_pairs(pairs: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """从 dataset_pairs 中发现子数据集，并生成 json_dir/image_dir 对。"""
    items: List[Dict[str, str]] = []
    for pair in pairs:
        image_root = pair["image_root"]
        label_root = pair["label_root"]
        pair_tag = pair.get("name") or os.path.basename(label_root.rstrip(os.sep)) or os.path.basename(image_root.rstrip(os.sep))

        if not os.path.exists(label_root):
            print(f"⚠️ 跳过 {label_root}：标注目录不存在")
            continue
        if not os.path.exists(image_root):
            print(f"⚠️ 跳过 {image_root}：图像目录不存在")
            continue

        direct_json_dir = _resolve_label_dir(label_root)
        if _has_json_files(direct_json_dir):
            if not os.path.isdir(image_root):
                print(f"⚠️ 跳过 {pair_tag}：图像目录不存在 {image_root}")
                continue
            name = pair_tag or os.path.basename(label_root.rstrip(os.sep)) or os.path.basename(image_root.rstrip(os.sep))
            items.append({
                "name": name,
                "json_dir": direct_json_dir,
                "image_dir": image_root
            })
            continue

        subdirs = sorted([name for name in os.listdir(label_root)
                          if os.path.isdir(os.path.join(label_root, name))])
        if not subdirs:
            json_dir = _resolve_label_dir(label_root)
            if not _has_json_files(json_dir):
                print(f"⚠️ 未在 {label_root} 下找到子目录，且未找到JSON标注文件")
                continue
            if not os.path.isdir(image_root):
                print(f"⚠️ 跳过 {pair_tag}：图像目录不存在 {image_root}")
                continue
            name = pair_tag or os.path.basename(label_root.rstrip(os.sep)) or os.path.basename(image_root.rstrip(os.sep))
            items.append({
                "name": name,
                "json_dir": json_dir,
                "image_dir": image_root
            })
            continue

        for sub in subdirs:
            json_base = os.path.join(label_root, sub)
            json_dir = _resolve_label_dir(json_base)
            image_dir = os.path.join(image_root, sub)
            if not os.path.isdir(image_dir):
                print(f"⚠️ 跳过 {pair_tag}/{sub}：图像目录不存在 {image_dir}")
                continue
            if not _has_json_files(json_dir):
                print(f"⚠️ 跳过 {pair_tag}/{sub}：未找到JSON标注文件 {json_dir}")
                continue
            name = f"{pair_tag}/{sub}" if pair_tag else sub
            items.append({
                "name": name,
                "json_dir": json_dir,
                "image_dir": image_dir
            })
    return items


def _build_dataset_items_from_legacy(datasets: List[str], base_path: str, json_base_path: str) -> List[Dict[str, str]]:
    """兼容旧的 datasets + json_base_path 配置."""
    items: List[Dict[str, str]] = []
    for dataset in datasets:
        items.append({
            "name": dataset,
            "json_dir": os.path.join(json_base_path, dataset, "label"),
            "image_dir": os.path.join(base_path, dataset),
        })
    return items


def _rebuild_outputs():
    """根据 OUTPUT_BASE_DIR 与 OUTPUTS_SECTION_RAW 重新生成输出目录配置。"""
    global OUTPUT_CONFIG, INTERMEDIATE_OUTPUTS, PARAM_LOG_PATH

    if not OUTPUTS_SECTION_RAW:
        return

    resolved_outputs: Dict[str, str] = {}
    for key, value in OUTPUTS_SECTION_RAW.items():
        if value is None:
            raise ValueError(f"outputs.{key} 为空")
        resolved_outputs[key] = resolve_path(value, OUTPUT_BASE_DIR)
    OUTPUT_CONFIG = resolved_outputs

    tmp_root = os.path.join(OUTPUT_BASE_DIR, "_tmp")
    INTERMEDIATE_OUTPUTS = {
        "roi_dir": resolve_path(OUTPUTS_SECTION_RAW.get("roi_dir") or "ROI", OUTPUT_BASE_DIR)
        if "roi_dir" in OUTPUTS_SECTION_RAW else os.path.join(tmp_root, "ROI"),
        "patch_dir": resolve_path(OUTPUTS_SECTION_RAW.get("patch_dir") or "patch", OUTPUT_BASE_DIR)
        if "patch_dir" in OUTPUTS_SECTION_RAW else os.path.join(tmp_root, "patch"),
        "cls_dir": resolve_path(OUTPUTS_SECTION_RAW.get("cls_dir") or "patch_det", OUTPUT_BASE_DIR)
        if "cls_dir" in OUTPUTS_SECTION_RAW else os.path.join(tmp_root, "patch_det"),
    }

    if PARAM_LOG_PATH_RAW:
        PARAM_LOG_PATH = resolve_path(PARAM_LOG_PATH_RAW, OUTPUT_BASE_DIR)
    else:
        PARAM_LOG_PATH = os.path.join(OUTPUT_BASE_DIR, "pipeline_params.json")


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
    global OUTPUT_BASE_DIR, DATASETS, DATASET_PAIRS, DATASET_ITEMS
    global OUTPUT_CONFIG, INTERMEDIATE_OUTPUTS, OUTPUTS_SECTION_RAW, PARAM_LOG_PATH_RAW
    global FIXED_PARAMS
    global PARAM_LOG_PATH, PARAM_LOG
    global REFERENCE_LABEL_MAP_PATH

    paths_section = profile_data.get("paths") or {}
    base_path_raw = paths_section.get("base_path")
    if not base_path_raw:
        raise ValueError(f"profile {profile_name} 缺少 paths.base_path")
    json_base_raw = paths_section.get("json_base_path")

    output_base_raw = paths_section.get("output_base_dir") or "pipeline_outputs"
    labelme_params = (profile_data.get("params") or {}).get("labelme2yolo", {})
    reference_label_map_raw = paths_section.get("reference_label_map_path")
    if not reference_label_map_raw and not labelme_params.get("unify_to_crack"):
        print(f"⚠️ 警告：profile {profile_name} 未配置 reference_label_map_path，将自动扫描标签")

    CONFIG_PATH = config_path
    ACTIVE_PROFILE_NAME = profile_name
    BASE_PATH = resolve_path(base_path_raw)
    JSON_BASE_PATH = resolve_path(json_base_raw, BASE_PATH) if json_base_raw else ""
    OUTPUT_BASE_DIR = resolve_path(output_base_raw, BASE_PATH)
    REFERENCE_LABEL_MAP_PATH = resolve_path(reference_label_map_raw, BASE_PATH) if reference_label_map_raw else ""

    dataset_pairs_raw = profile_data.get("dataset_pairs")
    if dataset_pairs_raw:
        if not isinstance(dataset_pairs_raw, list):
            raise ValueError(f"profile {profile_name} 的 dataset_pairs 必须是列表")
        DATASET_PAIRS = _normalize_dataset_pairs(dataset_pairs_raw, BASE_PATH)
        DATASET_ITEMS = _discover_dataset_items_from_pairs(DATASET_PAIRS)
        DATASETS = [item["name"] for item in DATASET_ITEMS]
    else:
        if not json_base_raw:
            if PENDING_DATASET_PAIRS:
                DATASET_PAIRS = []
                DATASET_ITEMS = []
                DATASETS = []
            else:
                raise ValueError(f"profile {profile_name} 缺少 paths.json_base_path")
        datasets = profile_data.get("datasets") or []
        if not isinstance(datasets, list):
            raise ValueError(f"profile {profile_name} 的 datasets 必须是列表")
        DATASETS = list(datasets)
        DATASET_PAIRS = []
        DATASET_ITEMS = _build_dataset_items_from_legacy(DATASETS, BASE_PATH, JSON_BASE_PATH)

    outputs_section = profile_data.get("outputs") or {}
    if not isinstance(outputs_section, dict) or not outputs_section:
        raise ValueError(f"profile {profile_name} 缺少 outputs 定义")
    OUTPUTS_SECTION_RAW = copy.deepcopy(outputs_section)
    _rebuild_outputs()

    FIXED_PARAMS = copy.deepcopy(profile_data.get("params") or {})

    param_log_raw = profile_data.get("param_log_path")
    PARAM_LOG_PATH_RAW = param_log_raw or ""
    _rebuild_outputs()

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
        "dataset_pairs": list(DATASET_PAIRS),
        "dataset_items": list(DATASET_ITEMS),
        "output_base_dir": OUTPUT_BASE_DIR,
        "selected_steps": [],
        "commands": []
    }


def apply_preprocess_overrides(preprocess_cfg: Dict):
    """使用 params.yaml 覆盖 profile 中的参数。"""
    global FIXED_PARAMS, REFERENCE_LABEL_MAP_PATH, OUTPUT_BASE_DIR
    global DATASET_PAIRS, DATASET_ITEMS, DATASETS

    if not preprocess_cfg:
        return

    overrides = preprocess_cfg.get("__overrides__") or {}
    if overrides:
        FIXED_PARAMS = _deep_update(FIXED_PARAMS, overrides)

    override_label_map = preprocess_cfg.get("reference_label_map_path")
    if override_label_map:
        REFERENCE_LABEL_MAP_PATH = resolve_path(override_label_map, BASE_PATH)

    override_pairs = preprocess_cfg.get("dataset_pairs")
    if override_pairs:
        DATASET_PAIRS = _normalize_dataset_pairs(override_pairs, BASE_PATH)
        DATASET_ITEMS = _discover_dataset_items_from_pairs(DATASET_PAIRS)
        DATASETS = [item["name"] for item in DATASET_ITEMS]
        PARAM_LOG["dataset_pairs"] = list(DATASET_PAIRS)
        PARAM_LOG["dataset_items"] = list(DATASET_ITEMS)
        PARAM_LOG["datasets"] = list(DATASETS)

    override_output_base = preprocess_cfg.get("output_base_dir")
    if override_output_base:
        output_base = resolve_path(override_output_base, BASE_PATH)
        if output_base != OUTPUT_BASE_DIR:
            OUTPUT_BASE_DIR = output_base
            _rebuild_outputs()
            PARAM_LOG["output_base_dir"] = OUTPUT_BASE_DIR

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
  python %(prog)s --steps 1234567  # 运行全部（会合并为 1/23/456/7）
  python %(prog)s --steps 12       # 只运行步骤1和步骤23
  python %(prog)s --steps 456      # 只运行步骤456
  python %(prog)s --steps 7        # 只运行合并后的COCO

步骤说明（合并版）：
  1   : Labelme转YOLO格式
  23  : YOLO ROI提取 + 竖图旋转
  456 : 图像裁剪增强 + seg2det + YOLO→COCO
  7   : COCO数据集合并
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
        '--seed',
        type=int,
        default=None,
        help='随机种子（用于数据划分、平衡等可复现控制）'
    )

    parser.add_argument(
        '--config-path',
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help=f'配置文件路径 (默认: {DEFAULT_CONFIG_PATH})'
    )

    parser.add_argument(
        '--params-path',
        type=str,
        default=DEFAULT_PARAMS_PATH,
        help=f'参数文件路径 (默认: {DEFAULT_PARAMS_PATH})'
    )

    parser.add_argument(
        '--profile',
        type=str,
        default=None,
        help='配置文件中要使用的 profile 名称（默认使用 default_profile 或操作系统匹配项）'
    )

    return parser.parse_args()

def validate_steps(steps_str: str) -> List[str]:
    """验证并返回要执行的步骤列表（合并 23 / 456）"""
    valid_steps = set('1234567')
    requested = []

    for char in steps_str:
        if char in valid_steps:
            if char not in requested:  # 避免重复
                requested.append(char)
        else:
            print(f"⚠️ 警告：忽略无效的步骤编号 '{char}'")

    if not requested:
        print("❌ 错误：没有有效的步骤可执行！")
        sys.exit(1)

    merged_steps: List[str] = []
    if '1' in requested:
        merged_steps.append('1')
    if '2' in requested or '3' in requested:
        merged_steps.append('23')
    if '4' in requested or '5' in requested or '6' in requested:
        merged_steps.append('456')
    if '7' in requested:
        merged_steps.append('7')

    return merged_steps

def collect_all_labels(dataset_items: List[Dict[str, str]],
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
    dataset_labels: Dict[str, set] = {}

    for item in dataset_items:
        dataset_name = item.get("name") or "unknown"
        json_dir = item.get("json_dir")
        if not json_dir or not os.path.exists(json_dir):
            print(f"  ⚠️ 跳过 {dataset_name}：标注目录不存在 {json_dir}")
            continue

        dataset_labels[dataset_name] = set()

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
                        dataset_labels[dataset_name].add(label)
                        all_labels.add(label)

            except Exception as e:
                print(f"  ⚠️ 读取文件失败 {json_file}: {e}")

        if dataset_labels[dataset_name]:
            print(f"  ✓ {dataset_name}: 发现 {len(dataset_labels[dataset_name])} 个标签")

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

def process_labelme2yolo_unified(dataset_items: List[Dict[str, str]], output_dir: str, seed: int = None):
    """
    直接处理所有数据集到主目录，使用统一的标签映射
    """

    # 获取 unify_to_crack 设置
    unify_to_crack = FIXED_PARAMS["labelme2yolo"].get("unify_to_crack", False)
    if unify_to_crack:
        print("\n⚠️ 注意：已启用 unify_to_crack，所有标签将被统一为 'crack'")

    # 第一步：收集所有标签，建立统一映射
    label_map = collect_all_labels(
        dataset_items,
        unify_to_crack,
        REFERENCE_LABEL_MAP_PATH
    )

    if not label_map:
        print("❌ 错误：未找到任何标签！")
        sys.exit(1)

    # 第二步：直接处理所有数据集到主目录
    script_path = get_abs_path(FIXED_PARAMS["labelme2yolo"]["script_path"])

    for item in dataset_items:
        dataset_name = item.get("name") or "unknown"
        image_dir = item.get("image_dir")
        json_dir = item.get("json_dir")

        if not image_dir or not json_dir or not os.path.exists(image_dir) or not os.path.exists(json_dir):
            print(f"⚠️ 跳过 {dataset_name}：路径不存在")
            continue

        print(f"\n处理数据集: {dataset_name}")

        command = [
            sys.executable, script_path,
            "--json_dir", json_dir,
            "--image_dir", image_dir,
            "--output_dir", output_dir,
            "--label_map", json.dumps(dict(label_map))  # 传递统一的标签映射
        ]

        val_manifest = FIXED_PARAMS["labelme2yolo"].get("val_manifest")
        if val_manifest:
            command.extend(["--val_manifest", str(val_manifest)])

        if FIXED_PARAMS["labelme2yolo"]["seg"]:
            command.append("--seg")

        if seed is not None:
            command.extend(["--seed", str(seed)])

        # 执行转换
        run_command(
            command,
            f"Labelme转YOLO - {dataset_name}",
            param_key="labelme2yolo",
            extra_info={"dataset": dataset_name, "json_dir": json_dir, "image_dir": image_dir}
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

def process_patch_enhance(input_dir: str, output_dir: str, seed: int = None):
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
    if seed is not None:
        command.extend(["--seed", str(seed)])

    run_command(command, "图像裁剪与增强", param_key="patchandenhance")

def seg2det(input_dir: str, output_dir: str, seed: int = None):
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
    if seed is not None:
        command.extend(["--seed", str(seed)])

    run_command(command, "训练任务转换", param_key="seg2det")


def process_yolo2coco(input_dir: str, output_dir: str, seed: int = None):
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
    if seed is not None:
        command.extend(["--split_seed", str(seed)])
    elif yolo2coco_cfg.get("split_seed") is not None:
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
        DATASET_ITEMS,
        OUTPUT_CONFIG["yolo_dir"],
        seed=PIPELINE_SEED,
    )

def step23_roi_rotate():
    """步骤23: YOLO ROI提取 + 竖图旋转"""
    print("\n" + "=" * 100)
    print("📝 步骤23: 执行 YOLO ROI 提取 + 竖图旋转")
    print("=" * 100)

    if not os.path.exists(OUTPUT_CONFIG["yolo_dir"]):
        print(f"⚠️ 警告：YOLO 数据集目录不存在 {OUTPUT_CONFIG['yolo_dir']}")
        print("  提示：可能需要先执行步骤1")

    roi_dir = INTERMEDIATE_OUTPUTS["roi_dir"]
    process_roi_extractor(OUTPUT_CONFIG["yolo_dir"], roi_dir)
    process_rotate_yolo(roi_dir, OUTPUT_CONFIG["roi_rotate"])


def step456_patch_seg2det_coco():
    """步骤456: 图像裁剪增强 + seg2det + YOLO→COCO"""
    print("\n" + "=" * 100)
    print("📝 步骤456: 图像裁剪增强 + seg2det + YOLO→COCO")
    print("=" * 100)

    if not os.path.exists(OUTPUT_CONFIG["roi_rotate"]):
        print(f"⚠️ 警告：ROI 旋转目录不存在 {OUTPUT_CONFIG['roi_rotate']}")
        print("  提示：可能需要先执行步骤23")

    patch_dir = INTERMEDIATE_OUTPUTS["patch_dir"]
    cls_dir = INTERMEDIATE_OUTPUTS["cls_dir"]

    process_patch_enhance(OUTPUT_CONFIG["roi_rotate"], patch_dir, seed=PIPELINE_SEED)
    seg2det(patch_dir, cls_dir, seed=PIPELINE_SEED)
    process_yolo2coco(cls_dir, OUTPUT_CONFIG["coco_dir"], seed=PIPELINE_SEED)


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
        params_data = load_params_config(args.params_path)
        preprocess_cfg = extract_preprocess_overrides(params_data)
        global PENDING_DATASET_PAIRS
        PENDING_DATASET_PAIRS = preprocess_cfg.get("dataset_pairs") if preprocess_cfg else []
        profile_override = preprocess_cfg.get("profile") if preprocess_cfg else None
        selected_profile = args.profile or profile_override
        active_profile = load_pipeline_profile(args.config_path, selected_profile)
        apply_preprocess_overrides(preprocess_cfg)
    except Exception as exc:
        print(f"❌ 配置加载失败：{exc}")
        sys.exit(1)

    global PIPELINE_SEED
    seed_from_params = preprocess_cfg.get("seed") if preprocess_cfg else None
    PIPELINE_SEED = args.seed if args.seed is not None else seed_from_params

    print("🚀 数据处理流水线启动（可控版本）！")
    print(f"配置文件：{CONFIG_PATH}")
    if params_data.get("__params_path__"):
        print(f"参数文件：{params_data['__params_path__']}")
    print(f"使用的profile：{active_profile}")
    print(f"基础路径：{BASE_PATH}")
    if DATASET_PAIRS:
        print(f"数据集对数量：{len(DATASET_PAIRS)}")
        for pair in DATASET_PAIRS:
            print(f"  - images: {pair['image_root']} | labels: {pair['label_root']}")
        print(f"待处理子数据集数量：{len(DATASET_ITEMS)}")
    else:
        print(f"待处理数据集：{DATASETS}")
    
    # 验证步骤
    steps = validate_steps(args.steps)
    PARAM_LOG["selected_steps"] = list(steps)
    PARAM_LOG["seed"] = PIPELINE_SEED
    if params_data.get("__params_path__"):
        PARAM_LOG["params_path"] = params_data["__params_path__"]
    if preprocess_cfg:
        PARAM_LOG["params_overrides"] = preprocess_cfg.get("__overrides__", {})
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
