#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""针对 crop_weld_data 目录结构的精简数据处理流水线."""

import os
import sys
import json
import argparse
import subprocess
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List
import copy

# ========================= 路径配置 =========================
SYSTEM = sys.platform
if os.name == "nt":
    raise EnvironmentError("当前脚本仅配置了Linux路径，请在Linux环境下使用")

DATA_ROOT = Path("/home/lenovo/code/CHT/datasets/Xray/opensource/SWRD8bit").resolve()
IMAGES_ROOT = DATA_ROOT / "crop_weld_images"
JSON_ROOT = DATA_ROOT / "crop_weld_jsons_merged"
OUTPUT_BASE_DIR = DATA_ROOT / "swr_pipeline"
OUTPUT_CONFIG = {
    "yolo_dir": str(OUTPUT_BASE_DIR / "yolo"),
    "patch_dir": str(OUTPUT_BASE_DIR / "patch")
}

FIXED_PARAMS = {
    "labelme2yolo": {
        "seg": True,
        "unify_to_crack": False,
        "script_path": "convert/labelme2yolo.py"
    },
    "patchandenhance": {
        "overlap": 0.5,
        "enhance_mode": "windowing",
        "no_slice": False,
        "window_size": [1280, 640],
        "label_mode": "seg",
        "script_path": "convert/pj/patchandenhance.py"
    }
}

PARAM_LOG_PATH = OUTPUT_BASE_DIR / "pipeline_params_SWRD.json"
PARAM_LOG: Dict = {
    "data_root": str(DATA_ROOT),
    "image_root": str(IMAGES_ROOT),
    "json_root": str(JSON_ROOT),
    "output_base_dir": str(OUTPUT_BASE_DIR),
    "datasets": [],
    "selected_steps": [],
    "commands": []
}

STEP_INFO = {
    "1": {"name": "Labelme转YOLO", "func": "step1_labelme2yolo", "output": "yolo_dir"},
    "2": {"name": "图像裁剪与增强", "func": "step2_patch_enhance", "output": "patch_dir"}
}

DATASET_ENTRIES: List[Dict[str, str]] = []

# ========================= 工具函数 =========================

def _ensure_log_dir():
    PARAM_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

def save_param_log():
    _ensure_log_dir()
    with PARAM_LOG_PATH.open("w", encoding="utf-8") as f:
        json.dump(PARAM_LOG, f, ensure_ascii=False, indent=2)

def log_command(step_name: str, command: List[str], param_key: str = None,
                extra_info: Dict = None):
    arguments = command[2:] if len(command) > 2 else []
    params = {}
    if param_key and param_key in FIXED_PARAMS:
        params = copy.deepcopy(FIXED_PARAMS[param_key])
        params.pop("script_path", None)

    entry = {"step": step_name, "arguments": arguments}
    if params:
        entry["params"] = params
    if extra_info:
        entry["extra"] = extra_info

    PARAM_LOG["commands"].append(entry)
    save_param_log()

def run_command(command: List[str], step_name: str, param_key: str = None,
                extra_info: Dict = None):
    log_command(step_name, command, param_key, extra_info)
    print(f"\n{'=' * 80}")
    print(f"📌 正在执行【{step_name}】")
    print(f"命令：{' '.join(command)}")
    print(f"{'=' * 80}")

    try:
        subprocess.run(command, check=True, text=True)
        print(f"\n✅ 【{step_name}】执行成功！")
    except subprocess.CalledProcessError as exc:
        print(f"\n❌ 【{step_name}】执行失败，错误码：{exc.returncode}")
        sys.exit(1)

def get_abs_path(relative_path: str) -> str:
    current_script_dir = Path(__file__).parent
    return str((current_script_dir / relative_path).resolve())

# ========================= 数据集扫描与步骤 =========================

def discover_crop_weld_datasets(image_root: Path, json_root: Path) -> List[Dict[str, str]]:
    if not image_root.exists():
        raise FileNotFoundError(f"图像根目录不存在: {image_root}")
    if not json_root.exists():
        raise FileNotFoundError(f"标注根目录不存在: {json_root}")

    datasets: List[Dict[str, str]] = []
    for orient_dir in sorted([p for p in image_root.iterdir() if p.is_dir()]):
        for subset_dir in sorted([p for p in orient_dir.iterdir() if p.is_dir()]):
            json_dir = json_root / orient_dir.name / subset_dir.name
            if not json_dir.exists():
                print(f"  ⚠️ 跳过 {orient_dir.name}/{subset_dir.name}：缺少标注 {json_dir}")
                continue
            datasets.append({
                "name": f"{orient_dir.name}_{subset_dir.name}",
                "image_dir": str(subset_dir.resolve()),
                "json_dir": str(json_dir.resolve())
            })

    return datasets

def collect_all_labels(datasets: List[Dict[str, str]], unify_to_crack: bool = False) -> OrderedDict:
    if unify_to_crack:
        print("\n📊 启用了 unify_to_crack，所有标签统一为 'crack'")
        label_map = OrderedDict([('crack', 0)])
        print(f"📋 标签映射：{dict(label_map)}")
        return label_map

    print("\n📊 扫描所有数据集标签...")
    all_labels = set()
    dataset_labels: Dict[str, set] = {}

    for dataset in datasets:
        json_dir = dataset["json_dir"]
        dataset_labels[dataset["name"]] = set()

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
                        dataset_labels[dataset["name"]].add(label)
                        all_labels.add(label)
            except Exception as err:
                print(f"  ⚠️ 读取 {json_file} 失败：{err}")

        if dataset_labels[dataset["name"]]:
            print(f"  ✓ {dataset['name']}: {len(dataset_labels[dataset['name']])} 个标签")

    sorted_labels = sorted(all_labels)
    label_map = OrderedDict([(label, idx) for idx, label in enumerate(sorted_labels)])

    print(f"\n📋 合并标签映射（{len(label_map)} 类）：")
    for label, idx in label_map.items():
        owners = [d for d, labels in dataset_labels.items() if label in labels]
        print(f"  {idx}: {label} -> {', '.join(owners)}")

    return label_map

def process_labelme2yolo_unified(datasets: List[Dict[str, str]], output_dir: str):
    unify_to_crack = FIXED_PARAMS["labelme2yolo"].get("unify_to_crack", False)
    label_map = collect_all_labels(datasets, unify_to_crack)
    if not label_map:
        print("❌ 错误：未找到任何标签")
        sys.exit(1)

    script_path = get_abs_path(FIXED_PARAMS["labelme2yolo"]["script_path"])
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for dataset in datasets:
        image_dir = dataset["image_dir"]
        json_dir = dataset["json_dir"]
        if not os.path.exists(image_dir) or not os.path.exists(json_dir):
            print(f"⚠️ 跳过 {dataset['name']}：路径不存在")
            continue

        print(f"\n处理数据集: {dataset['name']}")
        command = [
            sys.executable, script_path,
            "--json_dir", json_dir,
            "--image_dir", image_dir,
            "--output_dir", output_dir,
            "--label_map", json.dumps(dict(label_map))
        ]
        if FIXED_PARAMS["labelme2yolo"].get("seg"):
            command.append("--seg")

        run_command(command, f"Labelme转YOLO - {dataset['name']}",
                    param_key="labelme2yolo", extra_info=dataset)

def process_patch_enhance(input_dir: str, output_dir: str):
    script_path = get_abs_path(FIXED_PARAMS["patchandenhance"]["script_path"])
    Path(output_dir).mkdir(parents=True, exist_ok=True)

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
    if FIXED_PARAMS["patchandenhance"].get("no_slice"):
        command.append("--no_slice")

    run_command(command, "图像裁剪与增强", param_key="patchandenhance")

# ========================= CLI 与步骤控制 =========================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SWRD 数据处理流水线：Labelme→YOLO→裁剪增强",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法：
  python %(prog)s --steps 12   # 运行全部两步
  python %(prog)s --steps 1    # 仅转换Labelme
  python %(prog)s --steps 2    # 仅裁剪增强（需先完成步骤1）
        """
    )
    parser.add_argument('--steps', type=str, default='12',
                        help='需要执行的步骤编号组合 (默认: 12)')
    parser.add_argument('--force', action='store_true',
                        help='强制继续执行，即使检测到缺少输入目录')
    return parser.parse_args()

def validate_steps(steps_str: str) -> List[str]:
    valid_steps = set(STEP_INFO.keys())
    steps: List[str] = []
    for char in steps_str:
        if char in valid_steps and char not in steps:
            steps.append(char)
        elif char not in valid_steps:
            print(f"⚠️ 忽略无效步骤编号 '{char}'")
    if not steps:
        raise ValueError("未选择任何有效步骤")
    return steps

def step1_labelme2yolo():
    if not DATASET_ENTRIES:
        raise RuntimeError("未找到有效数据集，无法执行步骤1")
    process_labelme2yolo_unified(DATASET_ENTRIES, OUTPUT_CONFIG["yolo_dir"])

def step2_patch_enhance():
    input_dir = OUTPUT_CONFIG["yolo_dir"]
    if not os.path.exists(input_dir):
        print(f"⚠️ 警告：YOLO输出目录不存在 {input_dir}，请先运行步骤1")
    process_patch_enhance(input_dir, OUTPUT_CONFIG["patch_dir"])

# ========================= 主入口 =========================

def main():
    args = parse_arguments()
    try:
        steps = validate_steps(args.steps)
    except ValueError as err:
        print(f"❌ {err}")
        sys.exit(1)

    datasets = discover_crop_weld_datasets(IMAGES_ROOT, JSON_ROOT)
    if not datasets:
        print("❌ 未在 crop_weld_images 中发现有效子目录，请检查数据组织方式")
        sys.exit(1)

    global DATASET_ENTRIES
    DATASET_ENTRIES = datasets
    PARAM_LOG["datasets"] = [d["name"] for d in datasets]
    PARAM_LOG["selected_steps"] = steps
    save_param_log()

    print("🚀 SWRD 数据处理流水线启动！")
    print(f"图像根目录：{IMAGES_ROOT}")
    print(f"标注根目录：{JSON_ROOT}")
    print(f"输出根目录：{OUTPUT_BASE_DIR}")
    print(f"待执行步骤：{' '.join(steps)}")
    for step in steps:
        print(f"  {step}: {STEP_INFO[step]['name']}")

    print("\n" + "=" * 80)
    print("开始执行选定步骤")
    print("=" * 80)

    for step in steps:
        func_name = STEP_INFO[step]['func']
        func = globals()[func_name]
        try:
            func()
        except Exception as exc:
            print(f"\n❌ 步骤{step}执行失败：{exc}")
            if args.force:
                print("使用 --force，继续后续步骤")
                continue
            sys.exit(1)

    print("\n" + "🎉" * 10)
    print("所有选定步骤执行完成！")
    for step in steps:
        output_key = STEP_INFO[step].get('output')
        if output_key:
            print(f"  步骤{step}输出目录：{OUTPUT_CONFIG[output_key]}")
    print("🎉" * 10)

if __name__ == "__main__":
    main()
