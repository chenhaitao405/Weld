#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
自动化混合自有数据与开源数据、转换为 COCO、并发起训练实验的流水线脚本。

步骤：
1. 按照目标比例将开源 train 集按软链方式混入自有 train 集（val/test 维持自有数据）。
2. 生成新的 YOLO 数据集目录并可选写入 dataset.yaml。
3. 调用 convert/yolo2coco.py 转为 COCO 标注格式。
4. 按配置的 GPU 并发度启动训练命令模板。
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import random
import shutil
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.constants import IMAGE_EXTENSIONS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="自动混合数据并触发 RF-DETR 训练的流水线"
    )
    parser.add_argument("--self-yolo", required=True,
                        help="自有数据的 YOLO 根目录（包含 images/ 与 labels/ 子目录）")
    parser.add_argument("--open-yolo", required=True,
                        help="开源数据的 YOLO 根目录（仅使用 train split）")
    parser.add_argument("--output-root", default="mixed_experiments",
                        help="混合数据与实验的输出根目录")
    parser.add_argument("--ratios", type=float, nargs="+", default=[0.1, 0.5, 1.0, 2.0],
                        help="开源数据混入占自有 train 数的比例列表，例如 0.1 = 10%")
    parser.add_argument("--seed", type=int, default=42, help="混合抽样随机种子")
    parser.add_argument("--force", action="store_true",
                        help="如目标目录已存在则先删除后重建（仅在生成数据时生效）")
    parser.add_argument("--yolo2coco-script", default=str(PROJECT_ROOT / "convert" / "yolo2coco.py"),
                        help="convert/yolo2coco.py 脚本路径")
    parser.add_argument("--prepare-data", action="store_true",
                        help="开启后才会重新生成混合数据和 COCO 转换；未开启时直接进入训练阶段")
    parser.add_argument("--train-output-root", default=str(PROJECT_ROOT / "runs" / "mixed_experiments"),
                        help="训练输出保存根目录")
    parser.add_argument("--gpus", default="0,1,2",
                        help="可用 GPU ID 列表，逗号分隔（例如 0,1,2）")
    parser.add_argument("--max-parallel", type=int, default=3,
                        help="最大并行训练任务数（默认与 GPU 数相同）")
    parser.add_argument("--skip-train", action="store_true",
                        help="仅生成数据与 COCO 文件，不触发训练")
    parser.add_argument("--epochs", type=int, default=500, help="内置RF-DETR训练轮数")
    parser.add_argument("--batch-size", type=int, default=4, help="内置训练 batch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="内置训练梯度累积步数")
    parser.add_argument("--lr", type=float, default=1e-4, help="内置训练学习率")
    parser.add_argument("--early-stopping", action="store_true",
                        help="内置训练启用 early stopping")
    parser.add_argument("--model-variant", choices=["large", "base"], default="large",
                        help="内置训练使用的RF-DETR模型变体")
    return parser.parse_args()


def list_label_pairs(yolo_root: Path, split: str) -> List[Tuple[Path, Path, Path]]:
    labels_dir = yolo_root / "labels" / split
    images_dir = yolo_root / "images" / split
    if not labels_dir.exists():
        return []
    pairs: List[Tuple[Path, Path, Path]] = []
    for label_file in labels_dir.rglob("*.txt"):
        rel_path = label_file.relative_to(labels_dir)
        image_file = resolve_image_for_label(images_dir, rel_path)
        if not image_file:
            print(f"⚠️ 找不到与标签匹配的图像: {label_file}")
            continue
        pairs.append((label_file, image_file, rel_path))
    return sorted(pairs, key=lambda item: str(item[0]))


def resolve_image_for_label(images_dir: Path, rel_path: Path) -> Optional[Path]:
    for ext in IMAGE_EXTENSIONS:
        candidate = images_dir / rel_path.with_suffix(ext)
        if candidate.exists():
            return candidate
    # 兼容 label 与 image 同扩展
    candidate = images_dir / rel_path.with_suffix(".jpg")
    return candidate if candidate.exists() else None


def ratio_tag(value: float) -> str:
    scaled = int(round(value * 100))
    return f"p{scaled:03d}"


def ensure_clean_dir(path: Path, force: bool = False):
    if path.exists() and force:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _flatten_rel_path(tag: str, rel_path: Path) -> Path:
    base = rel_path.as_posix().replace("/", "__")
    prefix = f"{tag}__" if tag else ""
    return Path(prefix + base)


def _prefixed_rel_path(tag: Optional[str], rel_path: Path) -> Path:
    return (Path(tag) / rel_path) if tag else rel_path


def symlink_pair(label_file: Path, image_file: Path, target_root: Path,
                 split: str, rel_path: Path, tag: Optional[str] = None,
                 flatten: bool = False):
    if flatten:
        rel = _flatten_rel_path(tag or "", rel_path)
    else:
        rel = _prefixed_rel_path(tag, rel_path)

    target_label = target_root / "labels" / split / rel
    image_rel = rel.with_suffix(image_file.suffix)
    target_image = target_root / "images" / split / image_rel
    target_label.parent.mkdir(parents=True, exist_ok=True)
    target_image.parent.mkdir(parents=True, exist_ok=True)
    if not target_label.exists():
        target_label.symlink_to(label_file)
    if not target_image.exists():
        target_image.symlink_to(image_file)


def mix_train_split(self_pairs: List[Tuple[Path, Path, Path]],
                    open_pairs: List[Tuple[Path, Path, Path]],
                    target_root: Path,
                    ratio: float,
                    seed: int) -> Dict[str, int]:
    rng = random.Random(seed + int(ratio * 1000))
    target_stats = {
        "self_train": len(self_pairs),
        "open_candidates": len(open_pairs),
        "open_selected": 0
    }

    for label_file, image_file, rel_path in self_pairs:
        symlink_pair(label_file, image_file, target_root, "train", rel_path,
                     tag="self", flatten=True)

    required = int(math.ceil(len(self_pairs) * ratio))
    if required <= 0 or not open_pairs:
        return target_stats

    if required >= len(open_pairs):
        selected = open_pairs
    else:
        selected = rng.sample(open_pairs, required)

    target_stats["open_selected"] = len(selected)

    for label_file, image_file, rel_path in selected:
        symlink_pair(label_file, image_file, target_root, "train", rel_path,
                     tag="open", flatten=True)

    return target_stats


def mirror_split(source_root: Path, target_root: Path, split: str):
    labels_dir = source_root / "labels" / split
    images_dir = source_root / "images" / split
    if not labels_dir.exists():
        return
    for label_file in labels_dir.rglob("*.txt"):
        rel_path = label_file.relative_to(labels_dir)
        image_file = resolve_image_for_label(images_dir, rel_path)
        if not image_file:
            continue
        symlink_pair(label_file, image_file, target_root, split, rel_path)


def load_yaml(path: Path) -> Dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_dataset_yaml(template_yaml: Dict, target_root: Path):
    data = template_yaml.copy()
    data["path"] = str(target_root)
    data["train"] = "images/train"
    data["val"] = data.get("val", "images/val")
    data["test"] = data.get("test", "images/test")
    yaml_path = target_root / "dataset.yaml"
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def convert_to_coco(yolo_dir: Path, coco_dir: Path, script_path: Path):
    cmd = [
        sys.executable, str(script_path),
        "--input_dir", str(yolo_dir),
        "--output_dir", str(coco_dir),
        "--task", "det"
    ]
    print(f"[CMD] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def internal_training_worker(dataset_dir: Path,
                             output_dir: Path,
                             gpu_id: str,
                             train_cfg: Dict[str, Any]):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    from rfdetr import RFDETRBase, RFDETRLarge  # noqa: WPS433
    model_cls = RFDETRLarge if train_cfg["model_variant"] == "large" else RFDETRBase
    model = model_cls()
    model.train(
        dataset_dir=str(dataset_dir),
        epochs=train_cfg["epochs"],
        batch_size=train_cfg["batch_size"],
        grad_accum_steps=train_cfg["grad_accum"],
        lr=train_cfg["lr"],
        output_dir=str(output_dir),
        early_stopping=train_cfg["early_stopping"]
    )


def launch_internal_training(experiments: Sequence[Dict[str, Path]],
                             gpus: List[str],
                             max_parallel: int,
                             train_cfg: Dict[str, Any]):
    queue = deque(experiments)
    running: List[Dict[str, Any]] = []
    gpu_pool = deque(gpus)
    max_parallel = min(max_parallel, len(gpus))

    while queue or running:
        while queue and len(running) < max_parallel and gpu_pool:
            exp = queue.popleft()
            gpu = gpu_pool.popleft()
            train_dir = exp["train_dir"]
            train_dir.mkdir(parents=True, exist_ok=True)
            print(f"[TRAIN][GPU{gpu}] 内置RF-DETR训练 -> {train_dir}")
            proc = mp.Process(
                target=internal_training_worker,
                args=(exp["coco_dir"], train_dir, gpu, train_cfg),
                daemon=False
            )
            proc.start()
            running.append({"proc": proc, "gpu": gpu, "tag": exp["ratio_tag"]})

        if not running:
            break

        time.sleep(5)
        for job in running[:]:
            ret = job["proc"].exitcode
            if ret is None:
                continue
            running.remove(job)
            gpu_pool.append(job["gpu"])
            if ret != 0:
                raise RuntimeError(f"训练任务 ratio={job['tag']} 失败，返回码 {ret}")


def main():
    args = parse_args()
    self_root = Path(args.self_yolo).resolve()
    open_root = Path(args.open_yolo).resolve()
    output_root = Path(args.output_root).resolve()
    yolo2coco_script = Path(args.yolo2coco_script).resolve()

    if args.prepare_data:
        ensure_clean_dir(output_root, force=args.force)
        template_yaml = load_yaml(self_root / "dataset.yaml")
        self_train_pairs = list_label_pairs(self_root, "train")
        if not self_train_pairs:
            raise RuntimeError(f"未找到自有数据 train 标签: {self_root}")
        open_train_pairs = list_label_pairs(open_root, "train")
        if not open_train_pairs and any(r > 0 for r in args.ratios):
            raise RuntimeError(
                f"开源数据 train 目录为空（{open_root/'labels/train'}）。请确认路径或先生成 YOLO train。"
            )
    else:
        output_root.mkdir(parents=True, exist_ok=True)
        template_yaml = None
        self_train_pairs = []
        open_train_pairs = []

    experiments: List[Dict[str, Path]] = []

    for ratio in args.ratios:
        tag = ratio_tag(ratio)
        experiment_dir = output_root / f"ratio_{tag}"
        yolo_dir = experiment_dir / "yolo"
        coco_dir = experiment_dir / "coco"
        meta_dir = experiment_dir / "meta"
        if args.prepare_data:
            ensure_clean_dir(yolo_dir, force=True)
            ensure_clean_dir(coco_dir, force=True)
            ensure_clean_dir(meta_dir, force=True)

            stats = mix_train_split(self_train_pairs, open_train_pairs, yolo_dir, ratio, args.seed)
            mirror_split(self_root, yolo_dir, "val")
            mirror_split(self_root, yolo_dir, "valid")
            mirror_split(self_root, yolo_dir, "test")
            save_dataset_yaml(template_yaml, yolo_dir)

            meta = {
                "ratio": ratio,
                "ratio_tag": tag,
                "self_train": stats["self_train"],
                "open_candidates": stats["open_candidates"],
                "open_selected": stats["open_selected"]
            }
            with (meta_dir / "mix_summary.json").open("w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            convert_to_coco(yolo_dir, coco_dir, yolo2coco_script)
        else:
            if not yolo_dir.exists() or not coco_dir.exists():
                raise FileNotFoundError(
                    f"未找到已有的数据输出目录 {experiment_dir}，请先使用 --prepare-data 生成。"
                )

        experiments.append({
            "ratio": ratio,
            "ratio_tag": tag,
            "experiment_dir": experiment_dir,
            "yolo_dir": yolo_dir,
            "coco_dir": coco_dir,
            "train_dir": Path(args.train_output_root).resolve() / f"ratio_{tag}"
        })

    if args.skip_train:
        print("已按要求跳过训练阶段。")
        return

    gpu_list = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpu_list:
        raise ValueError("请通过 --gpus 指定至少一个 GPU ID")

    train_cfg = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "early_stopping": args.early_stopping,
        "model_variant": args.model_variant
    }
    launch_internal_training(
        experiments=experiments,
        gpus=gpu_list,
        max_parallel=args.max_parallel or len(gpu_list),
        train_cfg=train_cfg
    )


if __name__ == "__main__":
    main()
