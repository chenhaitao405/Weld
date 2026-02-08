import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import mlflow
import yaml

from rfdetr import RFDETRSeg2XLarge, RFDETRLarge, RFDETRSegPreview, RFDETRMedium,RFDETRSegXLarge,RFDETR2XLarge

CLASS_NAMES = [
    "其他",
    "内凹",
    "咬边",
    "圆形缺陷",
    "未焊透",
    "未熔合",
    "条形缺陷",
    "裂纹"
]

model = RFDETR2XLarge()
#预训练数据 /datasets/PAR/Xray/datasets_merge/patch640_ratio_1
# SWRD DATSET /datasets/PAR/Xray/opensource/SWRD8bit/swr_pipeline/patch_det_coco
#############/datasets/PAR/Xray/opensource/SWRD8bit/swr_pipeline/coco_patch_det_slice3
# 1120DATASET /datasets/PAR/Weld/data/pipeline_pair_1120/coco_from_patch\
# "/datasets/PAR/Weld/data/pipeline_pair_1120_mannualval/coco_from_patch"
# 1120DATASET 1208+1120 /datasets/PAR/Weld/data/datasets_merge/1120_1208/coco_det
        #####           /datasets/PAR/Weld/data/datasets_merge/1120_1208_primary/coco_det
        #/datasets/PAR/Weld/data/datasets_merge/1208_1120_mannual_swrdslice3/coco_det

##/datasets/PAR/Weld/data/pipeline_pair_1120_mannualval_patch880_enhance/coco_from_patch

DEFAULT_DATASET_DIR = "/datasets/PAR/Weld/data/pipeline_pair_1120_mannualval_patch880_enhance/coco_from_patch"
DEFAULT_BATCH_SIZE = 4
DEFAULT_GRAD_ACCUM_STEPS = 64
DEFAULT_LR = 1e-4
DEFAULT_OUTPUT_DIR = "outputs/RFDETR2XLarge/pipeline_SWRDpatch/manual_val/windowingtest"
DEFAULT_EARLY_STOPPING = True
DEFAULT_RUN_NAME = "1120_1120_SWRD_patch_enhance"
DEFAULT_EPOCHS = 500
## 预训练模型 /datasets/PAR/Weld/outputs/RFDETR2XLarge/patch640_ratio_1/checkpoint_best_regular.pth
DEFAULT_RESUME = "/datasets/PAR/Weld/outputs/RFDETR2XLarge/patch640_ratio_1/checkpoint_best_regular.pth" #"/datasets/PAR/Weld/outputs/RFDETR2XLarge/pipeline_SWRDpatch/manual_val/1120_SWRD/checkpoint_best_total.pth"
DEFAULT_METRICS_PATH = "metrics/rfdetr.json"
DEFAULT_KEEP_BEST_ONLY = False

BASE_DIR = Path(__file__).resolve().parent.parent


def _resolve_path(path_value: Optional[str]) -> Optional[Path]:
    if not path_value:
        return None
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = (BASE_DIR / path).resolve()
    return path


def _load_params_file(params_path: Path, parser: argparse.ArgumentParser) -> Dict[str, Any]:
    if not params_path.exists():
        parser.error(f"Params file {params_path} not found")
    raw_text = params_path.read_text(encoding="utf-8")
    if params_path.suffix.lower() == ".json":
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            parser.error(f"Params file {params_path} is not valid JSON: {exc}")
    else:
        try:
            data = yaml.safe_load(raw_text)
        except yaml.YAMLError as exc:
            parser.error(f"Params file {params_path} is not valid YAML: {exc}")
    if data is None:
        return {}
    if not isinstance(data, dict):
        parser.error("Params file must contain a dict at the top level")
    if "rfdetr" in data:
        rfdetr_section = data.get("rfdetr")
        if rfdetr_section is None:
            return {}
        if not isinstance(rfdetr_section, dict):
            parser.error("rfdetr section in params file must be a dict")
        return rfdetr_section
    return data


def _load_training_args() -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description="RF-DETR training entrypoint")
    parser.add_argument("--params-file", type=str, help="Path to JSON/YAML file overriding training args")
    parser.add_argument("--dataset-dir", type=str, default=None,
                        help=f"Dataset root directory (default: {DEFAULT_DATASET_DIR})")
    parser.add_argument("--batch-size", type=int, default=None,
                        help=f"Batch size (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--grad-accum-steps", type=int, default=None,
                        help=f"Gradient accumulation steps (default: {DEFAULT_GRAD_ACCUM_STEPS})")
    parser.add_argument("--output-dir", type=str, default=None,
                        help=("Output root directory; actual run output will be output_dir/run "
                              f"(default: {DEFAULT_OUTPUT_DIR})"))
    parser.add_argument("--run", type=str, default=None,
                        help=f"Run name, also used as MLflow run name (default: {DEFAULT_RUN_NAME})")
    parser.add_argument("--resume", type=str, default=None,
                        help=f"Resume checkpoint path (default: {DEFAULT_RESUME})")
    parser.add_argument("--resolution", type=int, default=None,
                        help="Input resolution")
    parser.add_argument("--metrics-path", type=str, default=None,
                        help=f"Metrics output path (default: {DEFAULT_METRICS_PATH})")
    parser.add_argument("--keep-best-only", dest="keep_best_only", action="store_true",
                        help="Keep only checkpoint_best_total.pth in the run output dir")
    parser.add_argument("--keep-all-checkpoints", dest="keep_best_only", action="store_false",
                        help="Keep all checkpoints in the run output dir")
    parser.set_defaults(keep_best_only=None)

    args = parser.parse_args()
    training_args: Dict[str, Any] = {
        "dataset_dir": DEFAULT_DATASET_DIR,
        "epochs": DEFAULT_EPOCHS,
        "batch_size": DEFAULT_BATCH_SIZE,
        "grad_accum_steps": DEFAULT_GRAD_ACCUM_STEPS,
        "lr": DEFAULT_LR,
        "output_dir": DEFAULT_OUTPUT_DIR,
        "early_stopping": DEFAULT_EARLY_STOPPING,
        "run": DEFAULT_RUN_NAME,
        "resume": DEFAULT_RESUME,
        "class_names": CLASS_NAMES,
        "num_classes": len(CLASS_NAMES),
        "metrics_path": DEFAULT_METRICS_PATH,
        "keep_best_only": DEFAULT_KEEP_BEST_ONLY,
    }

    if args.params_file:
        params_path = Path(args.params_file).expanduser()
        overrides = _load_params_file(params_path, parser)
        for key, value in overrides.items():
            if key in {"epochs", "lr", "early_stopping", "class_names", "num_classes"}:
                continue
            training_args[key] = value

    override_keys = [
        "dataset_dir",
        "batch_size",
        "grad_accum_steps",
        "output_dir",
        "run",
        "resume",
        "resolution",
        "metrics_path",
        "keep_best_only",
    ]

    for key in override_keys:
        value = getattr(args, key)
        if value is not None:
            training_args[key] = value

    return training_args


training_args = _load_training_args()
dataset_dir_path = _resolve_path(training_args.get("dataset_dir")) or BASE_DIR
resume_path = _resolve_path(training_args.get("resume"))
metrics_path = _resolve_path(training_args.get("metrics_path", "metrics/rfdetr.json")) or (BASE_DIR / "metrics/rfdetr.json")
metrics_path.parent.mkdir(parents=True, exist_ok=True)

output_root = _resolve_path(training_args["output_dir"]) or (BASE_DIR / "outputs/rfdetr")
run_name = training_args["run"]
run_output_dir = output_root / run_name
run_output_dir.mkdir(parents=True, exist_ok=True)
training_args["output_dir"] = str(output_root)
training_args["run_output_dir"] = str(run_output_dir)
training_args["dataset_dir"] = str(dataset_dir_path)
training_args["resume"] = str(resume_path) if resume_path else None
training_args["metrics_path"] = str(metrics_path)
MLFLOW_TRACKING_DIR = Path(__file__).resolve().parent.parent / "mlruns"
MLFLOW_TRACKING_DIR.mkdir(parents=True, exist_ok=True)
mlflow.set_tracking_uri(f"file:{MLFLOW_TRACKING_DIR}")
mlflow.set_experiment("rf-detr")

class _StreamTee:
    """将标准输出/错误同时写入多个流"""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self._streams:
            stream.flush()


def _log_terminal_output(log_path: Path):
    """上下文管理器：将stdout/stderr同时写入日志文件"""
    class _LogContext:
        def __enter__(self_nonlocal):
            self_nonlocal.log_file = log_path.open("a", encoding="utf-8")
            header = f"\n===== Run started {datetime.now().isoformat()} =====\n"
            self_nonlocal.log_file.write(header)
            self_nonlocal.log_file.flush()

            self_nonlocal.original_stdout = sys.stdout
            self_nonlocal.original_stderr = sys.stderr
            sys.stdout = _StreamTee(sys.stdout, self_nonlocal.log_file)
            sys.stderr = _StreamTee(sys.stderr, self_nonlocal.log_file)
            return self_nonlocal

        def __exit__(self_nonlocal, exc_type, exc_val, exc_tb):
            footer = f"\n===== Run ended {datetime.now().isoformat()} =====\n"
            self_nonlocal.log_file.write(footer)
            self_nonlocal.log_file.flush()
            sys.stdout = self_nonlocal.original_stdout
            sys.stderr = self_nonlocal.original_stderr
            self_nonlocal.log_file.close()

    return _LogContext()


def _stringify_params(params: Dict[str, Any]) -> Dict[str, Any]:
    formatted: Dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, (str, int, float, bool)):
            formatted[key] = value
        elif value is None:
            continue
        else:
            formatted[key] = json.dumps(value, ensure_ascii=False)
    return formatted


def _extract_map(metrics: Any, idx: int) -> Optional[float]:
    if isinstance(metrics, (list, tuple)) and len(metrics) > idx:
        value = metrics[idx]
        if value is not None:
            return float(value)
    return None


def _cleanup_checkpoints(output_dir: Path, keep_best_only: bool) -> None:
    if not keep_best_only:
        return
    keep_names = {"checkpoint_best_total.pth", "checkpoint_best_total.nodp.pth"}
    for checkpoint in output_dir.glob("checkpoint*.pth"):
        if checkpoint.name not in keep_names:
            try:
                checkpoint.unlink()
            except OSError:
                pass


def _build_mlflow_callbacks(log_path: Path, metrics_file: Path, keep_best_only: bool, output_dir: Path):
    state = {
        "best_map_50_95": None,
        "best_map_50": None,
        "best_epoch": None,
        "last_map_50_95": None,
        "last_map_50": None,
        "last_val_loss": None,
    }

    def _log_epoch_metrics(log_stats: Dict[str, Any]):
        epoch = int(log_stats.get("epoch", -1))
        regular_key = "test_coco_eval_bbox" if "test_coco_eval_bbox" in log_stats else "test_coco_eval_masks"
        regular_metrics = log_stats.get(regular_key)
        ema_key = "ema_test_coco_eval_bbox" if "ema_test_coco_eval_bbox" in log_stats else "ema_test_coco_eval_masks"
        ema_metrics = log_stats.get(ema_key)
        regular_map_50_95 = _extract_map(regular_metrics, 0)
        regular_map_50 = _extract_map(regular_metrics, 1)
        ema_map_50_95 = _extract_map(ema_metrics, 0)
        ema_map_50 = _extract_map(ema_metrics, 1)
        val_loss = log_stats.get("test_loss")

        if regular_map_50_95 is not None:
            mlflow.log_metric("map_50_95", regular_map_50_95, step=epoch)
            if regular_map_50 is not None:
                mlflow.log_metric("map_50", regular_map_50, step=epoch)
            state["last_map_50_95"] = regular_map_50_95
            state["last_map_50"] = regular_map_50
        elif ema_map_50_95 is not None:
            state["last_map_50_95"] = ema_map_50_95
            state["last_map_50"] = ema_map_50

        if val_loss is not None:
            state["last_val_loss"] = float(val_loss)

        candidates: Tuple[Tuple[str, Optional[float], Optional[float]], ...] = (
            ("regular", regular_map_50_95, regular_map_50),
            ("ema", ema_map_50_95, ema_map_50),
        )
        for _, map_50_95, map_50 in candidates:
            if map_50_95 is None:
                continue
            if state["best_map_50_95"] is None or map_50_95 > state["best_map_50_95"]:
                state["best_map_50_95"] = map_50_95
                state["best_map_50"] = map_50
                state["best_epoch"] = epoch
                mlflow.log_metric("best_map_50_95", map_50_95, step=epoch)

    def _on_train_end():
        if log_path.exists():
            mlflow.log_artifact(str(log_path), artifact_path="logs")
        metrics_payload = {
            "map_50_95": state["last_map_50_95"],
            "map_50": state["last_map_50"],
            "val_loss": state["last_val_loss"],
            "best_map_50_95": state["best_map_50_95"],
            "best_map_50": state["best_map_50"],
            "best_epoch": state["best_epoch"],
        }
        metrics_file.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(metrics_file), artifact_path="metrics")
        if state["best_map_50_95"] is not None:
            mlflow.log_metric("final_best_map_50_95", state["best_map_50_95"])
        _cleanup_checkpoints(output_dir, keep_best_only)

    return _log_epoch_metrics, _on_train_end


log_file_path = run_output_dir / "log_terminal.txt"
keep_best_only = training_args.get("keep_best_only", DEFAULT_KEEP_BEST_ONLY)
epoch_callback, train_end_callback = _build_mlflow_callbacks(
    log_file_path,
    metrics_path,
    bool(keep_best_only),
    run_output_dir,
)
model.callbacks["on_fit_epoch_end"].append(epoch_callback)
model.callbacks["on_train_end"].append(train_end_callback)

with _log_terminal_output(log_file_path):
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(_stringify_params(training_args))
        model.train(
            dataset_dir=training_args["dataset_dir"],
            epochs=training_args["epochs"],
            batch_size=training_args["batch_size"],
            grad_accum_steps=training_args["grad_accum_steps"],
            lr=training_args["lr"],
            output_dir=str(run_output_dir),
            early_stopping=training_args["early_stopping"],
            early_stopping_patience=30,
            run=training_args["run"],
            # resolution= training_args["resolution"],
            # positional_encoding_size= 1080//12,
            class_names=training_args["class_names"],
            num_classes=training_args["num_classes"],
            resume=DEFAULT_RESUME,
            # eval_max_dets=100,
            run_test=False,

        )
