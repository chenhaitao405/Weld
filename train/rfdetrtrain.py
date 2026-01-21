import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import mlflow
import yaml

from rfdetr import RFDETRSeg2XLarge, RFDETRLarge, RFDETRSegPreview, RFDETRMedium,RFDETRSegXLarge

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

model = RFDETRSeg2XLarge()

DEFAULT_TRAINING_ARGS = {
    "dataset_dir": "/datasets/PAR/Weld/data/pipeline_pair_1120/coco_patch_seg",
    "epochs": 500,
    "batch_size": 2,
    "grad_accum_steps": 8,
    "lr": 1e-4,
    "output_dir": "outputs/RFDETRSeg2XLarge",
    "early_stopping": True,
    "run": "patchresume_1120data",  # mlflow 的 run name
    "resume": "outputs/RFDETRSeg2XLarge/SWRD_PATCH/checkpoint_best_regular.pth",
    "resolution": 0,
    "class_names": CLASS_NAMES,
    "num_classes": len(CLASS_NAMES),
    "metrics_path": "metrics/rfdetr.json",
    "keep_best_only": False,
}

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
                        help=f"Dataset root directory (default: {DEFAULT_TRAINING_ARGS['dataset_dir']})")
    parser.add_argument("--epochs", type=int, default=None,
                        help=f"Number of epochs (default: {DEFAULT_TRAINING_ARGS['epochs']})")
    parser.add_argument("--batch-size", type=int, default=None,
                        help=f"Batch size (default: {DEFAULT_TRAINING_ARGS['batch_size']})")
    parser.add_argument("--grad-accum-steps", type=int, default=None,
                        help=f"Gradient accumulation steps (default: {DEFAULT_TRAINING_ARGS['grad_accum_steps']})")
    parser.add_argument("--lr", type=float, default=None,
                        help=f"Learning rate (default: {DEFAULT_TRAINING_ARGS['lr']})")
    parser.add_argument("--output-dir", type=str, default=None,
                        help=("Output root directory; actual run output will be output_dir/run "
                              f"(default: {DEFAULT_TRAINING_ARGS['output_dir']})"))
    parser.add_argument("--run", type=str, default=None,
                        help=f"Run name, also used as MLflow run name (default: {DEFAULT_TRAINING_ARGS['run']})")
    parser.add_argument("--resume", type=str, default=None,
                        help=f"Resume checkpoint path (default: {DEFAULT_TRAINING_ARGS['resume']})")
    parser.add_argument("--resolution", type=int, default=None,
                        help=f"Input resolution (default: {DEFAULT_TRAINING_ARGS['resolution']})")
    parser.add_argument("--class-names", nargs="+", default=None,
                        help="Space separated list of class names")
    parser.add_argument("--num-classes", type=int, default=None,
                        help="Number of classes; defaults to len(class_names)")
    parser.add_argument("--metrics-path", type=str, default=None,
                        help=f"Metrics output path (default: {DEFAULT_TRAINING_ARGS['metrics_path']})")
    parser.add_argument("--keep-best-only", dest="keep_best_only", action="store_true",
                        help="Keep only checkpoint_best_total.pth in the run output dir")
    parser.add_argument("--keep-all-checkpoints", dest="keep_best_only", action="store_false",
                        help="Keep all checkpoints in the run output dir")
    parser.add_argument("--early-stopping", dest="early_stopping", action="store_true",
                        help="Enable early stopping")
    parser.add_argument("--no-early-stopping", dest="early_stopping", action="store_false",
                        help="Disable early stopping")
    parser.set_defaults(early_stopping=None, keep_best_only=None)

    args = parser.parse_args()
    training_args: Dict[str, Any] = dict(DEFAULT_TRAINING_ARGS)

    if args.params_file:
        params_path = Path(args.params_file).expanduser()
        overrides = _load_params_file(params_path, parser)
        training_args.update(overrides)

    override_keys = [
        "dataset_dir",
        "epochs",
        "batch_size",
        "grad_accum_steps",
        "lr",
        "output_dir",
        "run",
        "resume",
        "resolution",
        "early_stopping",
        "metrics_path",
        "keep_best_only",
    ]

    for key in override_keys:
        value = getattr(args, key)
        if value is not None:
            training_args[key] = value

    if args.class_names is not None:
        training_args["class_names"] = args.class_names
    else:
        existing = training_args.get("class_names", CLASS_NAMES)
        training_args["class_names"] = list(existing)

    if args.num_classes is not None:
        training_args["num_classes"] = args.num_classes
    else:
        training_args["num_classes"] = training_args.get("num_classes", len(training_args["class_names"]))
        if training_args["num_classes"] is None:
            training_args["num_classes"] = len(training_args["class_names"])

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
keep_best_only = training_args.get("keep_best_only")
if keep_best_only is None:
    keep_best_only = DEFAULT_TRAINING_ARGS["keep_best_only"]
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
            run=training_args["run"],
            # resolution= 1080,
            # positional_encoding_size= 1080//12,
            class_names=training_args["class_names"],
            num_classes=training_args["num_classes"],
            resume=training_args["resume"],
            # eval_max_dets=100,
            run_test=False,

        )
