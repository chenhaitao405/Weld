import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Dict

import mlflow

from rfdetr import RFDETRBase, RFDETRLarge, RFDETRSegPreview,RFDETRMedium

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

model = RFDETRBase()

DEFAULT_TRAINING_ARGS = {
    "dataset_dir": "/datasets/PAR/Xray/self/1208/roi3_1215/coco_merged",
    "epochs": 500,
    "batch_size": 2,
    "grad_accum_steps": 8,
    "lr": 1e-4,
    "output_dir": "/datasets/PAR/detr_runs/1230/detrmedium",
    "early_stopping": True,
    "run": "patch_resume",  # mlflow 的 run name
    "resume" : "/datasets/PAR/detr_runs/1230/detrmedium/merge_patch/checkpoint_best_total.pth",
    "resolution": 840,
    # "lr_scheduler": "cosine",
    # "warmup_epochs": 5,
    # "lr_min_factor": 0.05,
    "class_names": CLASS_NAMES,
    "num_classes": len(CLASS_NAMES)
}


def _load_training_args() -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description="RF-DETR training entrypoint")
    parser.add_argument("--params-file", type=str, help="Path to JSON file overriding training args")
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
    parser.add_argument("--resolution", type=int, default=None,
                        help=f"Input resolution (default: {DEFAULT_TRAINING_ARGS['resolution']})")
    # parser.add_argument("--lr-scheduler", type=str, default=None,
    #                     help=f"LR scheduler type (default: {DEFAULT_TRAINING_ARGS['lr_scheduler']})")
    # parser.add_argument("--warmup-epochs", type=int, default=None,
    #                     help=f"Warmup epochs (default: {DEFAULT_TRAINING_ARGS['warmup_epochs']})")
    # parser.add_argument("--lr-min-factor", type=float, default=None,
    #                     help=f"LR minimum factor (default: {DEFAULT_TRAINING_ARGS['lr_min_factor']})")
    parser.add_argument("--class-names", nargs="+", default=None,
                        help="Space separated list of class names")
    parser.add_argument("--num-classes", type=int, default=CLASS_NAMES,
                        help="Number of classes; defaults to len(class_names)")
    parser.add_argument("--early-stopping", dest="early_stopping", action="store_true",
                        help="Enable early stopping")
    parser.add_argument("--no-early-stopping", dest="early_stopping", action="store_false",
                        help="Disable early stopping")
    parser.set_defaults(early_stopping=None)

    args = parser.parse_args()
    training_args: Dict[str, Any] = dict(DEFAULT_TRAINING_ARGS)

    if args.params_file:
        params_path = Path(args.params_file).expanduser()
        if not params_path.exists():
            parser.error(f"Params file {params_path} not found")
        try:
            overrides = json.loads(params_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            parser.error(f"Params file {params_path} is not valid JSON: {exc}")
        if not isinstance(overrides, dict):
            parser.error("Params file must contain a JSON object")
        training_args.update(overrides)

    override_keys = [
        "dataset_dir",
        "epochs",
        "batch_size",
        "grad_accum_steps",
        "lr",
        "output_dir",
        "run",
        "resolution",
        # "lr_scheduler",
        # "warmup_epochs",
        # "lr_min_factor",
        "early_stopping",
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
output_root = Path(training_args["output_dir"]).expanduser().resolve()
run_name = training_args["run"]
run_output_dir = output_root / run_name
run_output_dir.mkdir(parents=True, exist_ok=True)
training_args["output_dir"] = str(output_root)
training_args["run_output_dir"] = str(run_output_dir)
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


def _build_mlflow_callbacks(log_path: Path):
    state = {"best_map": -1.0}

    def _log_epoch_metrics(log_stats: Dict[str, Any]):
        epoch = int(log_stats.get("epoch", -1))
        coco_key = "test_coco_eval_bbox" if "test_coco_eval_bbox" in log_stats else "test_coco_eval_masks"
        metrics = log_stats.get(coco_key)
        if isinstance(metrics, (list, tuple)):
            metrics_list = list(metrics)
            if metrics_list:
                map_50_95 = float(metrics_list[0])
                mlflow.log_metric("map_50_95", map_50_95, step=epoch)
                if len(metrics_list) > 1:
                    mlflow.log_metric("map_50", float(metrics_list[1]), step=epoch)
                if map_50_95 > state["best_map"]:
                    state["best_map"] = map_50_95
                    mlflow.log_metric("best_map_50_95", map_50_95, step=epoch)

    def _on_train_end():
        if log_path.exists():
            mlflow.log_artifact(str(log_path), artifact_path="logs")
        if state["best_map"] >= 0:
            mlflow.log_metric("final_best_map_50_95", state["best_map"])

    return _log_epoch_metrics, _on_train_end


log_file_path = run_output_dir / "log_terminal.txt"
epoch_callback, train_end_callback = _build_mlflow_callbacks(log_file_path)
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
            resolution=training_args["resolution"],
            # lr_scheduler=training_args["lr_scheduler"],
            # warmup_epochs=training_args["warmup_epochs"],
            # lr_min_factor=training_args["lr_min_factor"],
            class_names=training_args["class_names"],
            num_classes=training_args["num_classes"],
            resume = training_args["resume"],
        )
