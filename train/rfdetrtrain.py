import json
import sys
from pathlib import Path
from datetime import datetime

from rfdetr import RFDETRBase, RFDETRLarge, RFDETRSegPreview

model = RFDETRLarge()

training_args = {
    "dataset_dir": "/datasets/PAR/Xray/datasets_merge/slice3_ratio_1",
    "epochs": 500,
    "batch_size": 4,
    "grad_accum_steps": 8,
    "lr": 1e-4,
    "output_dir": "./runs/1208/detrlarge/slice3mixed1",
    "early_stopping": True,
    "run":"4batch",
    # "resume" : "runs/1208/detrlarge/SWRD_patch640_res560/checkpoint_best_regular.pth",
    # "resolution":1120

}

output_dir = Path(training_args["output_dir"]).resolve()
output_dir.mkdir(parents=True, exist_ok=True)
training_args["output_dir"] = str(output_dir)

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


log_file_path = output_dir / "log_terminal.txt"

with _log_terminal_output(log_file_path):
    model.train(
        dataset_dir=training_args["dataset_dir"],
        epochs=training_args["epochs"],
        batch_size=training_args["batch_size"],
        grad_accum_steps=training_args["grad_accum_steps"],
        lr=training_args["lr"],
        output_dir=str(output_dir),
        early_stopping=training_args["early_stopping"],
        run = training_args["run"],
        # resolution = training_args["resolution"]
        # resume = training_args["resume"],
    )

    config_path = output_dir / "train_params.json"
    config_path.write_text(json.dumps(training_args, ensure_ascii=False, indent=2), encoding="utf-8")
