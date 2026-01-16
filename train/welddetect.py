import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Optional

import mlflow
from ultralytics import YOLO


def _parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "t"}:
        return True
    if text in {"0", "false", "no", "n", "f"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def _float_or_none(value) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN check
        return None
    return number


def _load_metrics_from_csv(results_csv: Path) -> Dict[str, float]:
    if not results_csv.exists():
        return {}

    with results_csv.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        return {}

    last = rows[-1]
    map50 = _float_or_none(last.get("metrics/mAP50(B)") or last.get("metrics/mAP50"))
    map50_95 = _float_or_none(last.get("metrics/mAP50-95(B)") or last.get("metrics/mAP50-95"))
    val_box = _float_or_none(last.get("val/box_loss"))
    val_cls = _float_or_none(last.get("val/cls_loss"))
    val_dfl = _float_or_none(last.get("val/dfl_loss"))

    losses = [value for value in (val_box, val_cls, val_dfl) if value is not None]
    val_loss = sum(losses) if losses else None

    metrics: Dict[str, float] = {}
    if map50 is not None:
        metrics["mAP50"] = map50
    if map50_95 is not None:
        metrics["mAP50-95"] = map50_95
    if val_loss is not None:
        metrics["val_loss"] = val_loss

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Weld crack detection training (YOLO)")
    parser.add_argument("--data", required=True, help="Path to dataset.yaml")
    parser.add_argument("--epochs", type=int, default=500, help="Number of epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate (lr0)")
    parser.add_argument("--output-dir", type=str, default="outputs/welddetect",
                        help="Root directory for training outputs")
    parser.add_argument("--run", type=str, default="crack_baseline", help="Run name")
    parser.add_argument("--model", type=str, default="yolo11l.pt", help="Model checkpoint or yaml")
    parser.add_argument("--pretrained", type=str, default="true",
                        help="Whether to use pretrained weights (true/false)")
    parser.add_argument("--degrees", type=float, default=180, help="Rotation augmentation degrees")

    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir / args.run

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr,
        project=str(output_dir),
        name=args.run,
        exist_ok=True,
        pretrained=_parse_bool(args.pretrained),
        degrees=args.degrees,
    )

    weights_dir = run_dir / "weights"
    best_path = weights_dir / "best.pt"
    last_path = weights_dir / "last.pt"
    if last_path.exists():
        last_path.unlink()

    metrics = _load_metrics_from_csv(run_dir / "results.csv")
    metrics_path = Path("metrics") / "welddetect.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    mlruns_dir = Path(__file__).resolve().parent.parent / "mlruns"
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"file:{mlruns_dir}")
    mlflow.set_experiment("welddetect")

    with mlflow.start_run(run_name=args.run):
        mlflow.log_params({
            "data": args.data,
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "lr0": args.lr,
            "output_dir": str(output_dir),
            "run": args.run,
            "model": args.model,
            "pretrained": _parse_bool(args.pretrained),
            "degrees": args.degrees,
        })
        if metrics:
            mlflow.log_metrics(metrics)
        if metrics_path.exists():
            mlflow.log_artifact(str(metrics_path), artifact_path="metrics")
        if best_path.exists():
            mlflow.log_artifact(str(best_path), artifact_path="weights")


if __name__ == "__main__":
    main()
