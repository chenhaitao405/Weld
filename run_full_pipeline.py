#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Convenience wrapper to run inference → validation → visualization sequentially."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional

try:  # pragma: no cover - optional dependency loaded lazily
    import mlflow  # type: ignore
except ImportError:  # pragma: no cover
    mlflow = None  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="串联 run_inference_pipeline、validate_inference_results、visualize_validation_results 的脚本"
    )
    parser.add_argument("--base-path", default="outputs/pipeline_run",
                        help="三个阶段产物的根目录，默认 outputs/pipeline_run")
    parser.add_argument("--infer-subdir", default="infer",
                        help="推理结果子目录（相对base-path），默认 infer")
    parser.add_argument("--valid-subdir", default="valid",
                        help="验证与可视化输出子目录（相对base-path），默认 valid")
    parser.add_argument("--inference-results", default="inference_results.json",
                        help="推理结果JSON文件名，默认 inference_results.json")
    parser.add_argument("--run-inference-opts", default="", help="追加到 run_inference_pipeline.py 的参数字符串")
    parser.add_argument("--validate-opts", default="", help="追加到 validate_inference_results.py 的参数字符串")
    parser.add_argument("--visualize-opts", default="", help="追加到 visualize_validation_results.py 的参数字符串")
    parser.add_argument("--steps", default="123",
                        help="选择性执行步骤：1=推理 2=验证 3=可视化，默认123全跑")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅打印命令但不执行")
    parser.add_argument("--mlflow", action="store_true",
                        help="启用 MLflow 记录参数/指标/产物")
    parser.add_argument("--mlflow-experiment", default="weld_pipeline",
                        help="MLflow 实验名称，默认 weld_pipeline")
    parser.add_argument("--mlflow-run-name",
                        help="可选，指定 MLflow run name，默认自动推断")
    parser.add_argument("--mlflow-tracking-uri",
                        help="可选，设置 MLflow tracking URI")
    parser.add_argument("--mlflow-tags", nargs="*", default=[],
                        help="附加到 MLflow run 的标签，格式 key=value")
    return parser.parse_args()


def split_opts(option_str: str) -> List[str]:
    option_str = option_str.strip()
    return shlex.split(option_str) if option_str else []


def ensure_flag(opts: List[str], flag: str) -> bool:
    return flag in opts


def get_flag_value(cmd: List[str], flag: str) -> Optional[str]:
    for idx, token in enumerate(cmd):
        if token == flag and idx + 1 < len(cmd):
            return cmd[idx + 1]
    return None


def parse_steps(steps_str: str) -> List[str]:
    valid = {'1', '2', '3'}
    steps = []
    for ch in steps_str:
        if ch not in valid:
            raise ValueError(f"无效步骤: {ch}，仅支持1/2/3")
        if ch not in steps:
            steps.append(ch)
    if not steps:
        raise ValueError("需要至少选择一个步骤")
    return steps


def run_stage(cmd: List[str], dry_run: bool) -> None:
    print("\n[CMD]", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def parse_kv_pairs(entries: List[str]) -> Dict[str, str]:
    tags: Dict[str, str] = {}
    for entry in entries:
        if '=' not in entry:
            raise ValueError(f"MLflow 标签项 '{entry}' 缺少 '=' 分隔符，示例：stage=baseline")
        key, value = entry.split('=', 1)
        key = key.strip()
        if not key:
            raise ValueError("MLflow 标签键不能为空")
        tags[key] = value.strip()
    return tags


def slugify(value: str, fallback: str) -> str:
    token = re.sub(r"[^0-9A-Za-z]+", "_", value)
    token = token.strip('_').lower()
    return token or fallback


def log_pipeline_params(payload: Dict[str, Any]) -> None:
    safe_params = {}
    for key, value in payload.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            safe_params[key] = str(value)
        else:
            safe_params[key] = json.dumps(value, ensure_ascii=False)
    if safe_params:
        mlflow.log_params(safe_params)


def log_artifact_if_exists(path: Path, artifact_path: str) -> None:
    if path.exists():
        mlflow.log_artifact(str(path), artifact_path=artifact_path)


def log_inference_summary(infer_json: Path) -> None:
    if not infer_json.exists():
        return
    with open(infer_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    mode = data.get("mode")
    results = data.get("results", [])
    mlflow.log_metric("inference_image_count", len(results))
    if mode is not None:
        mlflow.set_tag("inference_mode", str(mode))


def log_validation_summary(summary_path: Path) -> None:
    if not summary_path.exists():
        return
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    overall = summary.get("overall", {})
    metrics: Dict[str, float] = {}
    for key in ("defect_precision", "defect_recall", "classification_accuracy"):
        value = overall.get(key)
        if isinstance(value, (int, float)):
            metrics[f"validation_{key}"] = float(value)
    if metrics:
        mlflow.log_metrics(metrics)
    per_class_metrics: Dict[str, float] = {}
    for idx, class_metric in enumerate(summary.get("per_class", [])):
        class_name = class_metric.get("class_name") or f"class_{idx}"
        slug = slugify(str(class_name), f"class_{idx}")
        recall = class_metric.get("defect_recall")
        if isinstance(recall, (int, float)):
            per_class_metrics[f"{slug}_recall"] = float(recall)
    if per_class_metrics:
        mlflow.log_metrics(per_class_metrics)


def execute_stage(stage_key: str,
                  cmd: List[str],
                  dry_run: bool,
                  enable_mlflow: bool) -> bool:
    success = False
    start = time.perf_counter()
    lowercase_key = stage_key.lower()
    ctx = mlflow.start_run(run_name=f"{stage_key.title()}Stage", nested=True) if enable_mlflow else nullcontext()
    with ctx:
        try:
            run_stage(cmd, dry_run)
            success = True
            return True
        finally:
            duration = time.perf_counter() - start
            if enable_mlflow:
                mlflow.log_metric(f"{lowercase_key}_duration_sec", duration)
                mlflow.log_metric(f"{lowercase_key}_success", 1 if success else 0)
                mlflow.set_tag(f"{lowercase_key}_command", " ".join(cmd))


def main() -> None:
    args = parse_args()
    selected_steps = parse_steps(args.steps)
    base_path = Path(args.base_path).resolve()
    base_path.mkdir(parents=True, exist_ok=True)
    infer_dir = base_path / args.infer_subdir
    valid_dir = base_path / args.valid_subdir
    infer_dir.mkdir(parents=True, exist_ok=True)
    valid_dir.mkdir(parents=True, exist_ok=True)

    inference_results_path = infer_dir / args.inference_results

    run_infer_opts = split_opts(args.run_inference_opts)
    validate_opts = split_opts(args.validate_opts)
    visualize_opts = split_opts(args.visualize_opts)

    run_infer_cmd = ["python", "run_inference_pipeline.py"]
    if not ensure_flag(run_infer_opts, "--output-dir"):
        run_infer_cmd += ["--output-dir", str(infer_dir)]
    if not ensure_flag(run_infer_opts, "--results-json"):
        run_infer_cmd += ["--results-json", args.inference_results]
    run_infer_cmd += run_infer_opts

    output_dir_value = get_flag_value(run_infer_cmd, "--output-dir")
    infer_output_dir = Path(output_dir_value).resolve() if output_dir_value else infer_dir
    infer_output_dir.mkdir(parents=True, exist_ok=True)
    infer_dir = infer_output_dir
    results_name = get_flag_value(run_infer_cmd, "--results-json") or args.inference_results
    results_path_candidate = Path(results_name)
    if results_path_candidate.is_absolute():
        inference_results_path = results_path_candidate
    else:
        inference_results_path = infer_output_dir / results_path_candidate

    image_root = get_flag_value(run_infer_cmd, "--image-dir")
    if image_root is None:
        raise ValueError("必须在 --run-inference-opts 中指定 --image-dir，用于 validate 阶段的 --image-root")

    validate_cmd = ["python", "validate_inference_results.py"]
    if not ensure_flag(validate_opts, "--inference-json"):
        validate_cmd += ["--inference-json", str(inference_results_path)]
    if not ensure_flag(validate_opts, "--output-dir"):
        validate_cmd += ["--output-dir", str(valid_dir)]
    if not ensure_flag(validate_opts, "--image-root"):
        validate_cmd += ["--image-root", image_root]
    validate_cmd += validate_opts

    validate_output_dir_value = get_flag_value(validate_cmd, "--output-dir")
    if validate_output_dir_value:
        valid_dir = Path(validate_output_dir_value).resolve()
    valid_dir.mkdir(parents=True, exist_ok=True)
    report_path = valid_dir / "report.html"

    visualize_cmd = ["python", "visualize_validation_results.py"]
    if not ensure_flag(visualize_opts, "--validation-dir"):
        visualize_cmd += ["--validation-dir", str(valid_dir)]
    if not ensure_flag(visualize_opts, "--output-html"):
        visualize_cmd += ["--output-html", str(report_path)]
    visualize_cmd += visualize_opts
    output_html_arg = get_flag_value(visualize_cmd, "--output-html")
    if output_html_arg:
        report_path = Path(output_html_arg).expanduser().resolve()

    config_path = base_path / "pipeline_args.json"
    config_payload = vars(args).copy()
    config_payload["selected_steps"] = selected_steps
    config_payload["base_path"] = str(base_path)
    config_payload["infer_dir"] = str(infer_dir)
    config_payload["valid_dir"] = str(valid_dir)
    config_payload["report_path"] = str(report_path)
    config_payload["image_root"] = image_root
    config_payload["inference_json"] = str(inference_results_path)
    config_path.write_text(json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    mlflow_enabled = args.mlflow
    mlflow_tags: Dict[str, str] = {}
    if mlflow_enabled and mlflow is None:
        raise RuntimeError("检测到 --mlflow 但当前环境未安装 mlflow，请先 `pip install mlflow` 再运行。")
    if mlflow_enabled:
        mlflow_tags = parse_kv_pairs(args.mlflow_tags)
        if args.mlflow_tracking_uri:
            mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        mlflow.set_experiment(args.mlflow_experiment)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    image_root_name = Path(image_root).name if image_root else None
    default_run_name = args.mlflow_run_name or f"{image_root_name or base_path.name}_{timestamp}"
    run_ctx = mlflow.start_run(run_name=default_run_name, tags=mlflow_tags or None) if mlflow_enabled else nullcontext()

    summary_path = valid_dir / "metrics_summary.json"
    manifest_path = valid_dir / "data" / "manifest.json"
    per_class_plot = valid_dir / "per_class_metrics.png"

    with run_ctx:
        pipeline_params: Dict[str, Any] = {
            "base_path": str(base_path),
            "infer_subdir": args.infer_subdir,
            "valid_subdir": args.valid_subdir,
            "selected_steps": ''.join(selected_steps),
            "inference_results": args.inference_results,
            "dry_run": args.dry_run,
        }
        if args.run_inference_opts:
            pipeline_params["run_inference_opts"] = args.run_inference_opts
        if args.validate_opts:
            pipeline_params["validate_opts"] = args.validate_opts
        if args.visualize_opts:
            pipeline_params["visualize_opts"] = args.visualize_opts

        if mlflow_enabled:
            mlflow.set_tag("pipeline_status", "running")
            mlflow.set_tag("image_root", image_root)
            log_pipeline_params(pipeline_params)
            log_artifact_if_exists(config_path, "pipeline")

        try:
            if '1' in selected_steps:
                infer_success = execute_stage("inference", run_infer_cmd, args.dry_run, mlflow_enabled)
                if infer_success and mlflow_enabled and not args.dry_run:
                    log_artifact_if_exists(inference_results_path, "inference")
                    log_inference_summary(inference_results_path)
            else:
                print("\n[SKIP] 步骤1 推理 - 未选择执行")

            if '2' in selected_steps:
                validate_success = execute_stage("validation", validate_cmd, args.dry_run, mlflow_enabled)
                if validate_success and mlflow_enabled and not args.dry_run:
                    log_artifact_if_exists(summary_path, "validation")
                    log_artifact_if_exists(manifest_path, "validation")
                    log_artifact_if_exists(per_class_plot, "validation")
                    log_validation_summary(summary_path)
            else:
                print("\n[SKIP] 步骤2 验证 - 未选择执行")

            if '3' in selected_steps:
                visualize_success = execute_stage("visualization", visualize_cmd, args.dry_run, mlflow_enabled)
                if visualize_success and mlflow_enabled and not args.dry_run:
                    log_artifact_if_exists(report_path, "visualization")
            else:
                print("\n[SKIP] 步骤3 可视化 - 未选择执行")
        except Exception:
            if mlflow_enabled:
                mlflow.set_tag("pipeline_status", "failed")
            raise
        else:
            if mlflow_enabled:
                mlflow.set_tag("pipeline_status", "success")

    print("\n流水线已完成。推理在:", infer_dir)
    print("验证与可视化在:", valid_dir)
    print("HTML 报告:", report_path)


if __name__ == "__main__":
    main()
