#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Convenience wrapper to run inference → validation → visualization sequentially."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path
from typing import List, Optional


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


def main() -> None:
    args = parse_args()
    selected_steps = parse_steps(args.steps)
    base_path = Path(args.base_path).resolve()
    base_path.mkdir(parents=True, exist_ok=True)
    infer_dir = base_path / args.infer_subdir
    valid_dir = base_path / args.valid_subdir
    report_path = valid_dir / "report.html"
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

    visualize_cmd = [
        "python", "visualize_validation_results.py",
        "--validation-dir", str(valid_dir),
        "--output-html", str(report_path)
    ]
    visualize_cmd += visualize_opts

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

    if '1' in selected_steps:
        run_stage(run_infer_cmd, args.dry_run)
    else:
        print("\n[SKIP] 步骤1 推理 - 未选择执行")

    if '2' in selected_steps:
        run_stage(validate_cmd, args.dry_run)
    else:
        print("\n[SKIP] 步骤2 验证 - 未选择执行")

    if '3' in selected_steps:
        run_stage(visualize_cmd, args.dry_run)
    else:
        print("\n[SKIP] 步骤3 可视化 - 未选择执行")
    print("\n流水线已完成。推理在:", infer_dir)
    print("验证与可视化在:", valid_dir)
    print("HTML 报告:", report_path)


if __name__ == "__main__":
    main()
