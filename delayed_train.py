#!/usr/bin/env python3
"""
延迟执行训练脚本
在3小时后自动运行指定的训练命令
"""

import subprocess
import time
from datetime import datetime, timedelta


def main():
    # 延迟时间：3小时 = 3 * 60 * 60 秒
    delay_hours = 3
    delay_seconds = delay_hours * 60 * 60

    # 要执行的命令
    command = [
        "python", "./train/crackseg.py",
        "--data", "/home/lenovo/code/CHT/datasets/Xray/self/1120/labeled/roi2_merge/patchROTATE/dataset.yaml",
        "--name", "rotateroi2_11Xmerge_pretrian",
        "--batch", "16",
        "--model", "yolo11x-seg.yaml",
        "--epochs", "1000"
    ]

    # 计算预计执行时间
    start_time = datetime.now()
    execute_time = start_time + timedelta(hours=delay_hours)

    print("=" * 60)
    print("延迟训练脚本已启动")
    print("=" * 60)
    print(f"当前时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"延迟时间: {delay_hours} 小时")
    print(f"预计执行时间: {execute_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n将执行的命令:\n{' '.join(command)}")
    print("=" * 60)
    print("\n等待中... (可以使用 Ctrl+C 取消)\n")

    try:
        # 每分钟显示一次剩余时间
        remaining = delay_seconds
        while remaining > 0:
            hours = remaining // 3600
            minutes = (remaining % 3600) // 60
            seconds = remaining % 60
            print(f"\r剩余时间: {hours:02d}:{minutes:02d}:{seconds:02d}", end="", flush=True)

            # 睡眠1分钟或剩余时间（取较小值）
            sleep_time = min(60, remaining)
            time.sleep(sleep_time)
            remaining -= sleep_time

        print("\n\n" + "=" * 60)
        print(f"开始执行训练命令 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60 + "\n")

        # 执行命令
        result = subprocess.run(command, check=False)

        print("\n" + "=" * 60)
        print(f"训练完成 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"返回码: {result.returncode}")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\n已取消执行！")
        return 1

    return result.returncode


if __name__ == "__main__":
    exit(main())