"""
性能分析工具 - 用于诊断处理瓶颈
"""

import time
import cv2
import numpy as np
from pathlib import Path
from contextlib import contextmanager
from typing import Dict
import os


class PerformanceProfiler:
    """性能分析器"""

    def __init__(self):
        self.timings = {}

    @contextmanager
    def timer(self, name: str):
        """计时上下文管理器"""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            if name not in self.timings:
                self.timings[name] = []
            self.timings[name].append(elapsed)

    def report(self):
        """打印性能报告"""
        print("\n" + "=" * 60)
        print("性能分析报告")
        print("=" * 60)

        total_time = sum(sum(times) for times in self.timings.values())

        for name, times in sorted(self.timings.items(),
                                  key=lambda x: sum(x[1]), reverse=True):
            avg_time = sum(times) / len(times)
            total = sum(times)
            percentage = (total / total_time * 100) if total_time > 0 else 0

            print(f"{name:30} | 总计: {total:8.2f}s | 平均: {avg_time:6.3f}s | 占比: {percentage:5.1f}%")

        print("=" * 60)
        print(f"总耗时: {total_time:.2f}s")
        print("=" * 60)


def test_performance(image_path: str, output_dir: str = "./test_output"):
    """测试各个处理步骤的性能"""

    profiler = PerformanceProfiler()
    Path(output_dir).mkdir(exist_ok=True)

    # 1. 读取图像
    with profiler.timer("1. 读取TIF图像"):
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    print(f"图像信息: shape={image.shape}, dtype={image.dtype}")

    # 2. 滑动窗口裁剪
    patches = []
    with profiler.timer("2. 滑动窗口裁剪"):
        h, w = image.shape[:2]
        window_size = 512
        stride = 256

        for y in range(0, h - window_size + 1, stride):
            for x in range(0, w - window_size + 1, stride):
                patch = image[y:y + window_size, x:x + window_size]
                patches.append(patch)

    print(f"生成了 {len(patches)} 个patches")

    # 只测试前10个patches
    test_patches = patches[:10]

    # 3. 图像增强测试
    for i, patch in enumerate(test_patches):
        with profiler.timer("3. 图像增强(CLAHE)"):
            if patch.dtype == np.uint16:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(patch)
            else:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(patch)

    # 4. 16位直方图均衡测试
    if image.dtype == np.uint16:
        for i, patch in enumerate(test_patches):
            with profiler.timer("4. 16位直方图均衡(4096bins)"):
                hist, _ = np.histogram(patch.flatten(), bins=4096, range=[0, 65536])
                cdf = hist.cumsum()
                cdf_normalized = ((cdf - cdf.min()) * 65535 /
                                  (cdf.max() - cdf.min())).astype(np.uint16)

    # 5. PNG保存测试 - 不同压缩级别
    for compression in [0, 1, 3, 6]:
        for i, patch in enumerate(test_patches[:3]):
            output_path = Path(output_dir) / f"test_{compression}_{i}.png"
            with profiler.timer(f"5. PNG保存(压缩={compression})"):
                cv2.imwrite(str(output_path), patch,
                            [cv2.IMWRITE_PNG_COMPRESSION, compression])

    # 6. TIFF保存测试（作为对比）
    for i, patch in enumerate(test_patches[:3]):
        output_path = Path(output_dir) / f"test_{i}.tif"
        with profiler.timer("6. TIFF保存"):
            cv2.imwrite(str(output_path), patch)

    # 7. 内存拷贝测试
    for i, patch in enumerate(test_patches):
        with profiler.timer("7. 内存拷贝"):
            _ = patch.copy()

    # 8. 标注处理测试
    for i in range(10):
        with profiler.timer("8. 标注处理"):
            # 模拟标注处理
            points = np.random.rand(10, 2) * 512
            for j in range(len(points)):
                x, y = points[j]
                new_x = max(0, min(x, 511))
                new_y = max(0, min(y, 511))

    # 生成报告
    profiler.report()

    # 清理测试文件
    import shutil
    shutil.rmtree(output_dir)


def profile_batch_vs_single():
    """对比批量处理vs单个处理的性能"""
    print("\n批量处理 vs 单个处理性能对比")
    print("=" * 60)

    # 创建测试数据
    images = [np.random.randint(0, 65536, (512, 512), dtype=np.uint16)
              for _ in range(10)]

    # 单个处理
    start = time.perf_counter()
    for img in images:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        _ = clahe.apply(img)
    single_time = time.perf_counter() - start

    # 批量处理（模拟）
    start = time.perf_counter()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    for img in images:
        _ = clahe.apply(img)
    batch_time = time.perf_counter() - start

    print(f"单个处理: {single_time:.3f}s")
    print(f"批量处理: {batch_time:.3f}s")
    print(f"加速比: {single_time / batch_time:.2f}x")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("使用方法: python performance_profiler.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    print("开始性能分析...")
    test_performance(image_path)

    print("\n" + "=" * 60)
    profile_batch_vs_single()