"""
图像处理工具模块 - 性能优化版
提供图像增强、滑动窗口等图像处理功能
支持16位图像的输入和输出，保持处理精度
所有增强函数输出单通道图像，适合X射线等灰度图像处理
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from numba import jit


def apply_window_level(image: np.ndarray, window_width: int,
                       window_level: int, output_bits: int = 8) -> np.ndarray:
    """
    应用窗宽窗位变换

    Args:
        image: 输入图像（float32）
        window_width: 窗宽
        window_level: 窗位
        output_bits: 输出位深度（8或16）

    Returns:
        变换后的图像
    """
    window_min = window_level - window_width / 2
    window_max = window_level + window_width / 2

    if output_bits == 8:
        max_val = 255
        dtype = np.uint8
    else:
        max_val = 65535
        dtype = np.uint16

    # 向量化操作，避免使用mask
    output = np.clip(
        (image - window_min) / window_width * max_val,
        0, max_val
    ).astype(dtype)

    return output


def auto_window_level(image: np.ndarray) -> Tuple[int, int]:
    """
    自动计算窗宽窗位（基于统计信息）- 优化版

    Args:
        image: 输入图像

    Returns:
        (window_width, window_level)
    """
    # 使用百分位数代替全范围统计，更稳定且快速
    percentiles = np.percentile(image, [2, 98])
    img_min, img_max = percentiles[0], percentiles[1]
    img_mean = np.mean(image)
    img_std = np.std(image)

    # 使用均值作为窗位，4倍标准差作为窗宽
    window_level = int(img_mean)
    window_width = int(min(4 * img_std, img_max - img_min))

    # 确保窗宽至少为1
    window_width = max(1, window_width)

    return window_width, window_level


def apply_clahe_16bit(image: np.ndarray, clip_limit: float = 2.0,
                      tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    对16位图像应用CLAHE（自适应直方图均衡）

    Args:
        image: 输入图像（uint8或uint16）
        clip_limit: CLAHE的剪裁限制
        tile_grid_size: 网格大小

    Returns:
        处理后的图像（保持原始位深度）
    """
    if image.dtype == np.uint16:
        # OpenCV的CLAHE可以直接处理16位图像
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(image)
    elif image.dtype == np.uint8:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(image)
    else:
        raise ValueError(f"Unsupported image dtype: {image.dtype}")


def enhance_image_windowing(image: np.ndarray, output_bits: int = 8) -> np.ndarray:
    """
    使用窗宽窗位方法增强图像

    Args:
        image: 输入图像（可能是16位）
        output_bits: 输出位深度（8或16）

    Returns:
        增强后的单通道图像
    """
    # 如果是3通道图像，转换为单通道
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 直接处理，避免重复类型转换
    img_float = image.astype(np.float32, copy=False)

    # 自动计算窗宽窗位
    window_width, window_level = auto_window_level(img_float)

    # 应用窗宽窗位变换
    enhanced = apply_window_level(img_float, window_width, window_level, output_bits)

    # CLAHE对比度增强
    image_clahe = apply_clahe_16bit(enhanced, clip_limit=2.0, tile_grid_size=(8, 8))

    return image_clahe


def enhance_image_original(image: np.ndarray, output_bits: int = 8) -> np.ndarray:
    """
    原始图像增强处理方法

    Args:
        image: 输入图像（可能是16位）
        output_bits: 输出位深度（8或16）

    Returns:
        增强后的单通道图像
    """
    # 如果是3通道图像，转换为单通道
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 根据输出位深度处理
    if output_bits == 16:
        if image.dtype == np.uint8:
            # 8位转16位，扩展动态范围
            image_processed = (image.astype(np.uint16) * 257)  # 257 = 65535/255
        else:
            image_processed = image.astype(np.uint16, copy=False)

        # 16位直方图均衡 - 使用优化版本
        image_equalized = histogram_equalize_16bit_fast(image_processed)
    else:
        # 转换为8位
        if image.dtype == np.uint16:
            image_processed = (image / 256).astype(np.uint8)
        else:
            image_processed = image.astype(np.uint8, copy=False)

        # 8位直方图均衡
        image_equalized = cv2.equalizeHist(image_processed)

    # CLAHE处理
    image_clahe = apply_clahe_16bit(image_equalized, clip_limit=2.0, tile_grid_size=(8, 8))

    return image_clahe


def histogram_equalize_16bit_fast(image: np.ndarray, num_bins: int = 4096) -> np.ndarray:
    """
    优化的16位图像直方图均衡化
    使用较少的bins来加速计算，同时保持良好的效果

    Args:
        image: 16位灰度图像
        num_bins: 直方图的bin数量（默认4096，比65536快很多）

    Returns:
        均衡化后的16位图像
    """
    # 计算缩放因子
    scale_factor = 65536 / num_bins

    # 将图像缩放到0到num_bins-1范围
    scaled_image = (image / scale_factor).astype(np.uint16)

    # 计算直方图
    hist, _ = np.histogram(scaled_image.flatten(), bins=num_bins, range=[0, num_bins])

    # 计算累积分布函数
    cdf = hist.cumsum()

    # 归一化CDF
    cdf_normalized = ((cdf - cdf.min()) * 65535 /
                     (cdf.max() - cdf.min())).astype(np.uint16)

    # 应用均衡化
    image_equalized = cdf_normalized[scaled_image.flatten()].reshape(image.shape)

    return image_equalized


def histogram_equalize_16bit(image: np.ndarray) -> np.ndarray:
    """
    16位图像的直方图均衡化（保留原版供选择）

    Args:
        image: 16位灰度图像

    Returns:
        均衡化后的16位图像
    """
    # 使用优化版本
    return histogram_equalize_16bit_fast(image)


def enhance_image(image: np.ndarray, mode: str = 'original', output_bits: int = 8) -> np.ndarray:
    """
    统一的图像增强接口

    Args:
        image: 输入图像（单通道或3通道）
        mode: 增强模式 ('original' 或 'windowing')
        output_bits: 输出位深度（8或16）

    Returns:
        增强后的单通道图像
    """
    if mode == 'windowing':
        return enhance_image_windowing(image, output_bits)
    else:
        return enhance_image_original(image, output_bits)


def sliding_window_crop_batch(image: np.ndarray, window_size: Tuple[int, int],
                              stride: Tuple[int, int]) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    批量滑动窗口裁剪 - 优化版本，返回numpy数组以便批量处理
    修复：1. 图像单一维度小于窗口尺寸 2. 确保所有patch尺寸与窗口一致（不足部分填充）

    Args:
        image: 输入图像
        window_size: 窗口大小 (height, width)
        stride: 滑动步长 (y_stride, x_stride)

    Returns:
        形状为 (n_patches, window_h, window_w, channels) 的数组 + 位置列表
    """
    h, w = image.shape[:2]
    window_h, window_w = window_size
    y_stride, x_stride = stride

    # 确保步长至少为1（避免步长为0导致死循环）
    y_stride = max(1, y_stride)
    x_stride = max(1, x_stride)

    # 计算patches数量（处理单一维度小于窗口的情况）
    # 当维度小于窗口时，只生成1个patch
    n_y = 1 if h <= window_h else ((h - window_h) // y_stride + 1)
    n_x = 1 if w <= window_w else ((w - window_w) // x_stride + 1)

    # 处理边缘（确保最后一个patch能覆盖图像边缘）
    if h > window_h and (h - window_h) % y_stride != 0:
        n_y += 1
    if w > window_w and (w - window_w) % x_stride != 0:
        n_x += 1

    # 预分配数组（所有patch都强制为窗口尺寸）
    channels = image.shape[2] if len(image.shape) == 3 else 0
    if channels > 0:
        patches = np.zeros((n_y * n_x, window_h, window_w, channels), dtype=image.dtype)
    else:
        patches = np.zeros((n_y * n_x, window_h, window_w), dtype=image.dtype)

    positions = []
    idx = 0

    # 批量提取patches（不足部分用0填充）
    for i in range(n_y):
        for j in range(n_x):
            # 计算起始位置（确保最后一个patch对齐图像底部/右侧）
            if h > window_h:
                y = min(i * y_stride, h - window_h)
            else:
                y = 0  # 图像高度小于窗口，只取顶部

            if w > window_w:
                x = min(j * x_stride, w - window_w)
            else:
                x = 0  # 图像宽度小于窗口，只取左侧

            # 计算实际裁剪区域（避免超出图像边界）
            crop_y_end = min(y + window_h, h)
            crop_x_end = min(x + window_w, w)
            crop_h = crop_y_end - y
            crop_w = crop_x_end - x

            # 裁剪图像并填充到窗口尺寸（不足部分用0填充）
            if channels > 0:
                # 3通道图像
                patches[idx, :crop_h, :crop_w, :] = image[y:crop_y_end, x:crop_x_end, :]
                # 不足部分保持0填充（预分配时已初始化）
            else:
                # 单通道图像
                patches[idx, :crop_h, :crop_w] = image[y:crop_y_end, x:crop_x_end]

            positions.append((x, y))
            idx += 1

    # 去除未使用的空patch（如果有）
    if idx < len(patches):
        patches = patches[:idx]

    return patches, positions



def sliding_window_crop(image: np.ndarray, window_size: Tuple[int, int],
                        stride: Tuple[int, int]) -> List[Dict]:
    """
    滑动窗口裁剪（保持兼容性的接口）

    Args:
        image: 输入图像
        window_size: 窗口大小 (height, width)
        stride: 滑动步长 (y_stride, x_stride)

    Returns:
        裁剪后的图像patches列表，每个元素包含patch和位置信息
    """
    patches_array, positions = sliding_window_crop_batch(image, window_size, stride)

    patches = []
    for i, (x, y) in enumerate(positions):
        patch_info = {
            'patch': patches_array[i],
            'position': (x, y),
            'size': window_size
        }
        patches.append(patch_info)

    return patches


def calculate_stride(window_size: Tuple[int, int], overlap_ratio: float) -> Tuple[int, int]:
    """
    根据窗口大小和重叠率计算步长

    Args:
        window_size: 窗口大小 (height, width)
        overlap_ratio: 重叠率 (0.0-1.0)

    Returns:
        步长 (y_stride, x_stride)
    """
    return (int(window_size[0] * (1 - overlap_ratio)),
            int(window_size[1] * (1 - overlap_ratio)))