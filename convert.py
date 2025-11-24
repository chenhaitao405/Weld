#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
16-bit TIF to 8-bit BMP Batch Converter
将指定目录结构中的所有16位TIF图像转换为8位BMP图像
"""

import os
import sys
import numpy as np
from pathlib import Path
from PIL import Image
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def tif16_to_bmp8(input_path, output_path):
    """
    将16位TIF图像转换为8位BMP图像

    Args:
        input_path: 输入TIF文件路径
        output_path: 输出BMP文件路径

    Returns:
        bool: 转换是否成功
    """
    try:
        # 打开TIF图像
        img = Image.open(input_path)

        # 获取图像数组
        img_array = np.array(img)

        # 检查图像类型和位深
        if img_array.dtype == np.uint16:
            # 16位图像,需要转换
            # 方法1:线性缩放到8位
            # 找到实际的最小值和最大值
            min_val = img_array.min()
            max_val = img_array.max()

            if max_val > min_val:
                # 归一化到0-255
                img_array = ((img_array - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            else:
                # 如果图像是单一值
                img_array = np.zeros_like(img_array, dtype=np.uint8)

        elif img_array.dtype == np.uint8:
            # 已经是8位图像
            pass
        else:
            # 其他数据类型,尝试转换
            img_array = img_array.astype(np.float64)
            min_val = img_array.min()
            max_val = img_array.max()

            if max_val > min_val:
                img_array = ((img_array - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            else:
                img_array = np.zeros_like(img_array, dtype=np.uint8)

        # 创建新的PIL图像
        if len(img_array.shape) == 2:
            # 灰度图像
            new_img = Image.fromarray(img_array, mode='L')
        elif len(img_array.shape) == 3:
            # 彩色图像
            if img_array.shape[2] == 3:
                new_img = Image.fromarray(img_array, mode='RGB')
            elif img_array.shape[2] == 4:
                new_img = Image.fromarray(img_array, mode='RGBA')
            else:
                logger.error(f"不支持的图像通道数: {img_array.shape[2]}")
                return False
        else:
            logger.error(f"不支持的图像维度: {img_array.shape}")
            return False

        # 如果是RGBA,转换为RGB
        if new_img.mode == 'RGBA':
            # 创建白色背景
            background = Image.new('RGB', new_img.size, (255, 255, 255))
            background.paste(new_img, mask=new_img.split()[3])
            new_img = background

        # 保存为BMP格式
        new_img.save(output_path, 'BMP')

        return True

    except Exception as e:
        logger.error(f"转换文件 {input_path} 时出错: {str(e)}")
        return False


def process_folder_structure(source_root, target_root, folder_names):
    """
    处理文件夹结构,转换所有TIF文件为BMP

    Args:
        source_root: 源根目录路径
        target_root: 目标根目录路径
        folder_names: 要处理的文件夹名称列表(如 ['D1', 'D2', 'D3', 'D4'])
    """
    source_path = Path(source_root)
    target_path = Path(target_root)

    # 检查源目录是否存在
    if not source_path.exists():
        logger.error(f"源目录不存在: {source_root}")
        return

    # 创建目标根目录
    target_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"目标根目录: {target_path}")

    # 统计信息
    total_files = 0
    total_success = 0
    total_fail = 0
    folder_stats = {}

    # 遍历每个主文件夹
    for folder_name in folder_names:
        source_folder = source_path / folder_name
        target_folder = target_path / folder_name

        if not source_folder.exists():
            logger.warning(f"文件夹不存在,跳过: {source_folder}")
            continue

        logger.info(f"\n处理文件夹: {folder_name}")
        logger.info("=" * 50)

        # 创建目标文件夹
        target_folder.mkdir(parents=True, exist_ok=True)

        # 收集所有TIF文件(遍历所有子目录) - 使用set去重
        tif_files = set()
        for ext in ['*.tif', '*.TIF', '*.tiff', '*.TIFF']:
            # 递归查找所有子目录中的TIF文件
            tif_files.update(source_folder.rglob(ext))

        # 转换为列表并排序,便于处理
        tif_files = sorted(list(tif_files))

        if not tif_files:
            logger.warning(f"  未找到TIF文件在: {folder_name}")
            folder_stats[folder_name] = {'total': 0, 'success': 0, 'fail': 0}
            continue

        logger.info(f"  找到 {len(tif_files)} 个TIF文件")

        # 文件夹级别的统计
        folder_success = 0
        folder_fail = 0

        # 处理每个TIF文件
        for i, tif_file in enumerate(tif_files, 1):
            # 生成唯一的输出文件名
            # 如果有重名文件,添加父目录名称作为前缀
            relative_path = tif_file.relative_to(source_folder)

            # 如果文件在子目录中,使用子目录名作为前缀
            if len(relative_path.parts) > 1:
                # 获取子目录名称
                output_filename = f"{tif_file.stem}.bmp"
            else:
                output_filename = f"{tif_file.stem}.bmp"

            output_file_path = target_folder / output_filename

            # 显示进度
            progress = f"[{i}/{len(tif_files)}]"
            rel_path_display = str(relative_path)
            if len(rel_path_display) > 50:
                rel_path_display = "..." + rel_path_display[-47:]

            logger.info(f"  {progress} {rel_path_display}")
            logger.info(f"    -> {output_file_path.name}")

            # 执行转换
            if tif16_to_bmp8(tif_file, output_file_path):
                folder_success += 1
                total_success += 1
                logger.info(f"    ✓ 转换成功")
            else:
                folder_fail += 1
                total_fail += 1
                logger.error(f"    ✗ 转换失败")

        total_files += len(tif_files)
        folder_stats[folder_name] = {
            'total': len(tif_files),
            'success': folder_success,
            'fail': folder_fail
        }

        logger.info(f"\n  {folder_name} 完成: 成功 {folder_success}/{len(tif_files)}")

    # 输出总体统计信息
    logger.info("\n" + "=" * 60)
    logger.info("转换完成!统计信息:")
    logger.info("=" * 60)

    for folder_name, stats in folder_stats.items():
        if stats['total'] > 0:
            success_rate = (stats['success'] / stats['total']) * 100
            logger.info(
                f"{folder_name:10} - 总计: {stats['total']:4d} | 成功: {stats['success']:4d} | 失败: {stats['fail']:4d} | 成功率: {success_rate:.1f}%")
        else:
            logger.info(f"{folder_name:10} - 无文件")

    logger.info("-" * 60)
    logger.info(f"总计文件数: {total_files}")
    logger.info(f"成功转换: {total_success}")
    logger.info(f"失败转换: {total_fail}")
    if total_files > 0:
        logger.info(f"总体成功率: {(total_success / total_files) * 100:.1f}%")
    logger.info(f"输出目录: {target_path}")


def main():
    """主函数"""
    # 设置源目录和目标目录
    source_root = r"C:\Users\CHT\Desktop\datasets1117\oriImage"
    target_root = r"C:\Users\CHT\Desktop\datasets1117\bmpimage"

    # 要处理的文件夹列表
    folder_names = ['D1', 'D2', 'D3', 'D4']

    logger.info("16位TIF转8位BMP批量转换工具")
    logger.info("=" * 60)
    logger.info(f"源目录: {source_root}")
    logger.info(f"目标目录: {target_root}")
    logger.info(f"处理文件夹: {', '.join(folder_names)}")
    logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # 检查依赖库
    try:
        import PIL
        import numpy
    except ImportError as e:
        logger.error("缺少必要的依赖库,请安装:")
        logger.error("pip install pillow numpy")
        sys.exit(1)

    # 执行转换
    process_folder_structure(source_root, target_root, folder_names)

    logger.info(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()