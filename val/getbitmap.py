import cv2
import numpy as np
import os


def count_bmp_bit_depth(directory):
    """快速统计BMP文件位深度"""
    count_8bit = 0
    count_16bit = 0
    count_other = 0
    total = 0

    # 遍历目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.bmp') or file.lower().endswith('.tif'):
                file_path = os.path.join(root, file)
                total += 1

                # 读取图片
                img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

                if img is not None:
                    if img.dtype == np.uint8:
                        count_8bit += 1
                        print(f"8位: {file}")
                    elif img.dtype == np.uint16:
                        count_16bit += 1
                        print(f"16位: {file}")
                    else:
                        count_other += 1

    # 打印结果
    print(f"\n统计结果:")
    print(f"总计: {total} 个BMP文件")
    print(f"8位图像: {count_8bit} 个 ({count_8bit / total * 100:.1f}%)")
    print(f"16位图像: {count_16bit} 个 ({count_16bit / total * 100:.1f}%)")
    if count_other > 0:
        print(f"其他: {count_other} 个")


# 使用
count_bmp_bit_depth("/home/lenovo/code/CHT/datasets/Xray/opensource/crop_weld_data/crop_weld_images/L/1")