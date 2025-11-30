#!/usr/bin/env python3
"""
脚本功能：
遍历val文件夹中的bmp文件，根据文件名前缀找到对应的源图片和标签，
复制到输出目录。

文件名格式：{DATASET}_{原文件名}.bmp，例如 D1_11.bmp
源图片路径：BASE_PATH/{DATASET}/{原文件名}.bmp
源标签路径：JSON_BASE_PATH/{DATASET}/label/{原文件名}.json
"""

import os
import shutil
import platform

# ==================== 配置区域 ====================

# 根据操作系统自动选择路径
if platform.system() == "Windows":
    BASE_PATH = r"C:\Users\CHT\Desktop\datasets1117\labeled"
    JSON_BASE_PATH = r"C:\Users\CHT\Desktop\datasets1117\adjust"
elif platform.system() == "Linux":
    BASE_PATH = "/home/lenovo/code/CHT/datasets/Xray/self/1120/labeled"
    JSON_BASE_PATH = "/home/lenovo/code/CHT/datasets/Xray/self/1120/adjust"
else:
    raise EnvironmentError(
        f"不支持的操作系统：{platform.system()}\n"
        "请在配置区域添加对应系统的路径配置"
    )

DATASETS = [
    "D1",
    "D2",
    "D3",
    "D4",
    "img20250608",
    "img20250609"
]

# val文件夹路径
VAL_DIR = "/home/lenovo/code/CHT/datasets/Xray/self/1120/labeled/roi2_merge/yolo/images/val"

# 输出目录
OUTPUT_DIR = "/home/lenovo/code/CHT/datasets/Xray/self/1120/labeled/roi2_merge/yolo/val_sources"


# ==================== 主逻辑 ====================

def parse_filename(filename):
    """
    解析文件名，提取数据集前缀和原始文件名
    例如：D1_11.bmp -> ("D1", "11.bmp")
          img20250608_001.bmp -> ("img20250608", "001.bmp")
    """
    name_without_ext = os.path.splitext(filename)[0]
    ext = os.path.splitext(filename)[1]

    # 尝试匹配数据集前缀
    for dataset in DATASETS:
        prefix = dataset + "_"
        if name_without_ext.startswith(prefix):
            original_name = name_without_ext[len(prefix):] + ext
            return dataset, original_name

    return None, None


def main():
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "label"), exist_ok=True)

    # 统计信息
    total_files = 0
    copied_images = 0
    copied_labels = 0
    missing_images = []
    missing_labels = []
    parse_failed = []

    # 遍历val文件夹
    for filename in os.listdir(VAL_DIR):
        if not filename.lower().endswith('.bmp'):
            continue

        total_files += 1

        # 解析文件名
        dataset, original_name = parse_filename(filename)

        if dataset is None:
            parse_failed.append(filename)
            print(f"[警告] 无法解析文件名: {filename}")
            continue

        # 构建源路径
        src_image_path = os.path.join(BASE_PATH, dataset, original_name)

        # 标签文件名（将.bmp替换为.json）
        label_name = os.path.splitext(original_name)[0] + ".json"
        src_label_path = os.path.join(JSON_BASE_PATH, dataset, "label", label_name)

        # 构建目标路径（保留前缀以避免重名）
        dst_image_path = os.path.join(OUTPUT_DIR, filename)
        dst_label_name = os.path.splitext(filename)[0] + ".json"
        dst_label_path = os.path.join(OUTPUT_DIR, "label", dst_label_name)

        # 复制图片
        if os.path.exists(src_image_path):
            shutil.copy2(src_image_path, dst_image_path)
            copied_images += 1
        else:
            missing_images.append((filename, src_image_path))
            print(f"[警告] 源图片不存在: {src_image_path}")

        # 复制标签
        if os.path.exists(src_label_path):
            shutil.copy2(src_label_path, dst_label_path)
            copied_labels += 1
        else:
            missing_labels.append((filename, src_label_path))
            print(f"[警告] 源标签不存在: {src_label_path}")

    # 输出统计信息
    print("\n" + "=" * 50)
    print("处理完成！统计信息：")
    print("=" * 50)
    print(f"val文件夹中的bmp文件总数: {total_files}")
    print(f"成功复制的图片数: {copied_images}")
    print(f"成功复制的标签数: {copied_labels}")
    print(f"解析失败的文件数: {len(parse_failed)}")
    print(f"缺失的源图片数: {len(missing_images)}")
    print(f"缺失的源标签数: {len(missing_labels)}")
    print(f"\n输出目录: {OUTPUT_DIR}")
    print(f"标签目录: {os.path.join(OUTPUT_DIR, 'label')}")

    if parse_failed:
        print(f"\n解析失败的文件: {parse_failed[:10]}{'...' if len(parse_failed) > 10 else ''}")

    if missing_images:
        print(f"\n缺失的源图片示例: {missing_images[:5]}")

    if missing_labels:
        print(f"\n缺失的源标签示例: {missing_labels[:5]}")


if __name__ == "__main__":
    main()