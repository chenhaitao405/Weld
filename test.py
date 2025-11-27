#!/usr/bin/env python3
"""
焊缝缺陷检测数据集分析脚本
功能：
1. 检查单个数据集的训练集和验证集是否有重叠
2. 对比两个数据集之间的差异
"""

import os
import argparse
from pathlib import Path
from collections import defaultdict


def get_image_names(folder_path):
    """获取文件夹中所有图像的文件名（不含扩展名）"""
    if not os.path.exists(folder_path):
        print(f"警告: 路径不存在 - {folder_path}")
        return set()

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    names = set()

    for f in os.listdir(folder_path):
        ext = os.path.splitext(f)[1].lower()
        if ext in image_extensions:
            # 去掉扩展名，只保留文件名
            names.add(os.path.splitext(f)[0])

    return names


def check_overlap(dataset_path):
    """检查数据集的训练集和验证集是否有重叠"""
    print("=" * 60)
    print(f"检查数据集重叠: {dataset_path}")
    print("=" * 60)

    train_path = os.path.join(dataset_path, "images", "train")
    val_path = os.path.join(dataset_path, "images", "val")

    train_names = get_image_names(train_path)
    val_names = get_image_names(val_path)

    print(f"\n训练集图像数量: {len(train_names)}")
    print(f"验证集图像数量: {len(val_names)}")
    print(f"总计: {len(train_names) + len(val_names)}")

    # 检查重叠
    overlap = train_names & val_names

    if overlap:
        print(f"\n⚠️  发现重叠! 共 {len(overlap)} 张图像同时存在于训练集和验证集:")
        print("-" * 40)
        for name in sorted(overlap)[:20]:  # 最多显示20个
            print(f"  - {name}")
        if len(overlap) > 20:
            print(f"  ... 还有 {len(overlap) - 20} 个未显示")
    else:
        print("\n✅ 训练集和验证集没有重叠")

    return train_names, val_names, overlap


def compare_datasets(dataset1_path, dataset2_path, name1="数据集1", name2="数据集2"):
    """对比两个数据集的差异"""
    print("\n" + "=" * 60)
    print(f"对比数据集差异")
    print(f"  {name1}: {dataset1_path}")
    print(f"  {name2}: {dataset2_path}")
    print("=" * 60)

    # 获取两个数据集的图像名
    train1 = get_image_names(os.path.join(dataset1_path, "images", "train"))
    val1 = get_image_names(os.path.join(dataset1_path, "images", "val"))

    train2 = get_image_names(os.path.join(dataset2_path, "images", "train"))
    val2 = get_image_names(os.path.join(dataset2_path, "images", "val"))

    all1 = train1 | val1
    all2 = train2 | val2

    # 基本统计
    print("\n📊 基本统计:")
    print("-" * 40)
    print(f"{'':20} {name1:>12} {name2:>12} {'差异':>10}")
    print(f"{'训练集':20} {len(train1):>12} {len(train2):>12} {len(train1) - len(train2):>+10}")
    print(f"{'验证集':20} {len(val1):>12} {len(val2):>12} {len(val1) - len(val2):>+10}")
    print(f"{'总计':20} {len(all1):>12} {len(all2):>12} {len(all1) - len(all2):>+10}")

    # 分析差异
    print("\n📋 详细差异分析:")
    print("-" * 40)

    # 训练集差异
    only_in_train1 = train1 - train2
    only_in_train2 = train2 - train1
    common_train = train1 & train2

    print(f"\n【训练集对比】")
    print(f"  两者共有: {len(common_train)} 张")
    print(f"  仅在{name1}: {len(only_in_train1)} 张")
    print(f"  仅在{name2}: {len(only_in_train2)} 张")

    # 验证集差异
    only_in_val1 = val1 - val2
    only_in_val2 = val2 - val1
    common_val = val1 & val2

    print(f"\n【验证集对比】")
    print(f"  两者共有: {len(common_val)} 张")
    print(f"  仅在{name1}: {len(only_in_val1)} 张")
    print(f"  仅在{name2}: {len(only_in_val2)} 张")

    # 全量对比
    only_in_all1 = all1 - all2
    only_in_all2 = all2 - all1
    common_all = all1 & all2

    print(f"\n【总体对比】")
    print(f"  两者共有: {len(common_all)} 张")
    print(f"  仅在{name1}: {len(only_in_all1)} 张")
    print(f"  仅在{name2}: {len(only_in_all2)} 张")

    # 检查是否有图片在不同数据集中被划分到不同的集合
    print("\n🔄 划分一致性检查:")
    print("-" * 40)

    # 在两个数据集中都存在，但划分不同的图像
    moved_to_val = (train1 & val2) - val1  # 从train1移到val2
    moved_to_train = (val1 & train2) - train1  # 从val1移到train2

    if moved_to_val:
        print(f"\n⚠️  {len(moved_to_val)} 张图像: 在{name1}是训练集，在{name2}是验证集:")
        for name in sorted(moved_to_val)[:10]:
            print(f"    - {name}")
        if len(moved_to_val) > 10:
            print(f"    ... 还有 {len(moved_to_val) - 10} 个未显示")

    if moved_to_train:
        print(f"\n⚠️  {len(moved_to_train)} 张图像: 在{name1}是验证集，在{name2}是训练集:")
        for name in sorted(moved_to_train)[:10]:
            print(f"    - {name}")
        if len(moved_to_train) > 10:
            print(f"    ... 还有 {len(moved_to_train) - 10} 个未显示")

    if not moved_to_val and not moved_to_train:
        print("✅ 共同图像的训练/验证划分一致")

    # 输出详细文件列表
    print("\n" + "=" * 60)
    print("详细文件列表")
    print("=" * 60)

    def print_file_list(file_set, title, max_show=50):
        print(f"\n{title} ({len(file_set)} 张):")
        if len(file_set) == 0:
            print("  (无)")
        else:
            for name in sorted(file_set)[:max_show]:
                print(f"  {name}")
            if len(file_set) > max_show:
                print(f"  ... 还有 {len(file_set) - max_show} 个未显示")

    print_file_list(only_in_train1, f"仅在{name1}训练集中的图像")
    print_file_list(only_in_val1, f"仅在{name1}验证集中的图像")
    print_file_list(only_in_train2, f"仅在{name2}训练集中的图像")
    print_file_list(only_in_val2, f"仅在{name2}验证集中的图像")

    return {
        'train1': train1, 'val1': val1,
        'train2': train2, 'val2': val2,
        'only_train1': only_in_train1,
        'only_val1': only_in_val1,
        'only_train2': only_in_train2,
        'only_val2': only_in_val2,
    }


def main():
    parser = argparse.ArgumentParser(description='分析YOLO数据集')
    parser.add_argument('--dataset1', '-d1', required=True, help='第一个数据集路径 (baseline)')
    parser.add_argument('--dataset2', '-d2', help='第二个数据集路径 (可选，用于对比)')
    parser.add_argument('--name1', default='数据集1(baseline)', help='第一个数据集名称')
    parser.add_argument('--name2', default='数据集2(新)', help='第二个数据集名称')

    args = parser.parse_args()

    # 检查第一个数据集
    print("\n" + "#" * 60)
    print("# 第一部分: 检查数据集内部重叠")
    print("#" * 60)

    train1, val1, overlap1 = check_overlap(args.dataset1)

    if args.dataset2:
        train2, val2, overlap2 = check_overlap(args.dataset2)

        # 对比两个数据集
        print("\n" + "#" * 60)
        print("# 第二部分: 对比两个数据集差异")
        print("#" * 60)

        compare_datasets(args.dataset1, args.dataset2, args.name1, args.name2)


if __name__ == "__main__":
    main()