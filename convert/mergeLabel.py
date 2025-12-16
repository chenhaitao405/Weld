import os
import json
import shutil
from pathlib import Path
from typing import Dict, List

# 基础路径配置
BASE_PATH ="/datasets/PAR/Xray/self/1120/1215adjust/1120"
OUTPUT_PATH = "/datasets/PAR/Xray/self/1120/1215adjust"

# 数据集列表
DATASETS = [
    "D1",
    "D2",
    "D3",
    "D4",
    "img20250608",
    "img20250609"
]

# DATASETS = [
#     "6-1_8bit",
#     "6-2_8bit",
#     "7-1_8bit",
#     "7-2_8bit",
#     "8_8bit",
#     "9-1_8bit",
#     "9-2_8bit",
#     "DP_8bit"
# ]

# 标注类型映射关系
LABEL_MAPPING = {
    # 裂纹类
    "裂纹": "裂纹",

    # 未熔合类
    "未熔合": "未熔合",
    "坡口未熔合": "未熔合",
    "层间未熔合": "未熔合",
    "接头未熔合": "未熔合",
    "根部未熔合": "未熔合",
    "未融合": "未熔合",

    # 未焊透类
    "未焊透": "未焊透",

    # 条形缺陷类
    "条形缺陷": "条形缺陷",
    "条状缺陷": "条形缺陷",

    # 圆形缺陷类
    "圆形缺陷": "圆形缺陷",
    "气孔": "圆形缺陷",
    "密孔": "圆形缺陷",
    "深孔": "圆形缺陷",
    "链状气孔": "圆形缺陷",
    "密集圆形缺陷":"圆形缺陷",

    # 咬边类
    "咬边": "咬边",
    "内咬边": "咬边",
    "外咬边": "咬边",

    # 内凹类
    "内凹": "内凹",

    # 其他类
    "条状缺陷，圆形缺陷": "其他",
    "未盖面": "其他",
    "未焊满": "其他",
    "凹陷": "其他",
    "漏焊": "其他",
    "焊瘤": "其他",
    "根部成形不良": "其他",
    "错口": "其他",
    "折口": "其他",
    "焊丝": "其他",
    "焊头": "其他",
    "伪缺线": "其他",
    "伪缺陷": "其他",
    "水渍": "其他",
    "划伤": "其他",
    "母材缺陷": "其他",
    "内腐蚀": "其他",
    "异物": "其他",
    "其他": "其他",
    "铁水": "其他"
}


def adjust_label(label: str) -> str:
    """
    根据映射关系调整标签

    Args:
        label: 原始标签

    Returns:
        调整后的标签
    """
    # 如果标签在映射表中，返回对应的新标签
    if label in LABEL_MAPPING:
        return LABEL_MAPPING[label]

    # 如果标签不在映射表中，打印警告并归类为"其他"
    print(f"警告: 未知标签 '{label}'，将归类为'其他'")
    return "其他"


def process_json_file(input_path: Path, output_path: Path) -> Dict:
    """
    处理单个JSON文件，调整其中的标注类型

    Args:
        input_path: 输入JSON文件路径
        output_path: 输出JSON文件路径

    Returns:
        包含处理结果的字典
    """
    stats = {
        "original_labels": {},
        "adjusted_labels": {},
        "adjustments": []
    }

    try:
        # 读取JSON文件
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 检查是否有shapes字段
        if 'shapes' in data:
            for shape in data['shapes']:
                if 'label' in shape:
                    original_label = shape['label']
                    adjusted_label = adjust_label(original_label)

                    # 统计原始标签
                    stats["original_labels"][original_label] = stats["original_labels"].get(original_label, 0) + 1

                    # 统计调整后的标签
                    stats["adjusted_labels"][adjusted_label] = stats["adjusted_labels"].get(adjusted_label, 0) + 1

                    # 记录调整
                    if original_label != adjusted_label:
                        stats["adjustments"].append(f"{original_label} -> {adjusted_label}")

                    # 更新标签
                    shape['label'] = adjusted_label

        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 保存调整后的JSON文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return stats

    except Exception as e:
        print(f"处理文件 {input_path} 时出错: {e}")
        return None


def main():
    """主函数"""
    print("=" * 60)
    print("LabelMe标注类型批量调整工具")
    print("=" * 60)

    total_files = 0
    total_adjustments = 0
    all_original_labels = {}
    all_adjusted_labels = {}

    # 遍历每个数据集
    for dataset in DATASETS:
        dataset_path = Path(BASE_PATH) / dataset
        label_dir = dataset_path / "label"

        if not label_dir.exists():
            print(f"\n警告: 标签目录不存在 - {label_dir}")
            continue

        print(f"\n处理数据集: {dataset}")
        print("-" * 40)

        # 获取所有JSON文件
        json_files = list(label_dir.glob("*.json"))

        if not json_files:
            print(f"  未找到JSON文件")
            continue

        print(f"  找到 {len(json_files)} 个JSON文件")

        dataset_adjustments = 0

        # 处理每个JSON文件
        for json_file in json_files:
            # 构建输出路径，保持相同的目录结构
            relative_path = json_file.relative_to(dataset_path)
            output_file = Path(OUTPUT_PATH) / dataset / relative_path

            # 处理文件
            stats = process_json_file(json_file, output_file)

            if stats:
                total_files += 1
                dataset_adjustments += len(stats["adjustments"])

                # 汇总统计
                for label, count in stats["original_labels"].items():
                    all_original_labels[label] = all_original_labels.get(label, 0) + count

                for label, count in stats["adjusted_labels"].items():
                    all_adjusted_labels[label] = all_adjusted_labels.get(label, 0) + count

        total_adjustments += dataset_adjustments
        print(f"  完成! 调整了 {dataset_adjustments} 个标签")

    # 打印总体统计
    print("\n" + "=" * 60)
    print("处理完成！总体统计：")
    print("=" * 60)
    print(f"处理文件总数: {total_files}")
    print(f"调整标签总数: {total_adjustments}")

    print("\n原始标签分布:")
    for label, count in sorted(all_original_labels.items()):
        print(f"  {label}: {count}")

    print("\n调整后标签分布:")
    for label, count in sorted(all_adjusted_labels.items()):
        print(f"  {label}: {count}")

    print(f"\n所有调整后的文件已保存至: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()