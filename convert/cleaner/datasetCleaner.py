import os
import shutil
import argparse
from pathlib import Path

# 支持的图片格式
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif', '.webp'}


def generate_manifest(input_dir, manifest_path):
    """
    模式1：生成清单模式
    遍历input_dir，生成剩余数据的清单
    """
    input_path = Path(input_dir)

    if not input_path.exists():
        print(f"错误: 输入目录 '{input_dir}' 不存在")
        return

    image_files = []

    # 遍历所有文件
    for file_path in input_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
            # 获取相对路径
            relative_path = file_path.relative_to(input_path)
            image_files.append(str(relative_path))

    # 排序以保持一致性
    image_files.sort()

    # 保存清单
    manifest_file = Path(manifest_path)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)

    with open(manifest_file, 'w', encoding='utf-8') as f:
        for img_path in image_files:
            f.write(f"{img_path}\n")

    print(f"✓ 清单生成完成")
    print(f"  - 找到 {len(image_files)} 个图片文件")
    print(f"  - 清单保存至: {manifest_path}")


def copy_from_manifest(input_dir, manifest_path, output_dir):
    """
    模式2：生成数据模式
    根据清单从input_dir复制文件到output_dir，保持目录结构
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    manifest_file = Path(manifest_path)

    # 检查输入
    if not input_path.exists():
        print(f"错误: 输入目录 '{input_dir}' 不存在")
        return

    if not manifest_file.exists():
        print(f"错误: 清单文件 '{manifest_path}' 不存在")
        return

    # 读取清单
    with open(manifest_file, 'r', encoding='utf-8') as f:
        file_list = [line.strip() for line in f if line.strip()]

    # 复制文件
    success_count = 0
    missing_count = 0

    for relative_path in file_list:
        src_file = input_path / relative_path
        dst_file = output_path / relative_path

        if src_file.exists():
            # 创建目标目录
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            # 复制文件
            shutil.copy2(src_file, dst_file)
            success_count += 1
            print(f"  复制: {relative_path}")
        else:
            print(f"  ⚠ 缺失: {relative_path}")
            missing_count += 1

    print(f"\n✓ 数据复制完成")
    print(f"  - 成功复制: {success_count} 个文件")
    if missing_count > 0:
        print(f"  - 缺失文件: {missing_count} 个")
    print(f"  - 输出目录: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='数据集清理工具 - 生成清单或根据清单复制数据',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

  模式1 - 生成清单:
    python script.py --mode manifest --input_dir ./cleaned_data --manifest ./manifest.txt

  模式2 - 根据清单生成数据:
    python script.py --mode copy --input_dir ./original_data --manifest ./manifest.txt --output_dir ./output_data
        """
    )

    parser.add_argument('--mode', required=True, choices=['manifest', 'copy'],
                        help='运行模式: manifest=生成清单, copy=根据清单复制数据')
    parser.add_argument('--input_dir', required=True,
                        help='输入目录路径')
    parser.add_argument('--manifest', required=True,
                        help='清单文件路径')
    parser.add_argument('--output_dir',
                        help='输出目录路径 (仅在copy模式下需要)')

    args = parser.parse_args()

    if args.mode == 'manifest':
        print("=" * 50)
        print("模式1: 生成清单")
        print("=" * 50)
        generate_manifest(args.input_dir, args.manifest)

    elif args.mode == 'copy':
        if not args.output_dir:
            parser.error("copy模式需要指定 --output_dir 参数")
        print("=" * 50)
        print("模式2: 根据清单生成数据")
        print("=" * 50)
        copy_from_manifest(args.input_dir, args.manifest, args.output_dir)


if __name__ == '__main__':
    main()