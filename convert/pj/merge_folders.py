import os
import shutil
import argparse
from pathlib import Path


def merge_folders(folder1, folder2, output_folder):
    """
    合并两个具有相同结构的文件夹

    参数:
        folder1: 第一个文件夹路径
        folder2: 第二个文件夹路径
        output_folder: 输出文件夹路径
    """
    folder1_path = Path(folder1)
    folder2_path = Path(folder2)
    output_path = Path(output_folder)

    # 检查输入文件夹是否存在
    if not folder1_path.exists():
        print(f"错误: 文件夹1不存在: {folder1}")
        return

    if not folder2_path.exists():
        print(f"错误: 文件夹2不存在: {folder2}")
        return

    # 创建输出文件夹
    output_path.mkdir(parents=True, exist_ok=True)

    # 收集所有需要处理的子目录结构
    all_subdirs = set()

    # 遍历文件夹1的结构
    for root, dirs, files in os.walk(folder1_path):
        rel_path = Path(root).relative_to(folder1_path)
        all_subdirs.add(rel_path)

    # 遍历文件夹2的结构
    for root, dirs, files in os.walk(folder2_path):
        rel_path = Path(root).relative_to(folder2_path)
        all_subdirs.add(rel_path)

    # 创建所有子目录
    for subdir in all_subdirs:
        (output_path / subdir).mkdir(parents=True, exist_ok=True)

    # 复制文件夹1的文件
    print(f"正在复制文件夹1的文件: {folder1}")
    file_count1 = 0
    for root, dirs, files in os.walk(folder1_path):
        rel_path = Path(root).relative_to(folder1_path)
        dest_dir = output_path / rel_path

        for file in files:
            src_file = Path(root) / file
            dest_file = dest_dir / file

            # 如果目标文件已存在,添加后缀
            if dest_file.exists():
                base_name = dest_file.stem
                extension = dest_file.suffix
                counter = 1
                while dest_file.exists():
                    dest_file = dest_dir / f"{base_name}_copy{counter}{extension}"
                    counter += 1

            shutil.copy2(src_file, dest_file)
            file_count1 += 1

    print(f"已复制 {file_count1} 个文件从文件夹1")

    # 复制文件夹2的文件
    print(f"正在复制文件夹2的文件: {folder2}")
    file_count2 = 0
    for root, dirs, files in os.walk(folder2_path):
        rel_path = Path(root).relative_to(folder2_path)
        dest_dir = output_path / rel_path

        for file in files:
            src_file = Path(root) / file
            dest_file = dest_dir / file

            # 如果目标文件已存在,添加后缀
            if dest_file.exists():
                base_name = dest_file.stem
                extension = dest_file.suffix
                counter = 1
                while dest_file.exists():
                    dest_file = dest_dir / f"{base_name}_copy{counter}{extension}"
                    counter += 1

            shutil.copy2(src_file, dest_file)
            file_count2 += 1

    print(f"已复制 {file_count2} 个文件从文件夹2")
    print(f"总共复制 {file_count1 + file_count2} 个文件到: {output_folder}")
    print("合并完成!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='合并两个具有相同结构的文件夹')
    parser.add_argument('folder1', help='第一个文件夹路径')
    parser.add_argument('folder2', help='第二个文件夹路径')
    parser.add_argument('output', help='输出文件夹路径')

    args = parser.parse_args()

    merge_folders(args.folder1, args.folder2, args.output)