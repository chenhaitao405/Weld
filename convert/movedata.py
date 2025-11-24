import os
import shutil

# 源目录路径
source_dir = "/home/num2/CODE/detect/dataprocess/preprocessed_data"
# 目标目录路径
target_dir = "/home/num2/CODE/detect/dataprocess/alldata"

# 创建目标目录（如果不存在）
os.makedirs(target_dir, exist_ok=True)

# 遍历源目录中的所有文件
for root, dirs, files in os.walk(source_dir):
    for file in files:
        # 构建源文件的完整路径
        source_path = os.path.join(root, file)
        # 构建目标文件的完整路径
        target_path = os.path.join(target_dir, file)

        # 处理文件名冲突：如果目标文件已存在，添加数字后缀
        counter = 1
        while os.path.exists(target_path):
            name, ext = os.path.splitext(file)
            target_path = os.path.join(target_dir, f"{name}_{counter}{ext}")
            counter += 1

        # 移动文件
        shutil.copy2(source_path, target_path)
        print(f"移动文件: {source_path} -> {target_path}")

print("所有文件移动完成！")
