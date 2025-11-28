import matplotlib

matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import argparse  # 导入命令行参数解析模块

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['AR PL UKai CN', 'Noto Sans CJK JP', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
print(f"\n已设置字体为: AR PL UKai CN")

from ultralytics import YOLO

# 1. 加载 YOLOv11M 预训练分类模型（自动下载权重）
model = YOLO("yolo11m-cls.yaml")  # build a new model from YAML

# 2. 训练分类模型（核心参数仅保留必要项，其他默认）
results = model.train(
    data="/home/lenovo/code/CHT/datasets/Xray/self/1120/labeled/CLS640/cls",  # 替换为你的数据集路径
    epochs=500,
    batch=128,
    imgsz=224,
    workers =8,
    name="cor_cls640_balance", erasing=0, mosaic=0, hsv_h=0.0,
    hsv_s=0.0, hsv_v=0.0, auto_augment=None, scale=0
)
