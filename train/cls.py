from ultralytics import YOLO

# 1. 加载 YOLOv11M 预训练分类模型（自动下载权重）
model = YOLO("yolo11m-cls.yaml")  # build a new model from YAML

# 2. 训练分类模型（核心参数仅保留必要项，其他默认）
results = model.train(
    data=r"C:\Users\CHT\Desktop\datasets1117\labeled\processed\cls",  # 替换为你的数据集路径
    epochs=500,
    batch=64,
    imgsz=224,
    workers =0,
    name="weld",                  # 任务名改为 "weld"（输出目录会变成 runs/cls/weld）
    augment=False,                # 关闭所有图像增强（核心参数）
    mixup=0.0,                    # 额外禁用mixup混合增强（保险起见，默认augment=False已包含）
    mosaic=0.0,                   # 禁用mosaic增强（分类任务默认关闭，此处明确指定）
    erasing=0.0,
)