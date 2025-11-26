from ultralytics import YOLO

# Load a model

model = YOLO("yolo11m.pt") # build from YAML and transfer weights

# Train the model
results = model.train(data="welddet.yaml", epochs=100, imgsz=640, batch=32 ,name = "11m_pretrain", pretrained =True)

# 数据增强删除