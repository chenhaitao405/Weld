from ultralytics import YOLO

model = YOLO("/home/lenovo/code/CHT/detect/ultralytics-main/runs/segment/SWRD11m2/weights/best.pt")  # load a custom model

# Validate the model
res = model.predict("/home/lenovo/code/CHT/datasets/Xray/self/20251020weld/SM3-16-DU-013_roi00_焊缝_conf0.719.jpg")
for re in res:
    re.show()

#A