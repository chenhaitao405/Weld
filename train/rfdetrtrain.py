from rfdetr import RFDETRBase ,RFDETRLarge

model = RFDETRLarge()

model.train(
    dataset_dir="/home/lenovo/code/CHT/datasets/Xray/self/1120/labeled/roi2_merge/coco_det",
    epochs=500,
    batch_size=2,
    grad_accum_steps=4,
    lr=1e-4,
    output_dir="./runs/detrlarge",
    early_stopping =True
)
