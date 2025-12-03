from rfdetr import RFDETRBase ,RFDETRLarge

model = RFDETRLarge()

model.train(
    dataset_dir="/home/lenovo/code/CHT/datasets/Xray/self/1120/labeled/roi2_merge/coco_det",
    epochs=500,
    batch_size=4,
    grad_accum_steps=4,
    lr=1e-4,
    # resolution = 840 ,
    output_dir="./runs/detrlarge_defaultgrd4",
    early_stopping =True
)
