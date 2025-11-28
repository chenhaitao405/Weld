from rfdetr import RFDETRBase

model = RFDETRBase()

model.train(
    dataset_dir="/home/lenovo/code/CHT/datasets/Xray/self/1120/labeled/roi2_merge/coco",
    epochs=10,
    batch_size=4,
    grad_accum_steps=4,
    lr=1e-4,
    output_dir="./runs/detr"
)
