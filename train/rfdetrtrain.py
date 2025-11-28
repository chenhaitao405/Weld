from rfdetr import RFDETRBase ,RFDETRMedium

model = RFDETRMedium()

model.train(
    dataset_dir="/home/lenovo/code/CHT/datasets/Xray/self/1120/labeled/roi2_merge/coco",
    epochs=500,
    batch_size=16,
    grad_accum_steps=4,
    lr=1e-4,
    output_dir="./runs/detrmedium"
)
