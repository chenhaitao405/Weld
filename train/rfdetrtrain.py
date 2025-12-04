import json
from pathlib import Path

from rfdetr import RFDETRBase, RFDETRLarge, RFDETRSegPreview

model = RFDETRSegPreview()

training_args = {
    "dataset_dir": "/home/lenovo/code/CHT/datasets/Xray/self/1120/labeled/roi2_merge/coco_seg_resize",
    "epochs": 500,
    "batch_size": 4,
    "grad_accum_steps": 2,
    "lr": 1e-4,
    "output_dir": "./runs/detrm_seg",
    "early_stopping": True,
    "run":"4batch"
}

output_dir = Path(training_args["output_dir"]).resolve()
output_dir.mkdir(parents=True, exist_ok=True)
training_args["output_dir"] = str(output_dir)

model.train(
    dataset_dir=training_args["dataset_dir"],
    epochs=training_args["epochs"],
    batch_size=training_args["batch_size"],
    grad_accum_steps=training_args["grad_accum_steps"],
    lr=training_args["lr"],
    output_dir=str(output_dir),
    early_stopping=training_args["early_stopping"],
    run = training_args["run"],

)

config_path = output_dir / "train_params.json"
config_path.write_text(json.dumps(training_args, ensure_ascii=False, indent=2), encoding="utf-8")
