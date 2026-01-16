# 焊缝检测数据处理与推理流水线

## 项目概览

- 通过 `run_data_pipeline.py` 将多源 Labelme 标注统一到 YOLO → ROI → 旋转 → 切片增强 → 任务转换 → COCO → 合并的完整七步流水线，生成可直接训练/评估的数据集。
- 通过 `run_full_pipeline.py` 串联 `run_inference_pipeline.py`、`validate_inference_results.py`、`visualize_validation_results.py`，一条命令完成推理、指标验证与可视化报告生成，并可选接入 MLflow。
- 所有路径、输出目录与脚本参数均由 `configs/pipeline_profiles.yaml` 里的 profile 控制，可根据不同数据源/环境切换。

## 环境要求

- Python 3.8+（建议 3.10），并安装 `tqdm`、`opencv-python`、`ultralytics`、`pyyaml`、`mlflow`（若启用 MLflow）等依赖。
- Windows 与 Linux 通用，但 profile 中的 `paths.base_path`、`paths.json_base_path` 等需要与当前系统路径一致。

## 配置体系（`configs/pipeline_profiles.yaml`）

1. `default_profile`：未显式指定 `--profile` 时默认启用的 profile；亦可根据 `platform` 自动匹配。
2. `profiles.<name>.paths`：
   - `base_path`：原始图像根目录。
   - `json_base_path`：Labelme JSON 根目录，脚本会在其下寻找 `<dataset>/label`.
   - `output_base_dir`：流水线所有中间结果的根目录。
   - `reference_label_map_path`：可选。若设置，步骤1会直接复用该 `dataset.yaml` 的 `label_id_map`；否则自动扫描所有 JSON 生成。
3. `datasets`：需要处理的数据集列表，脚本会逐个遍历。
4. `outputs`：为每个步骤声明输出目录关键字，例如 `yolo_dir`、`roi_dir`、`coco_dir`。所有键都会解析为绝对路径。
5. `params`：各步骤对应脚本及固定参数，例如
   - `labelme2yolo.script_path`、`unify_to_crack`。
   - `yolo_roi_extractor.model_path`、`roi_conf`、`mode` 等。
   - `patchandenhance.slice_mode`（1=仅增强，2=滑窗裁剪，3=横切纵拼）。
   - `seg2det.balance_data`、`yolo2coco.task`、`merge_coco.dataset_b` 等。
6. `param_log_path`（可选）：若在 profile 中设置，会将运行记录保存到指定路径；否则默认 `<output_base_dir>/pipeline_params.json`。

运行时可通过 `--config-path` 切换配置文件，或通过 `--profile` 切换 profile。

## 流水线一：`run_data_pipeline.py`

### 命令行参数

- `--steps 1234567`：选择要执行的步骤，默认全部 1~7。
- `--force`：前置输出缺失时继续执行（默认遇到错误直接退出）。
- `--config-path`：配置文件路径，默认 `configs/pipeline_profiles.yaml`。
- `--profile`：指定 profile 名称；不指定则使用默认或按系统自动匹配。

脚本启动后会输出当前 profile 的关键路径、数据集列表，并把本次选择的步骤写入 `pipeline_params.json`。

### 七个步骤

| 步骤 | 名称 | 主脚本/函数 | 输出目录键 | 说明 |
| ---- | ---- | ----------- | ---------- | ---- |
| 1 | Labelme 转 YOLO | `convert/labelme2yolo.py` | `yolo_dir` | 汇总所有数据集的标签，基于参考 `dataset.yaml` 或自动扫描构建统一 `label_id_map`（支持 `unify_to_crack`）。输出统一的 YOLO `images/labels` 以及新的 `dataset.yaml`。 |
| 2 | YOLO ROI 提取 | `convert/pj/yolo_roi_extractor.py` | `roi_dir` | 使用指定的 ROI 模型（如 `model/weldROI2.pt`）在 YOLO 分割结果上裁出焊缝区域，支持置信度、IoU、padding 调整。 |
| 3 | 竖图旋转归一 | `convert/pj/rotateYOLOdate.py` | `roi_rotate` | 自动检测竖图并旋转到统一方向，同时变换标签坐标。 |
| 4 | 图像裁剪与增强 | `convert/pj/patchandenhance.py` | `patch_dir` | 根据 `slice_mode` 选择仅增强、滑窗裁剪或纵横拼接；可配置 window、overlap、窗宽窗位等增强参数。 |
| 5 | 训练任务转换 | `convert/pj/seg2det.py` | `cls_dir` | 在分割、检测、分类任务间转换标签，可选 `--balance_data` 与 `--balance_ratio`。 |
| 6 | YOLO → COCO | `convert/yolo2coco.py` | `coco_dir` | 将步骤5的结果转换为 COCO JSON，支持 `task`、`test_split_ratio`、`split_seed`。 |
| 7 | COCO 合并 | `convert/merge_coco.py` | `merged_coco_dir` | 与外部 COCO 数据集合并，可配置 `dataset_b`、`splits`、`merge_ratio`、前缀及是否复制图片。 |

每一步都会调用对应脚本并在控制台打印完整命令；执行记录同时写入 `pipeline_params.json`，便于追踪输入参数。

### 输出目录结构

`OUTPUT_BASE_DIR` 下的典型子目录：

- `yolo/`：步骤1统一后的 YOLO 分割数据及 `dataset.yaml`。
- `ROI/`：步骤2抽取的焊缝 ROI 图像与标签。
- `ROI_rotate/`：步骤3旋转归一后的 ROI 数据。
- `patch_*`：步骤4 裁剪/增强得到的贴片。
- `patch_det`（或 profile 中自定义的 `cls_dir`）：步骤5 转换后的检测/分类格式。
- `coco_from_patch/`：步骤6 的 COCO JSON 与图片。
- `coco_merged/`：步骤7 合并后的 COCO 数据集。

### 常见命令

```bash
# 使用默认配置全流程处理
python run_data_pipeline.py --steps 1234567

# 指定 profile，仅运行 ROI→旋转→切片
python run_data_pipeline.py --profile linux_1205_folder --steps 234 --force

# 使用自定义配置文件并只执行 YOLO→COCO
python run_data_pipeline.py --config-path configs/custom.yaml --steps 123456
```

若某数据集缺失或单步失败，可修复后重新执行对应步骤；脚本暂未提供 dry-run 功能，运行前请确认 profile 配置无误。

## 流水线二：`run_full_pipeline.py`

该脚本将推理、验证、可视化三阶段串联，并自动补齐常用参数。

### 核心参数

- `--base-path`：流水线产出根目录，默认 `outputs/pipeline_run`。
- `--infer-subdir` / `--valid-subdir`：推理与验证子目录名，默认 `infer` / `valid`。
- `--inference-results`：推理结果 JSON 名称，默认 `inference_results.json`。
- `--run-inference-opts`：传递给 `run_inference_pipeline.py` 的原始参数字符串，**必须包含 `--image-dir`**，否则无法推断验证阶段的 `--image-root`。
- `--validate-opts` / `--visualize-opts`：分别透传到验证与可视化脚本。
- `--steps 123`：选择执行推理(1)、验证(2)、可视化(3)；默认全部执行。
- `--dry-run`：仅打印将要执行的命令。
- `--mlflow` 及其相关参数：`--mlflow-experiment`、`--mlflow-run-name`、`--mlflow-tracking-uri`、`--mlflow-tags key=value`。

### 执行流程

1. 根据提供的 opts 构建三条命令，并确保 `--output-dir`、`--results-json`、`--inference-json`、`--validation-dir`、`--output-html` 等参数齐全。
2. 创建 `<base-path>/pipeline_args.json` 保存本次运行的所有参数、目录、报告路径等上下文信息。
3. 依次执行推理、验证、可视化（可通过 `--steps` 跳过部分阶段）。每阶段会打印 `[CMD]` 日志，`--dry-run` 模式下只打印不执行。
4. 验证阶段默认输出 `metrics_summary.json`、`data/manifest.json`、`per_class_metrics.png`；可视化阶段生成 `report.html`。

### 漏标注筛选

- `--verified-threshold`（默认 0.75）可设置 `(1 - IoU) * 置信度` 判定阈值，满足条件且无标注匹配的预测会被标记为“漏标注”，在报告中以金色框高亮。
- 所有命中的原图会复制到 `verified_image/` 子目录（可用 `--verified-image-dir` 调整目录名），供团队内部留存。
- 如需直接打包给客户，可加 `--export-verified-bundle`，脚本会在 `verified_bundle/`（可用 `--verified-bundle-dir` 与 `--verified-bundle-title` 自定义）下生成仅包含漏标注样本的数据、media、thumbnails、verified_image 以及独立的 `report.html`，解压后在任意机器上即可查看。
- `visualize_validation_results.py` 会把“漏标注”作为独立类别纳入图例、筛选器与统计卡片。
- 若通过 `run_full_pipeline.py` 运行，可在 `--validate-opts` 里拼接这些参数，例如 `--validate-opts "--verified-threshold 0.8 --export-verified-bundle"`。

### MLflow 集成

启用 `--mlflow` 时脚本会：

- 创建/进入指定实验，支持自定义 tracking URI 和标签。
- 将基础参数写入 MLflow run，按阶段记录命令、耗时、成功标记。
- 自动上传 `pipeline_args.json`、推理结果 JSON、验证指标、可视化报告（若存在）。
- 解析 `metrics_summary.json`，把整体精度/召回以及 per-class 召回写入指标。

### 示例

```bash
python run_full_pipeline.py \
  --steps \
  123 \
  --base-path \
  outputs/valid/1120data/1215slice3mixedp1_SWRDpatch \
  --run-inference-opts \
  \"--image-dir /datasets/PAR/Xray/self/1120/labeled/roi2_merge/yolo_det_coco/valid --mode det --visualize --roi-weights ./model/weldROI3.pt --mode det --visualize --det-wide-slice --roi-weights ./model/weldROI3.pt --det-confidence 0.25  --primary-weights train/runs/1208/detrlarge/1215slice3mixedp1/checkpoint_best_regular.pth --class-names \\\"其他\\\" \\\"内凹\\\" \\\"咬边\\\" \\\"圆形缺陷\\\" \\\"未焊透\\\" \\\"未熔合\\\" \\\"条形缺陷\\\" \\\"裂纹\\\"  --det-secondary-variant large --secondary-weights train/runs/1208/detrlarge/mixedp1_patch640_res560_/checkpoint_best_regular.pth --det-patch-overlap 0.5 --det-fusion-iou 0.4 --patch-size 640 640 --det-secondary-confidence 0.3 \" \
  --validate-opts \
  \"--label-format coco --iou-threshold 0.01 --coco-json  /datasets/PAR/Xray/self/1120/labeled/roi2_merge/yolo_det_coco/valid/_annotations.coco.json --copy-images  --match-mode best\" \
  --visualize-opts \
  \"--title 焊缝检测报告\" \
  --mlflow \
  --mlflow-experiment \
  mixed50 \
  --mlflow-run-name \
  1215slice3mixedp1_SWRDpatch
```

执行完成后，推理结果位于 `<base-path>/<infer-subdir>`，验证与可视化输出位于 `<base-path>/<valid-subdir>`，可直接打开 `report.html` 浏览。

## 常见提示

- 首次运行前请确认 profile 中的脚本路径、模型文件、输出目录具有读写权限。
- 如果引用外部 `dataset.yaml`，务必保证其中的 `label_id_map` 完整，否则步骤1会抛出错误。
- `run_data_pipeline.py` 的任一步骤失败且未使用 `--force` 时流程会立即终止；排除问题后可再次执行剩余步骤。
- `run_full_pipeline.py` 要求 `run_inference_pipeline.py` 能在当前工作目录直接调用；若脚本位于其他位置，请在命令前加 `python path/to/run_inference_pipeline.py` 或使用虚拟环境确保可执行。
- 验证输出目录会额外生成 `verified_image/` 子目录，用于留存所有“漏标注”原图；若启用 `--export-verified-bundle`，还会生成 `verified_bundle/`，其中只包含需回传客户的样本及 `report.html`。
- 使用 MLflow 时请预先安装并配置 Tracking Server，否则会提示未安装或无法连接。

## DVC 实验流程（焊缝检测最小闭环）

以下流程演示 “Labelme → YOLO → 训练” 的 DVC 实验管理方式，配套 `dvc.yaml` 与 `params.yaml` 使用。

```bash
# 1) 切换数据版本（示例：用 git tag 切换 datasets/crack.dvc）
git checkout <tag> -- datasets/crack.dvc
dvc checkout

# 2) 运行完整实验（预处理 + 训练）
dvc exp run

# 3) 修改参数再跑一次（示例）
dvc exp run -S data.val_size=0.2 -S train.epochs=300 -S train.lr=0.0005

# 4) 对比实验
dvc exp show
```

产物说明：
- 预处理输出：`data/processed/crack_yolo`（不进 DVC cache）
- 训练输出：`outputs/welddetect/<run>`（只保留 best 权重）
- 指标文件：`metrics/welddetect.json`（由 DVC 追踪）
- MLflow 记录：`mlruns/`
