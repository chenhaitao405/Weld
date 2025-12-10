# 焊缝检测数据处理流水线

## 项目概述

本项目提供了一套完整的焊缝X射线图像数据处理流水线，能够自动化执行从原始标注数据到模型训练所需格式的全流程处理。通过简单的配置和命令行参数，即可完成数据格式转换、ROI区域提取、竖图旋转归一、图像裁剪增强和训练任务转换等关键步骤，大幅提高数据处理效率。

## 核心功能

- 自动化数据处理流水线，支持分步执行或全流程运行
- Labelme标注格式转YOLO训练格式
- 基于预训练模型的焊缝ROI区域自动提取
- 竖图自动旋转，保证ROI数据方向一致
- 图像裁剪与增强，提升模型泛化能力
- 分割任务与检测/分类任务间的格式转换

## 环境要求

- Python 3.x
- 依赖库：`tqdm`、`opencv-python`、`ultralytics`等（需自行安装）
- 支持Windows和Linux操作系统（需在配置中正确设置路径）

## 配置说明

核心配置位于`run_data_pipeline.py`的配置区域，主要包括：

1. **路径配置**：根据操作系统自动选择数据路径、模型路径
2. **数据集配置**：指定需要处理的数据集列表
3. **输出目录配置**：定义各步骤输出结果的保存路径
4. **固定参数配置**：各处理步骤的具体参数设置

默认配置支持的数据集：`D1`、`D2`、`D3`、`D4`、`img20250608`、`img20250609`

## 使用方法

### 数据处理流水线（`run_data_pipeline.py`）

**基本命令**

```bash
python run_data_pipeline.py --steps [步骤编号] [--force]
```

**步骤说明**

1. **Labelme转YOLO**（步骤1）
   - 将Labelme格式标签批量转换到统一的YOLO语义分割格式，并可选`unify_to_crack`
   - 自动收集所有数据集的标签生成`label_id_map`，写入`dataset.yaml`
2. **YOLO ROI提取**（步骤2）
   - 通过`convert/pj/yolo_roi_extractor.py`和指定`weldROI2.pt`模型抽取ROI，并记录运行参数
3. **竖图旋转**（步骤3）
   - 将竖直图像统一旋转到水平方向，保持标签一致
4. **图像裁剪与增强**（步骤4）
   - 滑窗裁剪、增强ROI，可设置窗口、overlap、窗宽窗位以及`--slice_mode`（1=仅增强，2=滑动裁剪，3=横裁纵拼方形）
5. **训练任务转换**（步骤5）
   - 使用`seg2det.py`在分割、检测、分类任务间转换，并支持`--balance_data`

**示例命令**

```bash
# 运行所有步骤
python run_data_pipeline.py --steps 12345

# 仅执行步骤2~5
python run_data_pipeline.py --steps 2345

# 跳过缺失输入继续执行
python run_data_pipeline.py --steps 23 --force
```

执行期间会在`BASE_PATH/roi2_unify/pipeline_params.json`记录所选步骤、命令行与参数，便于追踪。

### 推理→验证→可视化流水线（`run_full_pipeline.py`）

该脚本按顺序调用：

1. `run_inference_pipeline.py`
2. `validate_inference_results.py`
3. `visualize_validation_results.py`

**核心参数**

- `--base-path`：三阶段产物的根目录，默认`outputs/pipeline_run`
- `--infer-subdir`、`--valid-subdir`：推理和验证子目录，默认`infer`/`valid`
- `--inference-results`：推理结果文件名（默认`inference_results.json`）
- `--run-inference-opts`：传给`run_inference_pipeline.py`的追加参数，**必须包含**`--image-dir`
- `--validate-opts`、`--visualize-opts`：分别透传给验证与可视化脚本
- `--steps`：选择执行 1=推理 2=验证 3=可视化
- `--dry-run`：仅打印命令，便于检查配置

脚本会根据提供的参数自动补齐缺失的`--output-dir`、`--results-json`、`--inference-json`等选项，并在`<base-path>/pipeline_args.json`中记录当前流水线配置。所有阶段完成后，可在`<base-path>/<valid-subdir>/report.html`查看汇总可视化结果。

**示例命令**

```bash
python run_full_pipeline.py \
  --base-path outputs/pipeline_run \
  --run-inference-opts "--image-dir /data/images --weights runs/best.pt --device 0" \
  --validate-opts "--score-threshold 0.3" \
  --visualize-opts "--topk 50" \
  --steps 123
```

仅准备命令不执行，可加`--dry-run`用于生成待运行命令清单。

## 输出说明

处理结果默认保存在`BASE_PATH/roi2_unify`（可通过`OUTPUT_BASE_DIR`修改），包含以下子目录：

- `yolo`：步骤1输出，统一标签的YOLO分割数据
- `ROI`：步骤2输出，基于检测模型提取的ROI
- `ROI_rotate`：步骤3输出，方向统一后的ROI数据
- `patch`：步骤4输出，裁剪/增强后的图像块
- `det`（或`cls_dir`配置名）：步骤5输出，适配检测/分类训练的数据

`run_full_pipeline.py`将按照`<base-path>/<infer-subdir>`与`<valid-subdir>`组织推理与验证产物，在`report.html`中汇总评估结果。

## 注意事项

- 首次运行前请确保配置区域的路径设置正确，特别是模型路径
- 处理大型数据集时可能需要较长时间，请耐心等待
- 若某一步骤失败，建议检查输入数据格式和完整性后再重试
- 可通过`--force`参数强制重新执行某一步骤，但需确保相关依赖数据正确

## 扩展与定制

如需扩展功能，可修改以下配置：

- 在`DATASETS`列表中添加新的数据集名称
- 调整`FIXED_PARAMS`中的参数以优化处理效果
- 修改`OUTPUT_BASE_DIR`更改输出根目录
- 在`STEP_INFO`中添加新的处理步骤（需同时实现对应的处理函数）
