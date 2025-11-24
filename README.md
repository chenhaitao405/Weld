# 焊缝检测与缺陷识别系统

## 📌 项目概述

本项目提供了一套完整的焊缝X射线图像处理工具链，包括数据预处理、格式转换、区域提取和模型训练等功能，支持从原始图像到YOLO格式数据集的全流程处理。

## 🛠️ 核心脚本功能

### 1. 数据预处理脚本

#### `WeldImagePreprocessor_withYOLO.py`
- **功能**：按照SWRD数据集的目录结构对焊缝区域图像进行处理
- **主要操作**：
  - 滑窗裁剪
  - 图像增强
  - 标签格式转换
  - 按YOLO格式要求组织输出目录

### 2. 格式转换工具

#### `convert/labelme2yolo.py`
- **功能**：将LabelMe的JSON标注文件转换为YOLO格式
- **适用场景**：通用的标注格式转换

#### `convert/labelme2yolo_cj.py`
- **功能**：专门处理长江项目的数据目录
- **特点**：适配带有"convert"后缀的标签文件命名规则

### 3. ROI区域提取

#### `convert/pj/yolo_roi_extractor.py`
- **功能**：使用焊缝检测网络识别原始X射线图像中的ROI区域
- **特性**：
  - 自动识别焊缝区域
  - 转换并保留原YOLO格式标签
  - 生成仅包含焊缝区域的训练数据集

### 4. 数据增强工具

#### `convert/pj/patchandenhance.py`
- **功能**：对YOLO格式的焊缝区域训练数据进行处理
- **操作**：
  - 滑动窗口裁剪
  - 数据增强

## 📋 使用流程

### 方案一：基于SWRD开源数据集训练

1. **数据预处理**
   ```bash
   python WeldImagePreprocessor_withYOLO.py
   ```
   - 执行滑窗裁剪、图像增强、标签转换
   - 直接生成YOLO训练所需的数据格式
   - 保存在`/home/lenovo/code/CHT/detect/dataprocess/preprocessed_data2/YOLODataset_seg/` 

2. **模型训练**
   ```bash
   python ultralytics/SWRDCrak.py
   ```
   - 使用ultralytics框架进行分割模型训练

### 方案二：基于项目实际数据训练

1. **焊缝检测模型训练**
   
   a. 转换焊缝区域检测数据集
   ```bash
   python convert/labelme2yolo.py --input /home/lenovo/code/CHT/datasets/Xray/labelConvert/crack
   ```
   
   b. 训练焊缝检测模型
   ```bash
   python ultralytics/welddet.py
   ```

2. **缺陷识别模型训练**
   
   a. 数据清理
   - 手动删除 `/home/lenovo/code/CHT/datasets/Xray/labelConvert/data_out` 中的无效数据
   
   b. 格式转换
   ```bash
   python convert/labelme2yolo_cj.py --input /home/lenovo/code/CHT/datasets/Xray/labelConvert/data_out
   ```
   
   c. ROI区域提取
   ```bash
   python convert/pj/yolo_roi_extractor.py
   ```
   - 使用焊缝检测模型识别缺陷标注数据集的ROI区域
   - 生成对应的YOLO格式标签
   
   d. 数据增强
   ```bash
   python convert/pj/patchandenhance.py
   ```
   - 输出目录：`/home/lenovo/code/CHT/datasets/Xray/weld_patch_yolo_output`
   
    若训练分类网络，调用：
   ```bash
   python convert/pj/seg2det.py
   ```
   
   e. 模型训练
   ```bash
   python ultralytics/crackdet.py
   ```
   - 训练缺陷分割模型



## 🔍 数据路径说明

| 数据类型 | 路径 |
|---------|------|
| 焊缝检测数据集 | `/home/lenovo/code/CHT/datasets/Xray/labelConvert/crack` |
| 缺陷标注数据集 | `/home/lenovo/code/CHT/datasets/Xray/labelConvert/data_out` |
| 最终输出数据集 | `/home/lenovo/code/CHT/datasets/Xray/weld_patch_yolo_output` |

## 📝 注意事项

- 在进行缺陷识别模型训练前，请确保已手动清理无效数据
- 长江项目数据需使用专用的转换脚本 `labelme2yolo_cj.py`5yy
- 确保ultralytics框架已正确安装并配置
