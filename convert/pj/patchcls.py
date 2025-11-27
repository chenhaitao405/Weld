import os
import sys
import cv2
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
from ultralytics import YOLO
import matplotlib.cm as cm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
for path in {SCRIPT_DIR, PROJECT_ROOT}:
    if str(path) not in sys.path:
        sys.path.append(str(path))

from convert.pj.yolo_roi_extractor import WeldROIDetector
from utils.image_processing import (
    enhance_image, sliding_window_crop, calculate_stride
)

# 默认参数
DEFAULT_OVERLAP_RATIO = 0.5
DEFAULT_WINDOW_SIZE = 640
DEFAULT_CONFIDENCE_THRESHOLD = 0.9
DEFAULT_ALPHA = 0.25  # 降低默认透明度，让原图更清晰
DEFAULT_ROI_MODEL_PATH = "./model/weldDetect.pt"
DEFAULT_ROI_CONFIDENCE = 0.25
DEFAULT_ROI_IOU = 0.45
DEFAULT_ROI_PADDING = 0.1
DEFAULT_DEFECT_CLASSES = (0,)


class SlicePredictHeatmapVisualizer:
    """图像切片预测及热力图可视化处理器"""

    def __init__(self,
                 model_path: str,
                 window_size: Tuple[int, int] = (DEFAULT_WINDOW_SIZE, DEFAULT_WINDOW_SIZE),
                 overlap_ratio: float = DEFAULT_OVERLAP_RATIO,
                 enhance_mode: str = 'original',
                 confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
                 use_confidence_weight: bool = False,
                 colormap: str = 'hot',
                 alpha: float = DEFAULT_ALPHA,
                 display_mode: str = 'overlay',
                 roi_model_path: Optional[str] = DEFAULT_ROI_MODEL_PATH,
                 roi_confidence: float = DEFAULT_ROI_CONFIDENCE,
                 roi_iou: float = DEFAULT_ROI_IOU,
                 roi_padding: float = DEFAULT_ROI_PADDING,
                 defect_class_ids: Optional[List[int]] = None):
        """
        初始化处理器`

        Args:
            model_path: YOLO模型路径
            window_size: 滑动窗口大小 (height, width)
            overlap_ratio: 窗口重叠率 (0.0-1.0)
            enhance_mode: 图像增强模式 ('original' 或 'windowing')
            confidence_threshold: 置信度阈值
            use_confidence_weight: 是否使用置信度加权（True时累加置信度值，False时累加次数）
            colormap: 热力图颜色映射（'hot', 'jet', 'turbo', 'viridis'等）
            alpha: 热力图叠加透明度（默认0.25，值越小原图越清晰）
            display_mode: 显示模式 ('overlay'=叠加, 'contour'=轮廓, 'sparse'=稀疏点)
            roi_model_path: 焊缝ROI检测模型路径，None时跳过ROI检测
            roi_confidence: ROI检测置信度阈值
            roi_iou: ROI检测IOU阈值
            roi_padding: ROI区域padding比例
            defect_class_ids: 被视为缺陷的分类ID列表（默认0）
        """
        self.model = YOLO(model_path)
        self.window_size = window_size
        self.overlap_ratio = overlap_ratio
        self.enhance_mode = enhance_mode
        self.confidence_threshold = confidence_threshold
        self.use_confidence_weight = use_confidence_weight
        self.colormap = colormap
        self.alpha = alpha
        self.display_mode = display_mode
        self.roi_model_path = roi_model_path
        self.roi_confidence = roi_confidence
        self.roi_iou = roi_iou
        self.roi_padding = roi_padding
        self.defect_class_ids = set(
            defect_class_ids if defect_class_ids is not None else DEFAULT_DEFECT_CLASSES
        )

        # 计算步长
        self.stride = calculate_stride(window_size, overlap_ratio)

        # 获取颜色映射
        self.cmap = cm.get_cmap(colormap)
        self.roi_detector: Optional[WeldROIDetector] = self._initialize_roi_detector()

        print(f"热力图切片预测可视化器初始化:")
        print(f"  - 模型路径: {model_path}")
        print(f"  - 窗口大小: {window_size}")
        print(f"  - 重叠率: {overlap_ratio}")
        print(f"  - 增强模式: {enhance_mode}")
        print(f"  - 置信度阈值: {confidence_threshold}")
        print(f"  - 置信度加权: {use_confidence_weight}")
        print(f"  - 颜色映射: {colormap}")
        print(f"  - 透明度: {alpha}")
        print(f"  - 显示模式: {display_mode}")
        print(f"  - ROI模型: {roi_model_path if roi_model_path else '未启用'}")
        print(f"  - ROI参数: conf={roi_confidence}, iou={roi_iou}, padding={roi_padding}")
        print(f"  - 缺陷类别: {sorted(self.defect_class_ids)}")

    def _initialize_roi_detector(self) -> Optional[WeldROIDetector]:
        """根据配置初始化ROI检测器"""
        if not self.roi_model_path:
            print("  - 未提供ROI模型，使用整图作为ROI区域")
            return None

        model_path = self.roi_model_path
        if model_path and not os.path.exists(model_path):
            print(f"  - ROI模型文件不存在: {model_path}，将跳过ROI检测")
            return None

        try:
            print(f"  - 加载ROI检测模型: {model_path}")
            return WeldROIDetector(
                model_path=model_path,
                roi_conf_threshold=self.roi_confidence,
                roi_iou_threshold=self.roi_iou,
                padding_ratio=self.roi_padding
            )
        except Exception as exc:  # pragma: no cover
            print(f"  - ROI模型加载失败: {exc}，使用整图作为ROI区域")
            return None

    def predict_single_patch(self, patch: np.ndarray) -> Tuple[int, float]:
        """
        对单个patch进行预测

        Args:
            patch: 图像patch

        Returns:
            (预测类别, 置信度)
        """
        # 使用YOLO模型预测
        results = self.model.predict(patch, verbose=False)

        # 获取第一个结果（因为只有一张图片）
        if results and len(results) > 0:
            res = results[0]

            # 检查是否有分类结果
            if hasattr(res, 'probs') and res.probs is not None:
                # 获取最高置信度的类别
                top1 = res.probs.top1
                top1conf = res.probs.top1conf.cpu().numpy()

                return int(top1), float(top1conf)

        return -1, 0.0

    def detect_roi_boxes(self, image: np.ndarray,
                         image_id: Optional[str] = None) -> List[Tuple[int, int, int, int]]:
        """检测焊缝ROI，若失败则回退到整图"""
        height, width = image.shape[:2]
        fallback_box = [(0, 0, width, height)]

        if self.roi_detector is None:
            return fallback_box

        boxes = self.roi_detector.detect_with_padding(image, (height, width))
        if not boxes:
            label = image_id if image_id else "当前图像"
            print(f"  - ROI检测未发现区域（{label}），使用整图作为ROI")
            return fallback_box

        return boxes

    def prepare_patch_for_classification(self, patch: np.ndarray) -> np.ndarray:
        """参考数据预处理流程，对patch进行增强并转换成YOLO分类可用的格式"""
        enhanced_patch = enhance_image(patch, mode=self.enhance_mode, output_bits=8)
        if len(enhanced_patch.shape) == 2:
            enhanced_patch = cv2.cvtColor(enhanced_patch, cv2.COLOR_GRAY2BGR)
        return np.ascontiguousarray(enhanced_patch)

    def detect_slice_classify(self,
                              image: np.ndarray,
                              image_id: Optional[str] = None) -> Dict[str, Any]:
        """封装步骤2~5：ROI检测->切片增强->分类->映射回原图"""
        roi_boxes = self.detect_roi_boxes(image, image_id=image_id)

        pipeline_result: Dict[str, Any] = {
            'image_id': image_id,
            'image_shape': image.shape[:2],
            'rois': [],
            'patch_predictions': [],
            'total_patches': 0,
            'defect_patches': 0
        }

        progress_bar = None

        try:
            for roi_index, (x1, y1, x2, y2) in enumerate(roi_boxes):
                roi_width = max(0, x2 - x1)
                roi_height = max(0, y2 - y1)
                if roi_width <= 0 or roi_height <= 0:
                    continue

                roi_info: Dict[str, Any] = {
                    'roi_index': roi_index,
                    'box': {
                        'x1': int(x1), 'y1': int(y1),
                        'x2': int(x2), 'y2': int(y2),
                        'width': int(roi_width), 'height': int(roi_height)
                    },
                    'num_patches': 0,
                    'defect_patches': 0,
                    'patches': []
                }
                pipeline_result['rois'].append(roi_info)

                roi_image = image[y1:y2, x1:x2]
                if roi_image.size == 0:
                    continue

                patches = sliding_window_crop(roi_image, self.window_size, self.stride)
                if not patches:
                    continue

                for patch_info in patches:
                    if progress_bar is None:
                        progress_bar = tqdm(desc="分类切片", unit="patch", leave=False)

                    prepared_patch = self.prepare_patch_for_classification(patch_info['patch'])
                    pred_class, confidence = self.predict_single_patch(prepared_patch)

                    is_defect = (
                        pred_class in self.defect_class_ids and
                        confidence >= self.confidence_threshold
                    )

                    rel_x, rel_y = patch_info['position']
                    patch_width = patch_info['size'][1]
                    patch_height = patch_info['size'][0]
                    abs_x = int(x1 + rel_x)
                    abs_y = int(y1 + rel_y)

                    patch_result = {
                        'roi_index': roi_index,
                        'position': (abs_x, abs_y),
                        'size': (int(patch_width), int(patch_height)),
                        'relative_position': (int(rel_x), int(rel_y)),
                        'confidence': float(confidence),
                        'class': int(pred_class),
                        'is_defect': bool(is_defect)
                    }

                    pipeline_result['patch_predictions'].append(patch_result)
                    roi_info['patches'].append(patch_result)
                    roi_info['num_patches'] += 1
                    pipeline_result['total_patches'] += 1

                    if is_defect:
                        roi_info['defect_patches'] += 1
                        pipeline_result['defect_patches'] += 1

                    progress_bar.update(1)
        finally:
            if progress_bar is not None:
                progress_bar.close()

        return pipeline_result

    def generate_heatmap(self,
                        image_shape: Tuple[int, int],
                        predictions: List[Dict]) -> np.ndarray:
        """
        生成热力图

        Args:
            image_shape: 原图尺寸 (height, width)
            predictions: 预测结果列表，每个元素包含position, size, confidence等

        Returns:
            热力图矩阵（累积值矩阵）
        """
        height, width = image_shape[:2]

        # 创建累积矩阵
        heatmap = np.zeros((height, width), dtype=np.float32)

        # 对每个预测为缺陷的切片进行累积
        for pred in predictions:
            is_defect = pred.get('is_defect')
            if is_defect is None:
                pred_class = pred.get('class', -1)
                pred_conf = pred.get('confidence', 0.0)
                is_defect = (
                    pred_class in self.defect_class_ids and
                    pred_conf >= self.confidence_threshold
                )

            if not is_defect:
                continue

            x, y = pred['position']
            w, h = pred['size']

            # 确保不超出边界
            x_end = min(x + w, width)
            y_end = min(y + h, height)

            # 累加值：使用置信度或固定值1
            if self.use_confidence_weight:
                increment = pred.get('confidence', 0.0)
            else:
                increment = 1.0

            # 在对应区域累加
            heatmap[y:y_end, x:x_end] += increment

        return heatmap

    def apply_colormap(self, heatmap: np.ndarray) -> np.ndarray:
        """
        将热力图矩阵转换为彩色图像

        Args:
            heatmap: 热力图累积矩阵

        Returns:
            彩色热力图图像 (BGR格式)
        """
        # 归一化到0-1范围
        if heatmap.max() > 0:
            normalized = heatmap / heatmap.max()
        else:
            normalized = heatmap

        # 应用颜色映射
        colored = self.cmap(normalized)

        # 转换为BGR格式（OpenCV使用）
        colored_bgr = (colored[:, :, [2, 1, 0]] * 255).astype(np.uint8)

        return colored_bgr

    def create_heatmap_overlay(self,
                               original_image: np.ndarray,
                               heatmap: np.ndarray,
                               apply_gaussian_blur: bool = True) -> np.ndarray:
        """
        创建热力图叠加图像

        Args:
            original_image: 原始图像
            heatmap: 热力图累积矩阵
            apply_gaussian_blur: 是否对热力图应用高斯模糊以平滑显示

        Returns:
            叠加后的图像
        """
        # 确保原图是3通道
        if len(original_image.shape) == 2:
            original_3ch = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        else:
            original_3ch = original_image.copy()

        # 根据显示模式处理
        if self.display_mode == 'contour':
            # 轮廓模式：只显示热力图的轮廓线
            return self._create_contour_overlay(original_3ch, heatmap)
        elif self.display_mode == 'sparse':
            # 稀疏模式：只在高值区域显示颜色
            return self._create_sparse_overlay(original_3ch, heatmap, apply_gaussian_blur)
        else:
            # 默认叠加模式，但使用更低的透明度
            return self._create_standard_overlay(original_3ch, heatmap, apply_gaussian_blur)

    def _create_standard_overlay(self, original_3ch, heatmap, apply_gaussian_blur):
        """标准叠加模式"""
        # 如果需要，应用高斯模糊使热力图更平滑
        if apply_gaussian_blur and heatmap.max() > 0:
            kernel_size = max(5, min(31, int(min(heatmap.shape) * 0.02) | 1))
            heatmap_smooth = cv2.GaussianBlur(heatmap, (kernel_size, kernel_size), 0)
        else:
            heatmap_smooth = heatmap

        # 生成彩色热力图
        colored_heatmap = self.apply_colormap(heatmap_smooth)

        # 创建更柔和的掩码
        mask = (heatmap_smooth > 0).astype(np.float32)

        # 根据热力值调整透明度，让低值区域更透明
        if heatmap_smooth.max() > 0:
            # 归一化热力值
            normalized_heat = heatmap_smooth / heatmap_smooth.max()
            # 使用非线性映射，让低值区域更透明
            alpha_mask = np.power(normalized_heat, 1.5) * self.alpha
            alpha_mask = np.stack([alpha_mask] * 3, axis=-1)
        else:
            alpha_mask = mask * self.alpha
            alpha_mask = np.stack([alpha_mask] * 3, axis=-1)

        # 叠加，使用动态透明度
        overlay = original_3ch.astype(np.float32) * (1 - alpha_mask) + \
                 colored_heatmap.astype(np.float32) * alpha_mask

        return overlay.astype(np.uint8)

    def _create_contour_overlay(self, original_3ch, heatmap):
        """轮廓模式：只显示热力区域的轮廓"""
        overlay = original_3ch.copy()

        if heatmap.max() > 0:
            # 归一化热力图
            normalized = (heatmap / heatmap.max() * 255).astype(np.uint8)

            # 创建多个等级的轮廓
            contour_levels = [30, 60, 90, 120, 150, 180, 210]
            colors = self.cmap(np.array(contour_levels) / 255.0)[:, :3]
            colors = (colors[:, [2, 1, 0]] * 255).astype(np.uint8)

            for i, level in enumerate(contour_levels):
                # 创建二值图
                _, binary = cv2.threshold(normalized, level, 255, cv2.THRESH_BINARY)

                # 找轮廓
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # 画轮廓
                color = colors[i].tolist()
                cv2.drawContours(overlay, contours, -1, color, 2)

        return overlay

    def _create_sparse_overlay(self, original_3ch, heatmap, apply_gaussian_blur):
        """稀疏模式：只在高值区域显示热力图"""
        overlay = original_3ch.copy()

        if heatmap.max() > 0:
            # 应用模糊
            if apply_gaussian_blur:
                kernel_size = max(5, min(31, int(min(heatmap.shape) * 0.02) | 1))
                heatmap_smooth = cv2.GaussianBlur(heatmap, (kernel_size, kernel_size), 0)
            else:
                heatmap_smooth = heatmap

            # 只显示高于某个阈值的区域
            threshold = heatmap_smooth.max() * 0.3  # 只显示前70%的热力值
            high_heat_mask = (heatmap_smooth > threshold).astype(np.float32)

            # 生成彩色热力图
            colored_heatmap = self.apply_colormap(heatmap_smooth)

            # 创建渐变透明度
            if heatmap_smooth.max() > threshold:
                alpha_values = (heatmap_smooth - threshold) / (heatmap_smooth.max() - threshold)
                alpha_mask = alpha_values * high_heat_mask * self.alpha * 1.5  # 稍微增强透明度
                alpha_mask = np.clip(alpha_mask, 0, self.alpha)
            else:
                alpha_mask = high_heat_mask * self.alpha

            alpha_mask = np.stack([alpha_mask] * 3, axis=-1)

            # 叠加
            overlay = original_3ch.astype(np.float32) * (1 - alpha_mask) + \
                     colored_heatmap.astype(np.float32) * alpha_mask

            overlay = overlay.astype(np.uint8)

        return overlay

    def add_colorbar(self, image: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
        """
        在图像旁边添加颜色条

        Args:
            image: 叠加后的图像
            heatmap: 热力图矩阵（用于获取值范围）

        Returns:
            带颜色条的图像
        """
        h, w = image.shape[:2]

        # 创建颜色条
        colorbar_width = max(30, int(w * 0.03))
        colorbar_height = int(h * 0.6)
        colorbar_x = w + 20

        # 创建扩展的画布
        extended_width = w + colorbar_width + 40
        canvas = np.ones((h, extended_width, 3), dtype=np.uint8) * 255

        # 复制原图到画布
        canvas[:, :w] = image

        # 生成颜色条
        colorbar_values = np.linspace(0, 1, colorbar_height).reshape(-1, 1)
        colorbar_values = np.repeat(colorbar_values, colorbar_width, axis=1)
        colorbar_colored = self.cmap(colorbar_values)
        colorbar_bgr = (colorbar_colored[:, :, [2, 1, 0]] * 255).astype(np.uint8)

        # 放置颜色条
        bar_y = (h - colorbar_height) // 2
        canvas[bar_y:bar_y+colorbar_height, colorbar_x:colorbar_x+colorbar_width] = colorbar_bgr

        # 添加刻度标签
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1

        # 添加最大值和最小值标签
        max_val = heatmap.max()
        min_val = 0

        # 最大值标签
        text = f"{max_val:.1f}"
        cv2.putText(canvas, text,
                   (colorbar_x + colorbar_width + 5, bar_y + 10),
                   font, font_scale, (0, 0, 0), font_thickness)

        # 最小值标签
        text = f"{min_val:.1f}"
        cv2.putText(canvas, text,
                   (colorbar_x + colorbar_width + 5, bar_y + colorbar_height - 5),
                   font, font_scale, (0, 0, 0), font_thickness)

        # 中间值标签
        mid_val = max_val / 2
        text = f"{mid_val:.1f}"
        cv2.putText(canvas, text,
                   (colorbar_x + colorbar_width + 5, bar_y + colorbar_height // 2),
                   font, font_scale, (0, 0, 0), font_thickness)

        return canvas

    def process_image(self,
                     image_path: str,
                     output_path: Optional[str] = None,
                     save_heatmap_only: bool = False,
                     add_colorbar: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        处理单张图像

        Args:
            image_path: 输入图像路径
            output_path: 输出图像路径（可选）
            save_heatmap_only: 是否只保存热力图（不叠加原图）
            add_colorbar: 是否添加颜色条

        Returns:
            (叠加后的图像, 热力图矩阵)
        """
        # 读取图像
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"无法读取图像: {image_path}")
            return None, None

        print(f"处理图像: {image_path}")
        print(f"  - 原始尺寸: {image.shape[:2]}")

        classification_result = self.detect_slice_classify(image, image_id=image_path)
        predictions = classification_result['patch_predictions']
        roi_count = len(classification_result['rois'])
        total_patches = classification_result['total_patches']
        defect_patches = classification_result['defect_patches']

        print(f"  - ROI数量: {roi_count}")
        print(f"  - 切片数量: {total_patches}")
        print(f"  - 缺陷切片数量(>= {self.confidence_threshold:.2f}): {defect_patches}")

        # 生成热力图
        heatmap = self.generate_heatmap(image.shape, predictions)
        print(f"  - 热力图最大累积值: {heatmap.max():.2f}")

        # 创建可视化结果
        if save_heatmap_only:
            # 只保存热力图
            result_image = self.apply_colormap(heatmap)
        else:
            # 创建叠加图像
            result_image = self.create_heatmap_overlay(image, heatmap, apply_gaussian_blur=True)

        # 添加颜色条
        if add_colorbar and heatmap.max() > 0:
            result_image = self.add_colorbar(result_image, heatmap)

        # 保存结果图像
        if output_path:
            cv2.imwrite(output_path, result_image)
            print(f"  - 结果已保存至: {output_path}")

            # 可选：同时保存原始热力图数据（numpy格式）
            heatmap_data_path = str(Path(output_path).parent /
                                   f"{Path(output_path).stem}_heatmap.npy")
            np.save(heatmap_data_path, heatmap)
            print(f"  - 热力图数据已保存至: {heatmap_data_path}")

        return result_image, heatmap

    def process_directory(self, input_dir: str, output_dir: str):
        """
        处理图像目录

        Args:
            input_dir: 输入目录路径
            output_dir: 输出目录路径
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)

        # 创建输出目录
        output_path.mkdir(parents=True, exist_ok=True)

        # 支持的图像格式
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']

        # 获取所有图像文件
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f'*{ext}'))
            image_files.extend(input_path.glob(f'*{ext.upper()}'))

        if not image_files:
            print(f"在{input_dir}中未找到图像文件")
            return

        print(f"找到{len(image_files)}张图像，开始处理...")

        # 处理每张图像
        for image_file in image_files:
            output_file = output_path / f"{image_file.stem}_heatmap{image_file.suffix}"
            self.process_image(str(image_file), str(output_file))
            print()

        print(f"所有图像处理完成，结果保存在: {output_dir}")

    def generate_statistics(self, heatmap: np.ndarray) -> Dict:
        """
        生成热力图统计信息

        Args:
            heatmap: 热力图矩阵

        Returns:
            统计信息字典
        """
        stats = {
            'max_value': float(heatmap.max()),
            'mean_value': float(heatmap.mean()),
            'std_value': float(heatmap.std()),
            'non_zero_pixels': int(np.sum(heatmap > 0)),
            'total_pixels': int(heatmap.size),
            'coverage_ratio': float(np.sum(heatmap > 0) / heatmap.size),
            'high_confidence_area': int(np.sum(heatmap > heatmap.max() * 0.7))
        }

        return stats


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description='YOLO图像切片预测及热力图可视化工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 处理单张图像（基础用法，默认透明度0.25）
  python slice_predict_heatmap.py --image_path ./test.jpg --model_path ./best.pt --output_path ./output.jpg

  # 使用更高的透明度让原图更清晰（推荐用于缺陷检测）
  python slice_predict_heatmap.py --image_path ./test.jpg --model_path ./best.pt --alpha 0.15

  # 使用轮廓模式（只显示轮廓线，不遮挡原图）
  python slice_predict_heatmap.py --image_path ./test.jpg --model_path ./best.pt --display_mode contour

  # 使用稀疏模式（只在高置信度区域显示热力图）
  python slice_predict_heatmap.py --image_path ./test.jpg --model_path ./best.pt --display_mode sparse

  # 处理图像目录
  python slice_predict_heatmap.py --image_dir ./images --model_path ./best.pt --output_dir ./outputs

  # 使用置信度加权的热力图
  python slice_predict_heatmap.py --image_path ./test.jpg --model_path ./best.pt --use_confidence_weight

  # 自定义热力图颜色
  python slice_predict_heatmap.py --image_path ./test.jpg --model_path ./best.pt --colormap jet

  # 只保存热力图（不叠加原图）
  python slice_predict_heatmap.py --image_path ./test.jpg --model_path ./best.pt --heatmap_only

  # 不添加颜色条
  python slice_predict_heatmap.py --image_path ./test.jpg --model_path ./best.pt --no_colorbar
        """
    )

    # 输入参数（二选一）
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image_path', type=str,
                             help='输入图像路径')
    input_group.add_argument('--image_dir', type=str,
                             help='输入图像目录')

    # 输出参数
    parser.add_argument('--output_path', type=str,
                        help='输出图像路径（处理单张图像时使用）')
    parser.add_argument('--output_dir', type=str,
                        help='输出目录路径（处理目录时使用）')

    # 模型参数
    parser.add_argument('--model_path', type=str, required=True,
                        help='YOLO模型路径')
    parser.add_argument('--roi_model_path', type=str,
                        default=DEFAULT_ROI_MODEL_PATH,
                        help='焊缝ROI检测模型路径 (默认: ./model/weldDetect.pt)')
    parser.add_argument('--roi_conf', type=float, default=DEFAULT_ROI_CONFIDENCE,
                        help='ROI检测置信度阈值')
    parser.add_argument('--roi_iou', type=float, default=DEFAULT_ROI_IOU,
                        help='ROI检测IOU阈值')
    parser.add_argument('--roi_padding', type=float, default=DEFAULT_ROI_PADDING,
                        help='ROI区域padding比例')

    # 处理参数
    parser.add_argument('--window_size', type=int, nargs=2,
                        default=[DEFAULT_WINDOW_SIZE, DEFAULT_WINDOW_SIZE],
                        help='滑动窗口大小 [height width]')
    parser.add_argument('--overlap', type=float, default=DEFAULT_OVERLAP_RATIO,
                        help='滑动窗口重叠率 (0.0-1.0)')
    parser.add_argument('--enhance_mode', type=str,
                        choices=['original', 'windowing'],
                        default='windowing',
                        help='图像增强模式')
    parser.add_argument('--confidence', type=float,
                        default=DEFAULT_CONFIDENCE_THRESHOLD,
                        help='置信度阈值')
    parser.add_argument('--defect_classes', type=int, nargs='+',
                        default=list(DEFAULT_DEFECT_CLASSES),
                        help='被视为缺陷的分类ID (空格分隔)')

    # 热力图参数
    parser.add_argument('--use_confidence_weight', action='store_true',
                        help='使用置信度加权（否则使用计数）')
    parser.add_argument('--colormap', type=str, default='jet',
                        choices=['hot', 'jet', 'turbo', 'viridis', 'plasma',
                                'coolwarm', 'RdYlBu', 'YlOrRd'],
                        help='热力图颜色映射')
    parser.add_argument('--alpha', type=float, default=DEFAULT_ALPHA,
                        help='热力图叠加透明度 (0.0-1.0，默认0.25，值越小原图越清晰)')
    parser.add_argument('--display_mode', type=str, default='overlay',
                        choices=['overlay', 'contour', 'sparse'],
                        help='显示模式: overlay=叠加, contour=轮廓, sparse=稀疏')
    parser.add_argument('--heatmap_only', action='store_true',
                        help='只保存热力图，不叠加原图')
    parser.add_argument('--no_colorbar', action='store_true',
                        help='不添加颜色条')

    args = parser.parse_args()

    # 参数验证
    if args.image_path and not args.output_path:
        # 如果没有指定输出路径，自动生成
        input_path = Path(args.image_path)
        args.output_path = str(input_path.parent / f"{input_path.stem}_heatmap{input_path.suffix}")

    if args.image_dir and not args.output_dir:
        # 如果没有指定输出目录，自动生成
        args.output_dir = str(Path(args.image_dir).parent / f"{Path(args.image_dir).name}_heatmap")

    # 创建处理器
    visualizer = SlicePredictHeatmapVisualizer(
        model_path=args.model_path,
        window_size=tuple(args.window_size),
        overlap_ratio=args.overlap,
        enhance_mode=args.enhance_mode,
        confidence_threshold=args.confidence,
        use_confidence_weight=args.use_confidence_weight,
        colormap=args.colormap,
        alpha=args.alpha,
        display_mode=args.display_mode,
        roi_model_path=args.roi_model_path,
        roi_confidence=args.roi_conf,
        roi_iou=args.roi_iou,
        roi_padding=args.roi_padding,
        defect_class_ids=args.defect_classes
    )

    # 处理图像
    if args.image_path:
        # 处理单张图像
        result_image, heatmap = visualizer.process_image(
            args.image_path,
            args.output_path,
            save_heatmap_only=args.heatmap_only,
            add_colorbar=not args.no_colorbar
        )

        # 输出统计信息
        if heatmap is not None:
            stats = visualizer.generate_statistics(heatmap)
            print("\n热力图统计信息:")
            print(f"  - 最大累积值: {stats['max_value']:.2f}")
            print(f"  - 平均值: {stats['mean_value']:.2f}")
            print(f"  - 标准差: {stats['std_value']:.2f}")
            print(f"  - 覆盖率: {stats['coverage_ratio']*100:.1f}%")
            print(f"  - 高置信区域像素数: {stats['high_confidence_area']}")

    else:
        # 处理目录
        visualizer.process_directory(args.image_dir, args.output_dir)


if __name__ == "__main__":
    main()
