import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
# 设置matplotlib支持中文
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False
from pathlib import Path
import json
from datetime import datetime
import platform

def imread_universal(file_path, flags=cv2.IMREAD_COLOR):
    """跨平台读取图片函数，支持不同的读取模式"""
    if platform.system() == "Windows":
        # Windows下使用numpy+imdecode方式
        with open(file_path, 'rb') as f:
            img_data = f.read()
        img_array = np.frombuffer(img_data, np.uint8)
        return cv2.imdecode(img_array, flags)
    else:
        # Linux/Mac下直接使用cv2.imread
        return cv2.imread(file_path, flags)

class NegativeTransform:
    """负片预处理类 - 与训练脚本保持一致"""
    
    def __init__(self, apply_probability=1.0):
        """
        初始化负片变换
        Args:
            apply_probability: 应用负片变换的概率 (0-1之间)
        """
        self.apply_probability = apply_probability
    
    def apply_negative(self, image):
        """
        对图像应用负片变换
        Args:
            image: 输入图像 (H, W, C) RGB格式，数值范围0-255
        Returns:
            负片变换后的图像
        """
        # 负片变换公式: output = 255 - input
        negative_image = 255 - image
        return negative_image.astype(np.uint8)
    
    def __call__(self, image, mask=None):
        """
        Albumentations兼容的调用方式
        """
        # 根据概率决定是否应用负片变换
        if np.random.random() < self.apply_probability:
            image_negative = self.apply_negative(image)
        else:
            image_negative = image
        
        if mask is not None:
            return {"image": image_negative, "mask": mask}
        else:
            return {"image": image_negative}

class WeldSegmentationModel:
    
    def __init__(self, num_classes=2, encoder_name='resnet50', use_negative=True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_negative = use_negative
        
        # 初始化负片处理器 - 与训练脚本保持一致
        if use_negative:
            self.negative_processor = NegativeTransform(apply_probability=1.0)
            print("✅ 负片预处理已启用 (apply_probability=1.0)")
        
        # 创建模型 - 使用与训练脚本一致的ResNet50
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,  # 不使用预训练权重，因为要加载自己的模型
            in_channels=3,
            classes=num_classes,
        ).to(self.device)
        
        print(f"模型已创建，使用设备: {self.device}")
        print(f"编码器: {encoder_name}")
        print(f"负片预处理: {'启用' if use_negative else '禁用'}")
    
    def load_model(self, model_path):
        """加载训练好的模型"""
        if not os.path.exists(model_path):
            print(f"错误: 模型文件 {model_path} 不存在")
            return False
            
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✅ 模型已加载: {model_path}")
        print(f"   最佳Dice: {checkpoint.get('best_dice', 'N/A'):.4f}")
        if 'best_iou' in checkpoint:
            print(f"   最佳IoU: {checkpoint['best_iou']:.4f}")
        return True
    
    def predict_single_image(self, image_path, transform):
        """预测单张图片（包含负片预处理）"""
        # 读取图片
        image = imread_universal(str(image_path))
        if image is None:
            return None, None, None
            
        original_size = (image.shape[1], image.shape[0])  # (width, height)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 应用负片预处理 - 与训练脚本保持一致
        image_enhanced = image_rgb.copy()
        if self.use_negative:
            processed = self.negative_processor(image_rgb)
            image_enhanced = processed['image']
        
        # 其他预处理
        transformed = transform(image=image_enhanced)
        input_tensor = transformed['image'].unsqueeze(0).to(self.device)
        
        # 预测
        with torch.no_grad():
            output = self.model(input_tensor)
            pred = torch.softmax(output, dim=1)
            pred_mask = torch.argmax(pred, dim=1).cpu().numpy()[0]
            confidence = torch.max(pred, dim=1)[0].cpu().numpy()[0]
        
        # 恢复原始尺寸
        if pred_mask.shape != original_size[::-1]:  # (height, width)
            pred_mask = cv2.resize(pred_mask.astype(np.uint8), original_size, interpolation=cv2.INTER_NEAREST)
            confidence = cv2.resize(confidence, original_size, interpolation=cv2.INTER_LINEAR)
        
        return pred_mask, confidence, image_enhanced

def check_weld_detection(pred_mask, min_weld_pixels=500):
    """
    检查是否检测到焊缝
    Args:
        pred_mask: 预测掩码
        min_weld_pixels: 最小焊缝像素数阈值
    Returns:
        bool: True表示检测到焊缝，False表示未检测到
    """
    weld_pixels = np.sum(pred_mask == 1)
    return weld_pixels >= min_weld_pixels

def save_no_weld_images(no_weld_images, output_dir, prefix=""):
    """
    保存未检测到焊缝的图片名称到文本文件
    Args:
        no_weld_images: 未检测到焊缝的图片名称列表
        output_dir: 输出目录
        prefix: 文件名前缀
    """
    if not no_weld_images:
        print("📋 所有图片都检测到了焊缝")
        return None
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 创建带时间戳的文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}no_weld_detected_{timestamp}.txt"
    file_path = output_dir / filename
    
    # 保存到文本文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(f"未检测到焊缝的图片列表\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总计: {len(no_weld_images)} 张图片\n")
        f.write("=" * 50 + "\n\n")
        
        for i, img_name in enumerate(no_weld_images, 1):
            f.write(f"{i:3d}. {img_name}\n")
    
    print(f"📝 未检测焊缝图片列表已保存: {file_path}")
    print(f"📊 总计 {len(no_weld_images)} 张图片未检测到焊缝")
    
    return file_path

def get_rotated_bounding_box(pred_mask, image_to_draw_on, scale_factor=1.2):
    mask_uint8 = pred_mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return image_to_draw_on, []

    all_rects = []

    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue

        # 1. 获取原始旋转矩形信息
        rect = cv2.minAreaRect(contour)
        
        # 2. 提取信息并创建新的、扩大后的矩形
        center, size, angle = rect
        new_size = (size[0] * scale_factor, size[1] * scale_factor)
        
        # 创建新的旋转矩形对象
        new_rect = (center, new_size, angle)
        all_rects.append(new_rect)
        
        # 3. 获取新矩形的四个顶点坐标并绘制
        box = cv2.boxPoints(new_rect)
        box = np.intp(box)
        cv2.drawContours(image_to_draw_on, [box], 0, (0, 0, 255), 10)

    print(f"找到并绘制了 {len(all_rects)} 个扩大后的旋转矩形区域。")
    return image_to_draw_on, all_rects

def get_bounding_box_from_mask(pred_mask, image_to_draw_on, scale_factor=1.2):
    mask_uint8 = pred_mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("在掩码中未找到任何轮廓。")
        return image_to_draw_on, []

    all_boxes = []

    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue

        # 1. 获取原始矩形
        x, y, w, h = cv2.boundingRect(contour)

        # 2. 计算新的宽度和高度
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)

        # 3. 计算新的左上角坐标，以保持中心点不变
        new_x = int(x - (new_w - w) / 2)
        new_y = int(y - (new_h - h) / 2)

        # 4. 将新坐标存入列表
        all_boxes.append((new_x, new_y, new_w, new_h))

        # 5. 在图像上绘制扩大后的新矩形
        cv2.rectangle(image_to_draw_on, (new_x, new_y), (new_x + new_w, new_y + new_h), (0, 255, 0), 10)
    
    print(f"找到并绘制了 {len(all_boxes)} 个扩大后的有效矩形区域。")
    return image_to_draw_on, all_boxes

def get_test_transform():
    """获取测试时的变换 - 与训练脚本保持一致"""
    return A.Compose([
        A.Resize(512, 512),  # 使用与训练一致的512x512分辨率
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def visualize_prediction(image_path, pred_mask, confidence, enhanced_image, output_dir, use_negative=True):
    """可视化单个预测结果（包含负片对比）"""
    # 读取原图
    image = imread_universal(str(image_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_to_draw_on, rect = get_rotated_bounding_box(pred_mask, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image_to_draw_on1, rect1 = get_bounding_box_from_mask(pred_mask, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # 创建叠加图
    overlay = enhanced_image.copy()
    
    # 焊缝区域用红色标出
    weld_mask = (pred_mask == 1)
    overlay[weld_mask] = overlay[weld_mask] * 0.6 + np.array([255, 0, 0]) * 0.4
    
    # 绘制边界
    contours, _ = cv2.findContours(weld_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay_bgr = cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.drawContours(overlay_bgr, contours, -1, (0, 255, 0), 2)
    overlay = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    
    # 计算统计信息
    total_pixels = pred_mask.shape[0] * pred_mask.shape[1]
    weld_pixels = np.sum(weld_mask)
    weld_ratio = weld_pixels / total_pixels * 100
    avg_confidence = np.mean(confidence[weld_mask]) if weld_pixels > 0 else 0
    
    # 创建可视化（包含负片对比）
    if use_negative:
        fig, axes = plt.subplots(3, 3, figsize=(18, 12))
        
        # 第一行：原图、负片增强、预测掩码
        axes[0, 0].imshow(image_rgb)
        axes[0, 0].set_title('原始图片', fontsize=12)
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(enhanced_image)
        axes[0, 1].set_title('负片变换后 (255 - input)', fontsize=12)
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(pred_mask, cmap='gray')
        axes[0, 2].set_title('预测掩码', fontsize=12)
        axes[0, 2].axis('off')
        
        # 第二行：叠加结果、置信度图、对比图
        axes[1, 0].imshow(overlay)
        axes[1, 0].set_title('检测结果叠加', fontsize=12)
        axes[1, 0].axis('off')
        
        im = axes[1, 1].imshow(confidence, cmap='jet', vmin=0, vmax=1)
        axes[1, 1].set_title('置信度图', fontsize=12)
        axes[1, 1].axis('off')
        plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
        
        # 负片前后对比
        comparison = np.hstack([image_rgb[:, :image_rgb.shape[1]//2], 
                               enhanced_image[:, enhanced_image.shape[1]//2:]])
        axes[1, 2].imshow(comparison)
        axes[1, 2].set_title('负片前后对比\n(左:原图 右:负片)', fontsize=12)
        axes[1, 2].axis('off')

        # 第三行：边界框检测结果
        axes[2, 0].imshow(image_to_draw_on)
        axes[2, 0].set_title('旋转边界框检测', fontsize=12)
        axes[2, 0].axis('off')

        axes[2, 1].imshow(image_to_draw_on1)
        axes[2, 1].set_title('普通边界框检测', fontsize=12)
        axes[2, 1].axis('off')
        
        # 空白位置显示统计信息
        axes[2, 2].axis('off')
        stats_text = f'焊缝像素: {weld_pixels:,}\n总像素: {total_pixels:,}\n焊缝占比: {weld_ratio:.2f}%\n平均置信度: {avg_confidence:.3f}'
        axes[2, 2].text(0.1, 0.5, stats_text, fontsize=14, transform=axes[2, 2].transAxes, 
                        verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        
    else:
        # 不使用负片时的简化版本
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        axes[0, 0].imshow(image_rgb)
        axes[0, 0].set_title('原始图片', fontsize=12)
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(pred_mask, cmap='gray')
        axes[0, 1].set_title('预测掩码', fontsize=12)
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(overlay)
        axes[1, 0].set_title('叠加结果', fontsize=12)
        axes[1, 0].axis('off')
        
        im = axes[1, 1].imshow(confidence, cmap='jet', vmin=0, vmax=1)
        axes[1, 1].set_title('置信度图', fontsize=12)
        axes[1, 1].axis('off')
        plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    # 添加统计信息
    negative_info = " (负片增强)" if use_negative else ""
    info_text = f'焊缝像素: {weld_pixels:,}\n总像素: {total_pixels:,}\n焊缝占比: {weld_ratio:.2f}%\n平均置信度: {avg_confidence:.3f}'
    fig.suptitle(f'{Path(image_path).name}{negative_info}\n{info_text}', fontsize=11)
    
    plt.tight_layout()
    
    # 保存结果
    output_path = output_dir / f'result_{Path(image_path).stem}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return {
        'weld_pixels': int(weld_pixels),
        'total_pixels': int(total_pixels),
        'weld_ratio': float(weld_ratio),
        'avg_confidence': float(avg_confidence),
        'num_contours': len(contours),
        'has_weld': weld_pixels >= 500  # 添加焊缝检测标志
    }

def save_negative_comparison(original_image, enhanced_image, image_name, output_dir):
    """保存负片前后对比图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    ax1.imshow(original_image)
    ax1.set_title('原始图像')
    ax1.axis('off')
    
    ax2.imshow(enhanced_image)
    ax2.set_title('负片变换后 (255 - input)')
    ax2.axis('off')
    
    plt.tight_layout()
    
    comparison_path = output_dir / f'negative_comparison_{Path(image_name).stem}.png'
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return comparison_path

def test_on_images(model, image_dir, output_dir, max_images=None, min_weld_pixels=500):
    """测试模型在图片目录上的表现（支持负片预处理和未检测记录）"""
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 创建子目录
    (output_dir / 'predictions').mkdir(exist_ok=True)
    if model.use_negative:
        (output_dir / 'negative_comparisons').mkdir(exist_ok=True)
    
    # 获取所有图片
    image_extensions = ['.bmp', '.jpg', '.jpeg', '.png', '.tif']
    all_images = []
    for ext in image_extensions:
        all_images.extend(list(image_dir.glob(f'*{ext}')))
    
    if max_images:
        all_images = all_images[:max_images]
    
    if not all_images:
        print(f"在 {image_dir} 中没有找到图片文件")
        return
    
    print(f"找到 {len(all_images)} 张图片，开始预测...")
    print(f"负片预处理: {'启用' if model.use_negative else '禁用'}")
    print(f"最小焊缝像素阈值: {min_weld_pixels}")
    
    transform = get_test_transform()
    results = []
    no_weld_images = []  # 存储未检测到焊缝的图片名称
    
    for i, img_path in enumerate(all_images):
        print(f"处理 {i+1}/{len(all_images)}: {img_path.name}")
        
        # 预测
        pred_result = model.predict_single_image(img_path, transform)
        if pred_result[0] is None:
            print(f"  跳过（无法读取图片）")
            continue
        
        pred_mask, confidence, enhanced_image = pred_result
        
        # 检查是否检测到焊缝
        has_weld = check_weld_detection(pred_mask, min_weld_pixels)
        if not has_weld:
            no_weld_images.append(img_path.name)
            print(f"  ⚠️ 未检测到焊缝（像素数: {np.sum(pred_mask == 1)}）")
        
        # 可视化和统计
        stats = visualize_prediction(
            img_path, pred_mask, confidence, enhanced_image, 
            output_dir / 'predictions', model.use_negative
        )
        stats['image_name'] = img_path.name
        stats['has_weld'] = has_weld
        results.append(stats)
        
        # 保存负片对比图（前5张作为示例）
        if model.use_negative and i < 5:
            original_image = cv2.cvtColor(imread_universal(str(img_path)), cv2.COLOR_BGR2RGB)
            save_negative_comparison(
                original_image, enhanced_image, img_path.name, 
                output_dir / 'negative_comparisons'
            )
        
        print(f"  焊缝占比: {stats['weld_ratio']:.2f}%, 置信度: {stats['avg_confidence']:.3f}")
    
    # 保存统计报告
    save_test_report(results, output_dir, model.use_negative)
    
    # 保存未检测到焊缝的图片列表
    no_weld_file = save_no_weld_images(no_weld_images, output_dir, 
                                       "negative_" if model.use_negative else "")
    
    # 打印总结
    total_tested = len(results)
    no_weld_count = len(no_weld_images)
    detection_rate = (total_tested - no_weld_count) / total_tested * 100 if total_tested > 0 else 0
    
    print(f"\n📊 检测总结:")
    print(f"  总测试图片: {total_tested}")
    print(f"  检测到焊缝: {total_tested - no_weld_count}")
    print(f"  未检测到焊缝: {no_weld_count}")
    print(f"  检测成功率: {detection_rate:.1f}%")
    
    return results

def save_test_report(results, output_dir, use_negative=True):
    """保存测试报告（包含负片信息和未检测统计）"""
    if not results:
        return
    
    # 计算整体统计
    total_images = len(results)
    weld_ratios = [r['weld_ratio'] for r in results]
    confidences = [r['avg_confidence'] for r in results if r['avg_confidence'] > 0]
    
    # 统计检测情况
    detected_count = sum(1 for r in results if r.get('has_weld', True))
    not_detected_count = total_images - detected_count
    detection_rate = detected_count / total_images * 100 if total_images > 0 else 0
    
    report = {
        'test_time': datetime.now().isoformat(),
        'total_images': total_images,
        'detected_images': detected_count,
        'not_detected_images': not_detected_count,
        'detection_rate': detection_rate,
        'negative_enabled': use_negative,
        'negative_params': {
            'formula': '255 - input',
            'apply_probability': 1.0
        } if use_negative else None,
        'statistics': {
            'weld_ratio': {
                'mean': float(np.mean(weld_ratios)),
                'std': float(np.std(weld_ratios)),
                'min': float(np.min(weld_ratios)),
                'max': float(np.max(weld_ratios))
            },
            'confidence': {
                'mean': float(np.mean(confidences)) if confidences else 0,
                'std': float(np.std(confidences)) if confidences else 0,
                'min': float(np.min(confidences)) if confidences else 0,
                'max': float(np.max(confidences)) if confidences else 0
            }
        },
        'details': results
    }
    
    # 保存JSON报告
    report_path = output_dir / 'test_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # 保存文本报告
    txt_report_path = output_dir / 'test_report.txt'
    with open(txt_report_path, 'w', encoding='utf-8') as f:
        f.write("焊缝检测测试报告\n")
        f.write("=" * 50 + "\n")
        f.write(f"测试时间: {report['test_time']}\n")
        f.write(f"测试图片数: {total_images}\n")
        f.write(f"检测到焊缝: {detected_count}\n")
        f.write(f"未检测到焊缝: {not_detected_count}\n")
        f.write(f"检测成功率: {detection_rate:.1f}%\n")
        f.write(f"负片预处理: {'启用' if use_negative else '禁用'}\n")
        if use_negative:
            f.write(f"负片公式: output = 255 - input\n")
        f.write("\n")
        
        f.write("焊缝占比统计:\n")
        f.write(f"  平均值: {report['statistics']['weld_ratio']['mean']:.2f}%\n")
        f.write(f"  标准差: {report['statistics']['weld_ratio']['std']:.2f}%\n")
        f.write(f"  最小值: {report['statistics']['weld_ratio']['min']:.2f}%\n")
        f.write(f"  最大值: {report['statistics']['weld_ratio']['max']:.2f}%\n\n")
        
        f.write("预测置信度统计:\n")
        f.write(f"  平均值: {report['statistics']['confidence']['mean']:.3f}\n")
        f.write(f"  标准差: {report['statistics']['confidence']['std']:.3f}\n")
        f.write(f"  最小值: {report['statistics']['confidence']['min']:.3f}\n")
        f.write(f"  最大值: {report['statistics']['confidence']['max']:.3f}\n\n")
        
        f.write("详细结果:\n")
        for result in results:
            weld_status = "✓" if result.get('has_weld', True) else "✗"
            f.write(f"{weld_status} {result['image_name']}: ")
            f.write(f"焊缝{result['weld_ratio']:.2f}%, ")
            f.write(f"置信度{result['avg_confidence']:.3f}\n")
    
    print(f"\n📊 测试报告已保存:")
    print(f"  详细报告: {report_path}")
    print(f"  文本报告: {txt_report_path}")

def quick_filter_false_welds(pred_mask, confidence_map=None, image_height=None):
    """
    强化版假焊缝过滤 - 专门针对上方文字/标记误识别问题
    """
    if image_height is None:
        image_height = pred_mask.shape[0]
    
    image_width = pred_mask.shape[1]
    filtered_mask = np.zeros_like(pred_mask)
    
    # 找到所有连通区域
    weld_mask = (pred_mask == 1).astype(np.uint8)
    contours, _ = cv2.findContours(weld_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"发现 {len(contours)} 个焊缝区域，开始强化过滤...")
    
    valid_contours = []
    
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        
        # 基本面积过滤
        if area < 500:
            print(f"  区域{i+1}: 面积太小({area:.0f}) - 过滤")
            continue
        
        # 计算区域几何特征
        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue
            
        cx = int(M["m10"] / M["m00"])  # 质心x坐标
        cy = int(M["m01"] / M["m00"])  # 质心y坐标
        
        # 计算最小外接矩形
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        w, h = rect[1]
        
        if w == 0 or h == 0:
            continue
            
        aspect_ratio = max(w, h) / min(w, h)
        
        # 计算区域位置（分为上中下三部分）
        position = "下部" if cy > 2 * image_height / 3 else ("中部" if cy > image_height / 3 else "上部")
        
        print(f"  区域{i+1}: {position} (x={cx}, y={cy}, 面积={area:.0f}, 长宽比={aspect_ratio:.2f})")
        
        # 上部区域（最严格）- 这里是文字/标记最容易出现的地方
        if cy < image_height / 3:
            print(f"    位于上部区域，应用严格过滤...")
            
            # 1. 面积过滤：上部小面积直接过滤
            if area < 3000:  # 提高上部面积阈值
                print(f"    上部面积过小({area:.0f} < 3000) - 过滤")
                continue
            
            # 2. 长宽比过滤：上部区域必须是明显的长条形
            if aspect_ratio < 3.0:  # 提高长宽比要求
                print(f"    上部长宽比不足({aspect_ratio:.2f} < 3.0) - 过滤")
                continue
            
            # 3. 置信度过滤：上部需要非常高的置信度
            if confidence_map is not None:
                mask_region = np.zeros_like(pred_mask)
                cv2.fillPoly(mask_region, [contour], 1)
                avg_confidence = np.mean(confidence_map[mask_region == 1])
                
                if avg_confidence < 0.85:  # 提高置信度阈值
                    print(f"    上部置信度不足({avg_confidence:.3f} < 0.85) - 过滤")
                    continue
            
            # 4. 形状特征过滤：检查是否像文字
            # 计算轮廓的凸包
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # 文字通常不够"实心"
            if solidity < 0.7:
                print(f"    上部实心度不足({solidity:.3f} < 0.7) - 疑似文字，过滤")
                continue
            
            # 5. 位置合理性检查：真正的焊缝通常在管道连接处
            # 检查是否在图像边缘附近（文字通常在中间）
            edge_distance = min(cx, image_width - cx, cy, image_height - cy)
            relative_edge_distance = edge_distance / min(image_width, image_height)
            
            if relative_edge_distance > 0.3:  # 距离边缘太远
                print(f"    距离边缘太远({relative_edge_distance:.3f}) - 疑似文字，过滤")
                continue
                
        # 中部区域（中等严格）
        elif cy < 2 * image_height / 3:
            print(f"    位于中部区域，应用中等过滤...")
            
            if area < 1500:
                print(f"    中部面积过小({area:.0f} < 1500) - 过滤")
                continue
                
            # 放宽中部区域的长宽比要求
            if aspect_ratio < 1.8:  # 从2.0降低到1.8
                print(f"    中部长宽比不足({aspect_ratio:.2f} < 1.8) - 过滤")
                continue
                
            if confidence_map is not None:
                mask_region = np.zeros_like(pred_mask)
                cv2.fillPoly(mask_region, [contour], 1)
                avg_confidence = np.mean(confidence_map[mask_region == 1])
                
                if avg_confidence < 0.6:  # 从0.65降低到0.6
                    print(f"    中部置信度不足({avg_confidence:.3f} < 0.6) - 过滤")
                    continue
        
        # 下部区域（相对宽松）- 真正的焊缝多在这里
        else:
            print(f"    位于下部区域，应用宽松过滤...")
            
            if area < 800:
                print(f"    下部面积过小({area:.0f} < 800) - 过滤")
                continue
                
            if aspect_ratio < 1.5:
                print(f"    下部长宽比不足({aspect_ratio:.2f} < 1.5) - 过滤")
                continue
        
        # 通过所有条件的区域
        print(f"    ✅ 区域{i+1}通过过滤")
        valid_contours.append(contour)
    
    # 绘制有效区域
    for contour in valid_contours:
        cv2.fillPoly(filtered_mask, [contour], 1)
    
    print(f"过滤结果: {len(contours)} -> {len(valid_contours)} 个区域")
    return filtered_mask

def visualize_filter_comparison(image_path, original_mask, filtered_mask, confidence, enhanced_image):
    """可视化过滤前后对比"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 计算统计信息
    orig_weld_pixels = np.sum(original_mask == 1)
    filt_weld_pixels = np.sum(filtered_mask == 1)
    total_pixels = original_mask.shape[0] * original_mask.shape[1]
    reduction_ratio = (orig_weld_pixels - filt_weld_pixels) / orig_weld_pixels if orig_weld_pixels > 0 else 0
    
    # 创建叠加图
    def create_overlay(img, mask, color=[255, 0, 0]):
        overlay = img.copy()
        weld_mask = (mask == 1)
        if np.any(weld_mask):
            overlay[weld_mask] = overlay[weld_mask] * 0.6 + np.array(color) * 0.4
            # 绘制边界
            contours, _ = cv2.findContours(weld_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            overlay_bgr = cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.drawContours(overlay_bgr, contours, -1, (0, 255, 0), 2)
            overlay = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
        return overlay
    
    original_overlay = create_overlay(enhanced_image, original_mask, [255, 0, 0])  # 红色
    filtered_overlay = create_overlay(enhanced_image, filtered_mask, [0, 255, 0])  # 绿色
    
    # 第一行
    axes[0, 0].imshow(enhanced_image)
    axes[0, 0].set_title('负片增强图', fontsize=12)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(original_mask, cmap='gray')
    axes[0, 1].set_title(f'原始预测\n焊缝占比: {orig_weld_pixels/total_pixels*100:.2f}%', fontsize=12)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(filtered_mask, cmap='gray')
    axes[0, 2].set_title(f'过滤后结果\n焊缝占比: {filt_weld_pixels/total_pixels*100:.2f}%', fontsize=12)
    axes[0, 2].axis('off')
    
    # 第二行
    axes[1, 0].imshow(original_overlay)
    axes[1, 0].set_title('原始预测叠加（红色）', fontsize=12)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(filtered_overlay)
    axes[1, 1].set_title('过滤后叠加（绿色）', fontsize=12)
    axes[1, 1].axis('off')
    
    # 置信度图
    im = axes[1, 2].imshow(confidence, cmap='jet', vmin=0, vmax=1)
    axes[1, 2].set_title('置信度图', fontsize=12)
    axes[1, 2].axis('off')
    plt.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04)
    
    # 总标题
    fig.suptitle(f'{Path(image_path).name} - 假焊缝过滤对比\n减少了 {reduction_ratio*100:.1f}% 的假阳性', fontsize=14)
    plt.tight_layout()
    
    # 保存对比图
    output_path = Path('filter_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"过滤对比图已保存: {output_path}")

def main():
    """主函数"""
    print("焊缝检测模型测试工具 - 负片预处理版本")
    print("=" * 60)
    
    # 检查模型文件
    model_paths = [
        'best_weld_model_negative_3080.pth',   # 负片版本优先
        # 'best_weld_model_negative.pth',
        # 'best_model_negative.pth',
        # 'best_weld_model_v2.pth',              # 备选
        # 'best_weld_model.pth'                  # 默认名称
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        print(f"❌ 找不到模型文件，请检查以下路径是否存在:")
        for path in model_paths:
            print(f"  - {path}")
        return
    
    # 询问是否启用负片预处理
    print(f"\n找到模型文件: {model_path}")
    print("是否启用负片预处理？")
    print("  建议：如果模型使用负片训练，测试时也应启用负片预处理")
    use_negative_input = input("启用负片预处理？(Y/n): ").strip().lower()
    use_negative = use_negative_input != 'n'
    
    # 设置最小焊缝像素阈值
    min_weld_pixels_input = input("设置最小焊缝像素阈值 (默认500): ").strip()
    min_weld_pixels = int(min_weld_pixels_input) if min_weld_pixels_input.isdigit() else 500
    print(f"最小焊缝像素阈值: {min_weld_pixels}")
    
    # 创建模型并加载权重
    # 根据模型文件名自动选择编码器
    encoder_name =  'resnet50'
    model = WeldSegmentationModel(encoder_name=encoder_name, use_negative=use_negative)
    if not model.load_model(model_path):
        return
    
    # 选择测试方式
    print("\n选择测试方式:")
    print("1. 测试单张图片")
    print("2. 测试test目录所有图片")
    print("3. 测试test目录前10张图片（快速预览）")
    print("4. 测试指定目录")
    print("5. 负片效果对比测试")
    print("6. 假焊缝过滤效果测试（推荐）")
    print("7. 单张图片过滤测试（显示详细过程）")
    print("8. 批量检测未检测到焊缝的图片（新功能）")
    
    choice = input("请输入选择 (1-8): ").strip()
    
    if choice == '1':
        # 测试单张图片
        img_path = input("请输入图片路径: ").strip()
        # 处理路径中的引号
        img_path = img_path.strip("'\"")
        
        if not os.path.exists(img_path):
            print("图片文件不存在")
            return
        
        output_dir = Path('single_test_result_negative')
        output_dir.mkdir(exist_ok=True)
        
        transform = get_test_transform()
        pred_result = model.predict_single_image(img_path, transform)
        
        if pred_result[0] is not None:
            pred_mask, confidence, enhanced_image = pred_result
            has_weld = check_weld_detection(pred_mask, min_weld_pixels)
            
            stats = visualize_prediction(
                img_path, pred_mask, confidence, enhanced_image, 
                output_dir, use_negative
            )
            print(f"\n预测完成:")
            print(f"  焊缝检测: {'✅ 检测到' if has_weld else '❌ 未检测到'}")
            print(f"  焊缝占比: {stats['weld_ratio']:.2f}%")
            print(f"  平均置信度: {stats['avg_confidence']:.3f}")
            print(f"  负片预处理: {'启用' if use_negative else '禁用'}")
            print(f"  结果保存在: {output_dir}")
            
            if not has_weld:
                no_weld_file = save_no_weld_images([Path(img_path).name], output_dir)
    
    elif choice == '2':
        # 测试test目录所有图片
        test_dir = '标注/test'
        output_dir = f'test_results_full{"_negative" if use_negative else ""}'
        test_on_images(model, test_dir, output_dir, min_weld_pixels=min_weld_pixels)
        
    elif choice == '3':
        # 测试前10张
        test_dir = '标注/test'
        output_dir = f'test_results_preview{"_negative" if use_negative else ""}'
        test_on_images(model, test_dir, output_dir, max_images=10, min_weld_pixels=min_weld_pixels)
        
    elif choice == '4':
        # 测试指定目录
        test_dir = input("请输入图片目录路径: ").strip()
        if not os.path.exists(test_dir):
            print("目录不存在:", test_dir)
            return
        output_dir = f'test_results_custom{"_negative" if use_negative else ""}'
        max_imgs = input("最大图片数量（回车=全部）: ").strip()
        max_imgs = int(max_imgs) if max_imgs.isdigit() else None
        test_on_images(model, test_dir, output_dir, max_images=max_imgs, min_weld_pixels=min_weld_pixels)
    
    elif choice == '5':
        # 负片效果对比测试
        print("执行负片效果对比测试...")
        test_dir = input("请输入图片目录路径（回车使用标注/test）: ").strip()
        if not test_dir:
            test_dir = '标注/test'
        
        if not os.path.exists(test_dir):
            print("目录不存在")
            return
        
        # 分别用启用和禁用负片的模型测试
        output_dir = Path('negative_comparison_test')
        output_dir.mkdir(exist_ok=True)
        
        print("\n1. 使用负片预处理测试...")
        model_negative = WeldSegmentationModel(encoder_name=encoder_name, use_negative=True)
        model_negative.load_model(model_path)
        results_negative = test_on_images(model_negative, test_dir, output_dir / 'with_negative', 
                                          max_images=5, min_weld_pixels=min_weld_pixels)
        
        print("\n2. 不使用负片预处理测试...")
        model_no_negative = WeldSegmentationModel(encoder_name=encoder_name, use_negative=False)
        model_no_negative.load_model(model_path)
        results_no_negative = test_on_images(model_no_negative, test_dir, output_dir / 'without_negative', 
                                            max_images=5, min_weld_pixels=min_weld_pixels)
        
        # 对比分析
        if results_negative and results_no_negative:
            negative_avg_ratio = np.mean([r['weld_ratio'] for r in results_negative])
            no_negative_avg_ratio = np.mean([r['weld_ratio'] for r in results_no_negative])
            negative_avg_conf = np.mean([r['avg_confidence'] for r in results_negative if r['avg_confidence'] > 0])
            no_negative_avg_conf = np.mean([r['avg_confidence'] for r in results_no_negative if r['avg_confidence'] > 0])
            
            # 统计检测率
            negative_detection_rate = sum(1 for r in results_negative if r.get('has_weld', True)) / len(results_negative) * 100
            no_negative_detection_rate = sum(1 for r in results_no_negative if r.get('has_weld', True)) / len(results_no_negative) * 100
            
            comparison_report = output_dir / 'negative_comparison_report.txt'
            with open(comparison_report, 'w', encoding='utf-8') as f:
                f.write("负片效果对比报告\n")
                f.write("=" * 40 + "\n")
                f.write(f"使用负片 - 平均焊缝占比: {negative_avg_ratio:.2f}%\n")
                f.write(f"不用负片 - 平均焊缝占比: {no_negative_avg_ratio:.2f}%\n")
                f.write(f"焊缝占比差异: {negative_avg_ratio - no_negative_avg_ratio:.2f}%\n\n")
                f.write(f"使用负片 - 平均置信度: {negative_avg_conf:.3f}\n")
                f.write(f"不用负片 - 平均置信度: {no_negative_avg_conf:.3f}\n")
                f.write(f"置信度差异: {negative_avg_conf - no_negative_avg_conf:.3f}\n\n")
                f.write(f"使用负片 - 检测成功率: {negative_detection_rate:.1f}%\n")
                f.write(f"不用负片 - 检测成功率: {no_negative_detection_rate:.1f}%\n")
                f.write(f"检测率差异: {negative_detection_rate - no_negative_detection_rate:.1f}%\n")
            
            print(f"\n📊 对比测试完成！报告保存在: {comparison_report}")
    
    elif choice == '6':
        # 假焊缝过滤效果测试
        print("执行假焊缝过滤效果测试...")
        test_dir = input("请输入图片目录路径（回车使用标注/test）: ").strip()
        if not test_dir:
            test_dir = '标注/test'
        
        if not os.path.exists(test_dir):
            print("目录不存在")
            return
        
        # 获取图片列表
        image_extensions = ['.bmp', '.jpg', '.jpeg', '.png', '.tif']
        all_images = []
        for ext in image_extensions:
            all_images.extend(list(Path(test_dir).glob(f'*{ext}')))
        
        if not all_images:
            print("目录中没有找到图片")
            return
        
        max_test = min(5, len(all_images))  # 最多测试5张
        print(f"将测试前 {max_test} 张图片的过滤效果...")
        
        transform = get_test_transform()
        output_dir = Path('filter_test_results_negative')
        output_dir.mkdir(exist_ok=True)
        
        total_reduction = 0
        
        for i, img_path in enumerate(all_images[:max_test]):
            print(f"\n{'='*50}")
            print(f"测试图片 {i+1}/{max_test}: {img_path.name}")
            print(f"{'='*50}")
            
            # 原始预测
            pred_result = model.predict_single_image(img_path, transform)
            if pred_result[0] is None:
                print("跳过（无法读取）")
                continue
            
            original_mask, confidence, enhanced_image = pred_result
            
            # 应用过滤
            filtered_mask = quick_filter_false_welds(original_mask, confidence, original_mask.shape[0])
            
            # 计算统计
            orig_pixels = np.sum(original_mask == 1)
            filt_pixels = np.sum(filtered_mask == 1)
            reduction = (orig_pixels - filt_pixels) / orig_pixels if orig_pixels > 0 else 0
            total_reduction += reduction
            
            print(f"\n📊 统计结果:")
            print(f"  原始焊缝像素: {orig_pixels:,}")
            print(f"  过滤后像素: {filt_pixels:,}")
            print(f"  减少比例: {reduction*100:.1f}%")
            
            # 保存对比结果
            visualize_filter_comparison(img_path, original_mask, filtered_mask, confidence, enhanced_image)
            
            # 将对比图移动到结果目录
            if Path('filter_comparison.png').exists():
                new_path = output_dir / f'filter_comparison_{img_path.stem}.png'
                Path('filter_comparison.png').rename(new_path)
                print(f"  对比图保存: {new_path}")
        
        avg_reduction = total_reduction / max_test if max_test > 0 else 0
        print(f"\n🎯 总体结果:")
        print(f"  平均减少假阳性: {avg_reduction*100:.1f}%")
        print(f"  结果保存在: {output_dir}")
    
    elif choice == '7':
        # 单张图片详细过滤测试
        print("单张图片详细过滤测试...")
        img_path = input("请输入图片路径: ").strip()
        
        # 处理路径中的引号
        img_path = img_path.strip("'\"")
        
        if not os.path.exists(img_path):
            print(f"图片文件不存在: {img_path}")
            print("请检查路径是否正确")
            return
        
        print(f"\n{'='*60}")
        print(f"详细测试图片: {Path(img_path).name}")
        print(f"{'='*60}")
        
        transform = get_test_transform()
        
        # 原始预测
        print("1. 获取原始预测结果...")
        pred_result = model.predict_single_image(img_path, transform)
        if pred_result[0] is None:
            print("❌ 无法读取图片")
            return
        
        original_mask, confidence, enhanced_image = pred_result
        orig_pixels = np.sum(original_mask == 1)
        print(f"   原始焊缝像素: {orig_pixels:,}")
        
        # 应用过滤（显示详细过程）
        print("\n2. 应用假焊缝过滤...")
        filtered_mask = quick_filter_false_welds(original_mask, confidence, original_mask.shape[0])
        filt_pixels = np.sum(filtered_mask == 1)
        
        # 计算效果
        reduction = (orig_pixels - filt_pixels) / orig_pixels if orig_pixels > 0 else 0
        print(f"\n3. 过滤效果:")
        print(f"   过滤后像素: {filt_pixels:,}")
        print(f"   减少像素: {orig_pixels - filt_pixels:,}")
        print(f"   减少比例: {reduction*100:.1f}%")
        
        # 生成对比图
        print(f"\n4. 生成对比图...")
        output_dir = Path('single_filter_test_negative')
        output_dir.mkdir(exist_ok=True)
        
        visualize_filter_comparison(img_path, original_mask, filtered_mask, confidence, enhanced_image)
        
        # 移动结果文件
        if Path('filter_comparison.png').exists():
            result_path = output_dir / f'filter_result_{Path(img_path).stem}.png'
            Path('filter_comparison.png').rename(result_path)
            print(f"   对比图保存: {result_path}")
        
        print(f"\n✅ 单张图片测试完成！")
        if reduction > 0.1:
            print(f"🎉 成功过滤了 {reduction*100:.1f}% 的假阳性！")
        elif reduction == 0:
            print("ℹ️  未发现需要过滤的假焊缝")
        else:
            print("⚠️  可能需要调整过滤参数")
    
    elif choice == '8':
        # 批量检测未检测到焊缝的图片（新功能）
        print("批量检测未检测到焊缝的图片...")
        test_dir = input("请输入图片目录路径（回车使用标注/test）: ").strip()
        if not test_dir:
            test_dir = '标注/test'
        
        if not os.path.exists(test_dir):
            print("目录不存在:", test_dir)
            return
        
        output_dir = f'no_weld_detection_results{"_negative" if use_negative else ""}'
        max_imgs = input("最大检测图片数量（回车=全部）: ").strip()
        max_imgs = int(max_imgs) if max_imgs.isdigit() else None
        
        print(f"\n🔍 开始批量检测...")
        print(f"最小焊缝像素阈值: {min_weld_pixels}")
        
        results = test_on_images(model, test_dir, output_dir, max_images=max_imgs, min_weld_pixels=min_weld_pixels)
        
        if results:
            # 统计未检测图片
            no_weld_images = [r['image_name'] for r in results if not r.get('has_weld', True)]
            total_tested = len(results)
            no_weld_count = len(no_weld_images)
            detection_rate = (total_tested - no_weld_count) / total_tested * 100 if total_tested > 0 else 0
            
            print(f"\n📈 批量检测完成:")
            print(f"  总测试图片: {total_tested}")
            print(f"  检测到焊缝: {total_tested - no_weld_count}")
            print(f"  未检测到焊缝: {no_weld_count}")
            print(f"  检测成功率: {detection_rate:.1f}%")
            
            if no_weld_images:
                print(f"\n📝 未检测到焊缝的图片:")
                for i, img_name in enumerate(no_weld_images[:10], 1):  # 只显示前10个
                    print(f"  {i:2d}. {img_name}")
                if len(no_weld_images) > 10:
                    print(f"  ... 还有 {len(no_weld_images) - 10} 张图片")
                
                print(f"\n完整列表已保存到文本文件中。")
            else:
                print(f"\n🎉 所有图片都成功检测到焊缝！")
        
    else:
        print("无效选择")
        return
    
    print("\n✅ 测试完成！")
    if use_negative:
        print("🔄 负片预处理: output = 255 - input")
    print(f"🏗️ 模型架构: U-Net + {encoder_name}")
    print(f"🎯 最小焊缝像素阈值: {min_weld_pixels}")

if __name__ == "__main__":
    main()