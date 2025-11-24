import os
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from pathlib import Path
import base64
from PIL import Image
import io
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
    """负片预处理类"""
    
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

class LabelmeDataProcessor:
    """处理Labelme标注数据"""
    
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
    
    def parse_labelme_json(self, json_path):
        """解析labelme的JSON文件"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 获取图像信息
        img_width = data['imageWidth']
        img_height = data['imageHeight']
        
        # 解析标注信息
        shapes = data['shapes']
        polygons = []
        
        for shape in shapes:
            points = np.array(shape['points'], dtype=np.int32)
            label = shape['label']
            shape_type = shape.get('shape_type', 'polygon')
            polygons.append({
                'points': points,
                'label': label,
                'shape_type': shape_type
            })
        
        return {
            'width': img_width,
            'height': img_height,
            'polygons': polygons
        }
    
    def create_mask_from_polygons(self, polygons, img_width, img_height, class_map={'焊缝': 1}):
        """从多边形或矩形创建分割掩码"""
        mask = np.zeros((img_height, img_width), dtype=np.uint8)
        
        for poly in polygons:
            label = poly['label']
            points = poly['points']
            shape_type = poly.get('shape_type', 'polygon')
            
            if label in class_map:
                class_id = class_map[label]
                
                if shape_type == 'rectangle':
                    # 处理矩形标注
                    x1, y1 = int(points[0][0]), int(points[0][1])
                    x2, y2 = int(points[1][0]), int(points[1][1])
                    cv2.rectangle(mask, (x1, y1), (x2, y2), class_id, -1)
                else:
                    # 处理多边形标注
                    points_int = np.array(points, dtype=np.int32)
                    cv2.fillPoly(mask, [points_int], class_id)
        
        return mask
    
    def process_dataset(self, output_dir):
        """处理整个数据集"""
        output_path = Path(output_dir)
        images_dir = output_path / 'images'
        masks_dir = output_path / 'masks'
        
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)
        
        # 找到所有JSON文件
        json_files = list(self.data_dir.glob('*.json'))
        
        processed_count = 0
        for json_file in json_files:
            # 对应的图片文件
            img_file = json_file.with_suffix('.bmp')
            
            if not img_file.exists():
                img_file = json_file.with_suffix('.png')
                if not img_file.exists():
                    print(f"警告: 找不到对应的图片文件 {img_file}")
                    continue
            
            try:
                # 解析标注
                annotation = self.parse_labelme_json(json_file)
                
                # 读取图片
                image = imread_universal(str(img_file))
                if image is None:
                    print(f"警告: 无法读取图片 {img_file}")
                    continue
                
                # 创建掩码
                mask = self.create_mask_from_polygons(
                    annotation['polygons'],
                    annotation['width'],
                    annotation['height']
                )
                
                # 保存处理后的文件
                base_name = img_file.stem
                cv2.imwrite(str(images_dir / f'{base_name}.jpg'), image)
                cv2.imwrite(str(masks_dir / f'{base_name}.png'), mask)
                
                processed_count += 1
                
            except Exception as e:
                print(f"处理文件 {json_file} 时出错: {e}")
        
        print(f"成功处理 {processed_count} 个文件")
        return processed_count

class WeldDataset(Dataset):
    """焊缝分割数据集"""
    
    def __init__(self, images_dir, masks_dir, transform=None, use_negative=True, negative_prob=1.0):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        self.use_negative = use_negative
        
        # 初始化负片处理器
        if use_negative:
            self.negative_processor = NegativeTransform(apply_probability=negative_prob)
            print(f"✅ 负片预处理已启用 (应用概率={negative_prob})")
        
        # 获取所有图片文件
        self.image_files = list(self.images_dir.glob('*.jpg'))
        
        # 过滤出有对应掩码的图片
        self.valid_files = []
        for img_file in self.image_files:
            mask_file = self.masks_dir / f'{img_file.stem}.png'
            if mask_file.exists():
                self.valid_files.append(img_file.stem)
        
        print(f"数据集包含 {len(self.valid_files)} 个有效样本")
    
    def __len__(self):
        return len(self.valid_files)
    
    def __getitem__(self, idx):
        file_name = self.valid_files[idx]
        
        # 读取图片和掩码
        image_path = self.images_dir / f'{file_name}.jpg'
        mask_path = self.masks_dir / f'{file_name}.png'
        
        image = imread_universal(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = imread_universal(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # 应用负片预处理
        if self.use_negative:
            processed = self.negative_processor(image)
            image = processed['image']
        
        # 应用其他数据增强
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        return image, mask.long()

def get_transforms(phase='train'):
    """获取数据增强变换 - RTX 3080优化版"""
    if phase == 'train':
        return A.Compose([
            A.Resize(512, 512),  # 3080降低到512x512以节省显存
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.3
            ),
            A.GaussNoise(var_limit=(10.0, 50.0), mean=0, p=0.2),
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(-0.0625, 0.0625),
                rotate=(-15, 15),
                p=0.2
            ),
            A.GridDistortion(p=0.1),
            A.ElasticTransform(p=0.1),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(512, 512),  # 验证时也使用512x512
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

def add_memory_efficient_training():
    """添加显存效率训练技巧"""
    print("\n💡 RTX 3080显存优化技巧:")
    print("1. 混合精度训练 (AMP) - 节省50%显存:")
    print("   from torch.cuda.amp import autocast, GradScaler")
    print("   scaler = GradScaler()")
    print("   with autocast(): outputs = model(images)")
    print()
    print("2. 梯度累积 - 模拟大batch效果:")
    print("   accumulation_steps = 4  # 模拟batch_size=16")
    print("   if (batch_idx + 1) % accumulation_steps == 0:")
    print("       optimizer.step(); optimizer.zero_grad()")
    print()
    print("3. 检查点重计算 - 时间换显存:")
    print("   model = torch.utils.checkpoint.checkpoint_sequential(model)")
    print()
    print("4. 定期清理显存:")
    print("   torch.cuda.empty_cache()")
    print("   del intermediate_tensors")

def get_gpu_memory_info():
    """获取GPU显存使用信息"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        
        print(f"\n🔍 GPU显存使用情况:")
        print(f"  - 已分配: {allocated:.2f} GB")
        print(f"  - 已保留: {reserved:.2f} GB") 
        print(f"  - 峰值分配: {max_allocated:.2f} GB")
        print(f"  - 剩余可用: {10 - reserved:.2f} GB")
        
        if reserved > 8.5:
            print("⚠️  显存使用接近上限，建议:")
            print("  - 减小batch_size")
            print("  - 降低分辨率") 
            print("  - 使用混合精度训练")
    
    return allocated, reserved, max_allocated

class WeldSegmentationModel:
    """焊缝分割模型 - RTX 3080优化版"""
    
    def __init__(self, num_classes=2, encoder_name='resnet101'):  # 3080使用更轻量的ResNet50
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建模型 - 使用U-Net with ResNet50 encoder (3080适配)
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights='imagenet',
            in_channels=3,
            classes=num_classes,
        ).to(self.device)
        
        # 损失函数和优化器 - 3080适配的学习率
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5
        )
        
        print(f"模型已创建，使用设备: {self.device}")
        
        # 打印模型参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"模型参数量: {total_params:,}")
        print(f"显存优化: ResNet50 encoder + 512x512 resolution")
    
    def dice_coefficient(self, pred, target, smooth=1e-6):
        """计算Dice系数"""
        pred = torch.softmax(pred, dim=1)[:, 1]  # 获取焊缝类别的概率
        target = (target == 1).float()  # 转换为二进制
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        return dice
    
    def iou_coefficient(self, pred, target, smooth=1e-6):
        """计算IoU系数"""
        pred = torch.softmax(pred, dim=1)[:, 1]  # 获取焊缝类别的概率
        pred = (pred > 0.5).float()  # 二值化
        target = (target == 1).float()  # 转换为二进制
        
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        iou = (intersection + smooth) / (union + smooth)
        return iou
    
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_dice = 0
        total_iou = 0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(self.device), masks.to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            dice = self.dice_coefficient(outputs, masks)
            iou = self.iou_coefficient(outputs, masks)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_dice += dice.item()
            total_iou += iou.item()
            
            if batch_idx % 10 == 0:  # 3080减少输出频率以节省时间
                print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, Dice: {dice.item():.4f}, IoU: {iou.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        avg_dice = total_dice / len(train_loader)
        avg_iou = total_iou / len(train_loader)
        return avg_loss, avg_dice, avg_iou
    
    def validate(self, val_loader):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        total_dice = 0
        total_iou = 0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(self.device), masks.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                dice = self.dice_coefficient(outputs, masks)
                iou = self.iou_coefficient(outputs, masks)
                
                total_loss += loss.item()
                total_dice += dice.item()
                total_iou += iou.item()
        
        avg_loss = total_loss / len(val_loader)
        avg_dice = total_dice / len(val_loader)
        avg_iou = total_iou / len(val_loader)
        return avg_loss, avg_dice, avg_iou
    
    def train(self, train_loader, val_loader, epochs=250, save_path='best_model.pth'):
        """训练模型 - 针对小数据集优化"""
        best_dice = 0
        patience_counter = 0  # 早停计数器
        patience_limit = 30   # 连续30个epoch无改善则停止
        
        train_losses, train_dices, train_ious = [], [], []
        val_losses, val_dices, val_ious = [], [], []
        
        print(f"🎯 开始训练 - 使用负片预处理 (RTX 3080优化):")
        print(f"  - 总epochs: {epochs}")
        print(f"  - 早停patience: {patience_limit}")
        print(f"  - 当前学习率: {self.optimizer.param_groups[0]['lr']}")
        print(f"  - 预处理: 负片变换 (255 - input)")
        print(f"  - 显存优化: ResNet50 + 512x512 + 小batch")
        
        for epoch in range(epochs):
            print(f'\nEpoch {epoch+1}/{epochs}')
            print('-' * 30)
            
            # 训练
            train_loss, train_dice, train_iou = self.train_epoch(train_loader)
            
            # 验证
            val_loss, val_dice, val_iou = self.validate(val_loader)
            
            # 记录指标
            train_losses.append(train_loss)
            train_dices.append(train_dice)
            train_ious.append(train_iou)
            val_losses.append(val_loss)
            val_dices.append(val_dice)
            val_ious.append(val_iou)
            
            print(f'Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}, IoU: {train_iou:.4f}')
            print(f'Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}')
            
            # 学习率调度
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f'Learning Rate: {current_lr:.6f}')
            
            # 保存最佳模型和早停检查
            if val_dice > best_dice:
                best_dice = val_dice
                patience_counter = 0  # 重置计数器
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_dice': best_dice,
                    'best_iou': val_iou,
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'train_dices': train_dices,
                    'val_dices': val_dices,
                }, save_path)
                print(f'🎉 保存最佳模型，Dice: {best_dice:.4f}, IoU: {val_iou:.4f}')
            else:
                patience_counter += 1
                print(f'⏳ 验证Dice未改善: {patience_counter}/{patience_limit}')
            
            # 早停检查
            if patience_counter >= patience_limit:
                print(f'\n🛑 早停触发：连续{patience_limit}个epoch验证Dice无改善')
                print(f'最佳Dice: {best_dice:.4f} (Epoch {epoch-patience_limit+1})')
                break
            
            # 每50个epoch输出训练进度总结
            if (epoch + 1) % 50 == 0:
                print(f'\n📊 训练进度总结 (Epoch {epoch+1}):')
                print(f'  - 最佳验证Dice: {best_dice:.4f}')
                print(f'  - 当前训练Dice: {train_dice:.4f}')
                print(f'  - 过拟合检查: {"正常" if val_dice >= train_dice*0.85 else "可能过拟合"}')
        
        print(f'\n✅ 训练结束! 最佳验证Dice: {best_dice:.4f}')
        return train_losses, train_dices, val_losses, val_dices
    
    def predict(self, image, use_negative=True):
        """预测单张图片"""
        self.model.eval()
        
        # 如果输入是原始图像，先应用负片变换
        if use_negative and len(image.shape) == 3 and not isinstance(image, torch.Tensor):
            negative_processor = NegativeTransform(apply_probability=1.0)
            processed = negative_processor(image)
            image = processed['image']
        
        with torch.no_grad():
            if len(image.shape) == 3:
                image = image.unsqueeze(0)  # 添加batch维度
            
            image = image.to(self.device)
            output = self.model(image)
            pred = torch.softmax(output, dim=1)
            pred_mask = torch.argmax(pred, dim=1)
            
            return pred_mask.cpu().numpy()[0]
    
    def load_model(self, model_path):
        """加载训练好的模型"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"模型已加载，最佳Dice: {checkpoint['best_dice']:.4f}")

def visualize_predictions(model, dataset, num_samples=4):
    """可视化预测结果"""
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    
    for i in range(num_samples):
        # 获取样本
        image, true_mask = dataset[i]
        
        # 预测
        pred_mask = model.predict(image.unsqueeze(0))
        
        # 转换为可视化格式
        image_np = image.permute(1, 2, 0).cpu().numpy()
        image_np = image_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        image_np = np.clip(image_np, 0, 1)
        
        # 显示
        axes[i, 0].imshow(image_np)
        axes[i, 0].set_title('负片增强图')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(true_mask.cpu().numpy(), cmap='gray')
        axes[i, 1].set_title('真实掩码')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(pred_mask, cmap='gray')
        axes[i, 2].set_title('预测掩码')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.show()

def compare_negative_effect(image_path, save_comparison=True):
    """比较负片前后效果"""
    # 读取原始图像
    original = imread_universal(str(image_path))
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    # 应用负片变换
    negative_processor = NegativeTransform(apply_probability=1.0)
    processed = negative_processor(original_rgb)
    negative_image = processed['image']
    
    if save_comparison:
        # 可视化对比
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        ax1.imshow(original_rgb)
        ax1.set_title('原始图像')
        ax1.axis('off')
        
        ax2.imshow(negative_image)
        ax2.set_title('负片变换后 (255 - input)')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig('negative_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("✅ 负片对比图已保存为 negative_comparison.png")
    
    return original_rgb, negative_image

def create_negative_demo():
    """创建负片变换演示"""
    print("\n🎯 负片变换原理演示:")
    
    # 创建示例图像
    demo_image = np.zeros((200, 400, 3), dtype=np.uint8)
    
    # 添加不同亮度的区域
    demo_image[:, :100] = [50, 50, 50]    # 暗部 (模拟焊缝)
    demo_image[:, 100:200] = [128, 128, 128]  # 中等亮度
    demo_image[:, 200:300] = [200, 200, 200]  # 亮部
    demo_image[:, 300:] = [255, 255, 255]     # 最亮部
    
    # 应用负片变换
    negative_processor = NegativeTransform(apply_probability=1.0)
    processed = negative_processor(demo_image)
    negative_demo = processed['image']
    
    # 显示对比
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.imshow(demo_image)
    ax1.set_title('原始图像\n(左侧模拟暗部焊缝)')
    ax1.axis('off')
    
    ax2.imshow(negative_demo)
    ax2.set_title('负片变换后\n(暗部焊缝变亮，更易检测)')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('negative_principle_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ 负片变换原理演示图已保存为 negative_principle_demo.png")
    print("💡 关键优势:")
    print("  - 暗部焊缝 → 亮部特征，提高可见性")
    print("  - 增强焊缝与背景的对比度")
    print("  - 简单高效，无需复杂参数调优")

def main():
    """主函数"""
    print("焊缝区域精确分割系统 - RTX 3080优化版 + 负片增强")
    print("=" * 60)
    
    # GPU显存优化设置
    if torch.cuda.is_available():
        print(f"检测到GPU: {torch.cuda.get_device_name()}")
        print(f"显存容量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # 设置显存优化选项
        torch.backends.cudnn.benchmark = True  # 优化卷积性能
        torch.cuda.empty_cache()  # 清空显存缓存
        
        # 设置PyTorch显存分配策略
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        print("✅ 显存优化设置已启用")
    
    # 1. 处理训练数据 (train目录 - 有标注)
    print("\n1. 处理训练数据...")
    train_processor = LabelmeDataProcessor('标注/train')
    train_count = train_processor.process_dataset('processed_data/train')
    print(f"训练数据处理完成: {train_count} 个样本")
    
    if train_count == 0:
        print("错误: 训练集为空，请检查数据目录")
        return
    
    # 2. 检查test目录情况
    test_dir = Path('标注/test')
    test_bmps = list(test_dir.glob('*.bmp')) if test_dir.exists() else []
    test_jsons = list(test_dir.glob('*.json')) if test_dir.exists() else []
    
    print(f"\n2. 检查验证数据...")
    print(f"test目录图片数: {len(test_bmps)}")
    print(f"test目录标注数: {len(test_jsons)}")
    
    if len(test_jsons) == 0 and len(test_bmps) > 0:
        print("警告: test目录只有图片没有标注，将从train数据中划分验证集")
        use_train_split = True
    elif len(test_jsons) > 0:
        print("✅ test目录有标注，将作为验证集")
        use_train_split = False
        # 处理test目录的标注数据
        test_processor = LabelmeDataProcessor('标注/test')  
        val_count = test_processor.process_dataset('processed_data/val')
        print(f"验证数据处理完成: {val_count} 个样本")
    else:
        print("从train数据中划分验证集")
        use_train_split = True
    
    # 3. 演示负片效果
    print(f"\n3. 负片预处理演示...")
    train_images = list(Path('processed_data/train/images').glob('*.jpg'))
    if len(train_images) > 0:
        print(f"使用 {train_images[0].name} 展示负片效果")
        compare_negative_effect(train_images[0])
    
    # 4. 创建数据集 - 启用负片预处理
    print(f"\n4. 创建数据集（启用负片预处理）...")
    
    if use_train_split:
        # 从train数据中划分
        print("从训练数据中划分80%训练，20%验证")
        full_dataset = WeldDataset(
            'processed_data/train/images',
            'processed_data/train/masks',
            transform=get_transforms('train'),
            use_negative=True,  # 启用负片变换
            negative_prob=1.0   # 100%应用负片变换
        )
        
        # 划分训练集和验证集
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # 固定随机种子，保证可复现
        )
        
        # 为验证集创建单独的数据集实例
        val_dataset_instance = WeldDataset(
            'processed_data/train/images',
            'processed_data/train/masks',
            transform=get_transforms('val'),
            use_negative=True,  # 验证集也启用负片变换
            negative_prob=1.0   # 验证时也100%应用
        )
        
        # 获取验证集的索引
        val_indices = val_dataset.indices
        val_dataset_instance.valid_files = [full_dataset.valid_files[i] for i in val_indices]
        val_dataset = val_dataset_instance
        
    else:
        # 使用独立的验证集
        print("使用独立的验证集")
        train_dataset = WeldDataset(
            'processed_data/train/images',
            'processed_data/train/masks',
            transform=get_transforms('train'),
            use_negative=True,  # 启用负片变换
            negative_prob=1.0
        )
        
        val_dataset = WeldDataset(
            'processed_data/val/images',
            'processed_data/val/masks',
            transform=get_transforms('val'),
            use_negative=True,  # 启用负片变换
            negative_prob=1.0
        )
    
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")
    
    # RTX 3080 10GB显存优化的batch size和worker设置
    batch_size = 4  # 3080 10GB使用小batch size
    num_workers = 4  # 减少worker数量
    
    print(f"\n🔧 RTX 3080显存优化配置:")
    print(f"  - Batch Size: {batch_size} (显存优化)")
    print(f"  - Workers: {num_workers}")
    print(f"  - 图片分辨率: 512x512")
    print(f"  - 模型: U-Net + ResNet50")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,  # 保持pin_memory以提升性能
        persistent_workers=True  # 保持worker进程，减少重复启动开销
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    
    # 5. 创建和训练模型
    print(f"\n5. 创建并训练模型...")
    print(f"预处理: 负片变换 (255 - input)")
    
    model = WeldSegmentationModel(num_classes=2, encoder_name='resnet50')  # 使用ResNet50
    
    # 训练模型 - 针对230张图像优化的训练策略
    print("🔧 训练策略 (RTX 3080优化):")
    print(f"  - 数据集大小: ~230张图像")
    print(f"  - 推荐epochs: 200-250")
    print(f"  - 早停机制: 验证Dice连续30个epoch不提升时停止")
    print(f"  - 学习率调度: 验证损失10个epoch不下降时减半")
    print(f"  - 负片变换: 100%应用概率，反转明暗对比")
    print(f"  - 显存优化: 小batch + 轻量模型 + 低分辨率")
    
    train_losses, train_dices, val_losses, val_dices = model.train(
        train_loader, val_loader, epochs=500, save_path='best_weld_model_negative_3080.pth'
    )
    
    # 6. 可视化结果
    print("\n6. 可视化预测结果...")
    model.load_model('best_weld_model_negative_3080.pth')
    
    visualize_predictions(model, val_dataset, num_samples=4)  # 3080减少样例数量
    
    print(f"\n训练完成！")
    print(f"最佳模型已保存为: best_weld_model_negative_3080.pth")
    print(f"训练集样本: {len(train_dataset)}")
    print(f"验证集样本: {len(val_dataset)}")
    print(f"负片预处理: 100%应用，公式为 output = 255 - input")
    print(f"RTX 3080优化: ResNet50 + 512px + batch_size=4")
    
    # 7. 如果test目录有未标注图片，提供预测功能
    if len(test_bmps) > 0 and len(test_jsons) == 0:
        print(f"\n7. test目录有 {len(test_bmps)} 张未标注图片")
        print("可以使用训练好的模型进行预测")
        
        # 创建预测函数
        def predict_test_images():
            print("开始预测test目录图片...")
            test_output_dir = Path('test_predictions_negative_3080')
            test_output_dir.mkdir(exist_ok=True)
            
            transform = get_transforms('val')
            negative_processor = NegativeTransform(apply_probability=1.0)
            
            # 批量预测以节省时间
            for i, img_path in enumerate(test_bmps[:10]):  # 预测前10张作为示例
                # 读取图片
                image = imread_universal(str(img_path))
                if image is None:
                    continue
                    
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # 应用负片预处理
                processed = negative_processor(image_rgb)
                image_negative = processed['image']
                
                # 其他预处理
                transformed = transform(image=image_negative)
                input_tensor = transformed['image'].unsqueeze(0)
                
                # 预测
                pred_mask = model.predict(input_tensor, use_negative=False)  # 已经应用了负片变换
                
                # 保存结果
                output_path = test_output_dir / f'pred_{img_path.stem}.png'
                cv2.imwrite(str(output_path), pred_mask * 255)
                
                # 同时保存负片处理后的图片用于对比
                negative_img_path = test_output_dir / f'negative_{img_path.stem}.jpg'
                cv2.imwrite(str(negative_img_path), cv2.cvtColor(image_negative, cv2.COLOR_RGB2BGR))
                
                if i < 3:  # 显示前3个预测结果的信息
                    weld_pixels = np.sum(pred_mask > 0)
                    total_pixels = pred_mask.shape[0] * pred_mask.shape[1]
                    print(f"  {img_path.name}: 焊缝占比 {weld_pixels/total_pixels*100:.2f}%")
                
                # 清理显存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            print(f"预测完成，结果保存在 {test_output_dir}")
            print("包含: 预测掩码 (pred_*.png) 和 负片增强图 (negative_*.jpg)")
        
        # 询问是否进行预测
        print("是否对test图片进行预测？输入 'y' 确认：")
        # predict_test_images()  # 可以取消注释自动预测
        
    print("=" * 60)
    print("📝 RTX 3080优化总结:")
    print("  - 分辨率: 640x640 → 512x512 (节省显存)")
    print("  - 模型: ResNet101 → ResNet50 (减少参数)")
    print("  - Batch Size: 12 → 4 (适配10GB显存)")
    print("  - Workers: 8 → 4 (减少内存占用)")
    print("  - 学习率: 2e-4 → 1e-4 (更稳定)")
    print("  - Patience: 8 → 10 (更宽容的调度)")
    print("  - 负片变换: output = 255 - input")
    print("  - 显存管理: expandable_segments + empty_cache")

if __name__ == "__main__":
    # 显示显存优化技巧
    add_memory_efficient_training()
    
    # 首先演示负片变换原理
    create_negative_demo()
    
    # 运行主程序前检查显存
    if torch.cuda.is_available():
        get_gpu_memory_info()
    
    # 运行主程序
    main()
    
    # 训练完成后再次检查显存使用
    if torch.cuda.is_available():
        print("\n训练完成后的显存状态:")
        get_gpu_memory_info()