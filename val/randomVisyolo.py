import os
import random
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
# 设置matplotlib支持中文
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False
import yaml


def load_dataset_config(data_dir):
    """
    Load dataset configuration from dataset.yaml file.
    Returns the class names list.
    """
    yaml_path = os.path.join(data_dir, 'dataset.yaml')

    if not os.path.exists(yaml_path):
        print(f"Warning: dataset.yaml not found at {yaml_path}")
        return None

    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        if 'names' in config:
            return config['names']
        else:
            print("Warning: 'names' field not found in dataset.yaml")
            return None
    except Exception as e:
        print(f"Error loading dataset.yaml: {str(e)}")
        return None


def denormalize_coordinates(norm_coords, img_width, img_height):
    """
    Denormalizes the coordinates to the image scale.
    norm_coords: List of normalized coordinates [x1, y1, x2, y2, ..., xn, yn]
    img_width: Image width
    img_height: Image height
    """
    coords = []
    for i in range(0, len(norm_coords), 2):
        x = int(norm_coords[i] * img_width)
        y = int(norm_coords[i + 1] * img_height)
        coords.append((x, y))
    return coords


def denormalize_bbox(xc, yc, w_norm, h_norm, img_width, img_height):
    """
    把 YOLO 归一化的 [x_center, y_center, w, h] 转换为像素 [x, y, w, h]
    (x, y) 是左上角坐标
    """
    w = w_norm * img_width
    h = h_norm * img_height
    x = (xc * img_width) - w / 2
    y = (yc * img_height) - h / 2
    return x, y, w, h


def get_random_colors(num_classes):
    """
    Generate random colors for each class.
    """
    np.random.seed(42)  # 固定随机种子以保证颜色一致性
    colors = []
    for i in range(num_classes):
        # 生成较亮的颜色，避免太暗
        color = (
            np.random.uniform(0.3, 1.0),
            np.random.uniform(0.3, 1.0),
            np.random.uniform(0.3, 1.0)
        )
        colors.append(color)
    return colors


def visualize_segmentation(image, label_path, img_width, img_height, ax, class_names=None):
    """
    Visualizes segmentation annotations (polygons) with class labels.
    """
    # Read the labels
    with open(label_path, 'r') as f:
        lines = f.readlines()

    # 如果有类别名称，生成对应的颜色
    if class_names:
        colors = get_random_colors(len(class_names))
    else:
        colors = [(1, 0, 0)]  # 默认红色

    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])  # 获取类别ID
        coords = list(map(float, parts[1:]))  # 剩余的是坐标
        coords = denormalize_coordinates(coords, img_width, img_height)
        coords = np.array(coords)

        # 获取类别名称和颜色
        if class_names and 0 <= class_id < len(class_names):
            class_name = class_names[class_id]
            color = colors[class_id]
        else:
            class_name = f"Class {class_id}"
            color = colors[0] if colors else (1, 0, 0)

        # Plot the polygon (closed shape)
        ax.fill(coords[:, 0], coords[:, 1], color=color, alpha=0.3, edgecolor=color, linewidth=2)

        # 在多边形的中心添加类别标签
        center_x = np.mean(coords[:, 0])
        center_y = np.mean(coords[:, 1])

        # 添加带背景的文本标签
        bbox_props = dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8, edgecolor=color)
        ax.text(center_x, center_y, class_name,
                fontsize=10, fontweight='bold', color='black',
                ha='center', va='center', bbox=bbox_props)


def visualize_detection(image, label_path, img_width, img_height, ax, class_names=None):
    """
    Visualizes YOLO detection annotations (bounding boxes) with class labels.
    每行: class x_center y_center w h (全部归一化到0-1)
    """
    with open(label_path, 'r') as f:
        lines = f.readlines()

    # 如果有类别名称，生成对应的颜色
    if class_names:
        colors = get_random_colors(len(class_names))
    else:
        colors = [(1, 0, 0)]  # 默认红色

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:  # 应该有类别 + 4个数
            print(f"Warning: Invalid detection format in line: {line.strip()}")
            continue

        # 获取类别ID
        class_id = int(parts[0])
        # 取归一化的中心点和宽高
        xc, yc, w_norm, h_norm = map(float, parts[1:5])

        # 转换成像素坐标
        x, y, w, h = denormalize_bbox(xc, yc, w_norm, h_norm, img_width, img_height)

        # 获取类别名称和颜色
        if class_names and 0 <= class_id < len(class_names):
            class_name = class_names[class_id]
            color = colors[class_id]
        else:
            class_name = f"Class {class_id}"
            color = colors[0] if colors else (1, 0, 0)

        # 画矩形 (左上角(x, y)，宽w，高h)
        rect = plt.Rectangle((x, y), w, h,
                             linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)

        # 在矩形上方添加类别标签
        label_y = max(0, y - 5)  # 确保标签不超出图像边界
        bbox_props = dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8)
        ax.text(x, label_y, class_name,
                fontsize=10, fontweight='bold', color='white',
                va='bottom', bbox=bbox_props)


def visualize_image_and_labels(image_path, label_path, format_type='seg', class_names=None):
    """
    Visualizes the image with its corresponding YOLO label annotations.
    format_type: 'seg' for segmentation, 'det' for detection
    class_names: List of class names from dataset.yaml
    """
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # 使用IMREAD_UNCHANGED以支持各种格式

    # 如果是多通道TIF图像（例如4通道），只使用前3个通道用于显示
    if len(image.shape) == 3 and image.shape[2] > 3:
        image = image[:, :, :3]
    # 如果是单通道图像，转换为3通道用于显示
    elif len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # 处理不同位深的图像
    if image.dtype == np.uint16:
        # 16位图像，需要转换到8位
        # 方法1：线性缩放到0-255范围
        image_min = image.min()
        image_max = image.max()
        if image_max > image_min:
            image = ((image - image_min) / (image_max - image_min) * 255).astype(np.uint8)
        else:
            image = np.zeros_like(image, dtype=np.uint8)
    elif image.dtype == np.float32 or image.dtype == np.float64:
        # 浮点图像，假设范围在0-1之间，转换到0-255
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
    # 如果已经是uint8，则不需要转换

    img_height, img_width = image.shape[:2]

    # Plot the image
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Choose visualization based on format type
    if format_type == 'seg':
        visualize_segmentation(image, label_path, img_width, img_height, ax, class_names)
    elif format_type == 'det':
        visualize_detection(image, label_path, img_width, img_height, ax, class_names)
    else:
        print(f"Unknown format type: {format_type}")
        return

    # Add title
    title = f"{os.path.basename(image_path)} - {format_type.upper()} Format"
    if class_names:
        title += f" ({len(class_names)} classes)"
    ax.set_title(title, fontsize=14)
    ax.axis('off')

    # 添加图例（如果有类别名称）
    if class_names:
        # 创建图例
        from matplotlib.patches import Patch
        colors = get_random_colors(len(class_names))
        legend_elements = [Patch(facecolor=colors[i], alpha=0.5, label=class_names[i])
                           for i in range(len(class_names))]
        ax.legend(handles=legend_elements, loc='upper right',
                  bbox_to_anchor=(1.15, 1), fontsize=9)

    plt.tight_layout()
    plt.show()


def main(data_dir, format_type='seg'):
    """
    Main function to visualize YOLO dataset.
    data_dir: Path to the dataset directory
    format_type: 'seg' for segmentation, 'det' for detection
    """
    print(f"Visualizing dataset in {format_type.upper()} format")
    print(f"Dataset directory: {data_dir}")

    # 加载类别名称
    class_names = load_dataset_config(data_dir)
    if class_names:
        print(f"Loaded {len(class_names)} classes: {', '.join(class_names)}")
    else:
        print("Warning: Could not load class names from dataset.yaml")
        print("Labels will be displayed as 'Class 0', 'Class 1', etc.")

    # Get list of subdirectories (train and val)
    splits = ['train', 'valid', 'val']  # 添加'val'作为备选

    # Check which splits exist
    available_splits = []
    for split in splits:
        images_dir = os.path.join(data_dir, 'images', split)
        if os.path.exists(images_dir):
            available_splits.append(split)

    if not available_splits:
        print(f"No valid splits found in {data_dir}/images/")
        print(f"Expected one of: {', '.join(splits)}")
        return

    # Choose randomly between available splits
    split = random.choice(available_splits)
    print(f"Selected split: {split}")

    # Get list of images and corresponding label files in the selected split
    images_dir = os.path.join(data_dir, 'images', split)
    labels_dir = os.path.join(data_dir, 'labels', split)

    # List all images in the chosen split directory
    # 支持的图像格式扩展名（包括TIF/TIFF）
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    images = [f for f in os.listdir(images_dir) if f.lower().endswith(supported_extensions)]

    if not images:
        print(f"No images found in {images_dir}")
        print(f"Supported formats: {', '.join(supported_extensions)}")
        return

    print(f"Found {len(images)} images in {images_dir}")

    while True:
        # Select a random image
        img_file = random.choice(images)
        img_path = os.path.join(images_dir, img_file)

        # 获取文件扩展名并构造对应的标签文件名
        base_name = os.path.splitext(img_file)[0]
        label_file = base_name + '.txt'
        label_path = os.path.join(labels_dir, label_file)

        if not os.path.exists(label_path):
            print(f"Label file for {img_file} not found. Skipping.")
            continue

        print(f"\nDisplaying: {img_file}")

        # 获取图像信息
        img_info = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img_info is not None:
            print(f"  Image shape: {img_info.shape}")
            print(f"  Data type: {img_info.dtype}")
            print(f"  Value range: [{img_info.min():.1f}, {img_info.max():.1f}]")

        # 读取并显示标签信息
        with open(label_path, 'r') as f:
            labels = f.readlines()
            class_counts = {}
            for label in labels:
                class_id = int(label.strip().split()[0])
                if class_names and 0 <= class_id < len(class_names):
                    class_name = class_names[class_id]
                else:
                    class_name = f"Class {class_id}"
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

            print(f"  Annotations: {len(labels)} objects")
            for class_name, count in class_counts.items():
                print(f"    - {class_name}: {count}")

        try:
            # Visualize image with labels
            visualize_image_and_labels(img_path, label_path, format_type, class_names)
        except Exception as e:
            print(f"Error visualizing {img_file}: {str(e)}")
            continue

        # Wait for the user to press enter to continue to the next image
        user_input = input("\nPress Enter to continue to the next image (or 'q' to quit): ")
        if user_input.lower() == 'q':
            break


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Visualize YOLO dataset annotations with class labels')
    parser.add_argument('--data_dir', type=str,
                        default='/home/lenovo/code/CHT/detect/dataprocess/preprocessed_data2/test/SWRDsize112',
                        help='Path to the dataset directory containing dataset.yaml')
    parser.add_argument('--format', type=str, choices=['seg', 'det'], default='seg',
                        help='Format type: seg (segmentation) or det (detection)')

    args = parser.parse_args()

    # Run main function with specified parameters
    main(args.data_dir, args.format)