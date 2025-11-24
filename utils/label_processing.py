"""
标签处理工具模块
提供YOLO、LabelMe等格式的标签读写和转换功能
"""

import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path


def read_yolo_labels(label_path: str, mode: str = 'det') -> List[List[float]]:
    """
    读取YOLO格式的标签文件

    Args:
        label_path: 标签文件路径
        mode: 'det'(检测) 或 'seg'(分割)

    Returns:
        标签列表
        - det模式: [class_id, x_center, y_center, width, height]
        - seg模式: [class_id, x1, y1, x2, y2, ...]
    """
    labels = []
    if not Path(label_path).exists():
        return labels

    with open(label_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            parts = line.strip().split()
            if not parts:
                continue

            try:
                if mode == 'det':
                    if len(parts) == 5:
                        label = [float(x) for x in parts]
                        if all(0 <= label[i] <= 1 for i in range(1, 5)):
                            labels.append(label)
                elif mode == 'seg':
                    if len(parts) >= 7 and (len(parts) - 1) % 2 == 0:
                        label = [float(x) for x in parts]
                        coords_valid = all(0 <= label[i] <= 1 for i in range(1, len(label)))
                        if coords_valid:
                            labels.append(label)
            except ValueError:
                continue

    return labels


def save_yolo_labels(labels: List[List[float]], output_path: str, mode: str = 'det'):
    """
    保存YOLO格式的标签文件

    Args:
        labels: 标签列表
        output_path: 输出路径
        mode: 'det'(检测) 或 'seg'(分割)
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for label in labels:
            if len(label) == 0:
                continue

            if mode == 'det' and len(label) >= 5:
                class_id = int(label[0])
                coords = ' '.join([f"{x:.6f}" for x in label[1:5]])
                label_str = f"{class_id} {coords}"
            elif mode == 'seg' and len(label) >= 7:
                class_id = int(label[0])
                coords = ' '.join([f"{x:.6f}" for x in label[1:]])
                label_str = f"{class_id} {coords}"
            else:
                continue

            f.write(label_str + '\n')


def read_labelme_json(json_path: str) -> Dict:
    """
    读取LabelMe JSON文件

    Args:
        json_path: JSON文件路径

    Returns:
        标注数据字典
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_labelme_json(data: Dict, json_path: str):
    """
    保存LabelMe JSON文件

    Args:
        data: 标注数据
        json_path: 输出路径
    """
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_bbox(x1: int, y1: int, x2: int, y2: int,
                   img_width: int, img_height: int) -> Tuple[float, float, float, float]:
    """
    将像素坐标边界框转换为归一化坐标（YOLO格式）

    Args:
        x1, y1, x2, y2: 边界框的像素坐标
        img_width, img_height: 图像尺寸

    Returns:
        (x_center, y_center, width, height) 归一化坐标
    """
    x_center = (x1 + x2) / 2.0 / img_width
    y_center = (y1 + y2) / 2.0 / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height

    # 确保值在[0, 1]范围内
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    width = max(0, min(1, width))
    height = max(0, min(1, height))

    return x_center, y_center, width, height


def denormalize_bbox(x_center: float, y_center: float, width: float, height: float,
                     img_width: int, img_height: int) -> Tuple[int, int, int, int]:
    """
    将归一化坐标转换为像素坐标

    Args:
        x_center, y_center, width, height: 归一化坐标
        img_width, img_height: 图像尺寸

    Returns:
        (x1, y1, x2, y2) 像素坐标
    """
    x_center_px = x_center * img_width
    y_center_px = y_center * img_height
    width_px = width * img_width
    height_px = height * img_height

    x1 = int(x_center_px - width_px / 2)
    y1 = int(y_center_px - height_px / 2)
    x2 = int(x_center_px + width_px / 2)
    y2 = int(y_center_px + height_px / 2)

    return x1, y1, x2, y2


def adjust_bbox_for_crop(bbox: List[float], crop_x: int, crop_y: int,
                         crop_w: int, crop_h: int,
                         original_w: int, original_h: int) -> Optional[List[float]]:
    """
    根据裁剪区域调整边界框坐标

    Args:
        bbox: [class_id, x_center, y_center, width, height] 归一化坐标
        crop_x, crop_y: 裁剪区域左上角
        crop_w, crop_h: 裁剪区域尺寸
        original_w, original_h: 原始图像尺寸

    Returns:
        调整后的边界框或None
    """
    class_id = int(bbox[0])

    # 转换为像素坐标
    x1, y1, x2, y2 = denormalize_bbox(bbox[1], bbox[2], bbox[3], bbox[4],
                                      original_w, original_h)

    # 计算与裁剪区域的交集
    intersect_x1 = max(x1, crop_x)
    intersect_y1 = max(y1, crop_y)
    intersect_x2 = min(x2, crop_x + crop_w)
    intersect_y2 = min(y2, crop_y + crop_h)

    # 检查是否有交集
    if intersect_x1 >= intersect_x2 or intersect_y1 >= intersect_y2:
        return None

    # 调整坐标到裁剪图像的坐标系
    new_x1 = max(0, intersect_x1 - crop_x)
    new_y1 = max(0, intersect_y1 - crop_y)
    new_x2 = min(crop_w, intersect_x2 - crop_x)
    new_y2 = min(crop_h, intersect_y2 - crop_y)

    # 转换回归一化坐标
    new_x_center, new_y_center, new_width, new_height = normalize_bbox(
        new_x1, new_y1, new_x2, new_y2, crop_w, crop_h
    )

    return [class_id, new_x_center, new_y_center, new_width, new_height]


def calculate_polygon_area(points: List[float]) -> float:
    """
    计算多边形面积（使用Shoelace公式）

    Args:
        points: 归一化的多边形顶点坐标 [x1, y1, x2, y2, ...]

    Returns:
        归一化的面积
    """
    if len(points) < 6:  # 至少需要3个点
        return 0.0

    area = 0.0
    n = len(points) // 2

    for i in range(n):
        x1 = points[2 * i]
        y1 = points[2 * i + 1]
        x2 = points[2 * ((i + 1) % n)]
        y2 = points[2 * ((i + 1) % n) + 1]
        area += x1 * y2 - x2 * y1

    return abs(area) / 2.0


def clip_polygon_to_window(polygon: List[float],
                           window_bounds: Tuple[float, float, float, float]) -> List[float]:
    """
    使用Sutherland-Hodgman算法裁剪多边形到窗口内

    Args:
        polygon: 多边形顶点坐标 [x1, y1, x2, y2, ...]
        window_bounds: (x_min, y_min, x_max, y_max) 窗口边界

    Returns:
        裁剪后的多边形顶点坐标
    """
    if len(polygon) < 6:
        return []

    x_min, y_min, x_max, y_max = window_bounds

    # 转换为点列表
    points = []
    for i in range(0, len(polygon), 2):
        points.append([polygon[i], polygon[i + 1]])

    # 对四条边依次进行裁剪
    edges = [
        (x_min, 0, 'left'),
        (x_max, 0, 'right'),
        (y_min, 1, 'bottom'),
        (y_max, 1, 'top')
    ]

    for edge_val, coord_idx, edge_type in edges:
        if not points:
            break

        new_points = []
        for i in range(len(points)):
            curr_point = points[i]
            prev_point = points[i - 1]

            if edge_type in ['left', 'right']:
                curr_inside = (curr_point[0] >= x_min) if edge_type == 'left' else (curr_point[0] <= x_max)
                prev_inside = (prev_point[0] >= x_min) if edge_type == 'left' else (prev_point[0] <= x_max)
            else:
                curr_inside = (curr_point[1] >= y_min) if edge_type == 'bottom' else (curr_point[1] <= y_max)
                prev_inside = (prev_point[1] >= y_min) if edge_type == 'bottom' else (prev_point[1] <= y_max)

            if curr_inside:
                if not prev_inside:
                    # 计算交点
                    t = 0.0
                    if edge_type in ['left', 'right']:
                        if curr_point[0] != prev_point[0]:
                            t = (edge_val - prev_point[0]) / (curr_point[0] - prev_point[0])
                    else:
                        if curr_point[1] != prev_point[1]:
                            t = (edge_val - prev_point[1]) / (curr_point[1] - prev_point[1])

                    intersection = [
                        prev_point[0] + t * (curr_point[0] - prev_point[0]),
                        prev_point[1] + t * (curr_point[1] - prev_point[1])
                    ]
                    new_points.append(intersection)
                new_points.append(curr_point)
            elif prev_inside:
                # 计算交点
                t = 0.0
                if edge_type in ['left', 'right']:
                    if curr_point[0] != prev_point[0]:
                        t = (edge_val - prev_point[0]) / (curr_point[0] - prev_point[0])
                else:
                    if curr_point[1] != prev_point[1]:
                        t = (edge_val - prev_point[1]) / (curr_point[1] - prev_point[1])

                intersection = [
                    prev_point[0] + t * (curr_point[0] - prev_point[0]),
                    prev_point[1] + t * (curr_point[1] - prev_point[1])
                ]
                new_points.append(intersection)

        points = new_points

    # 转换回扁平列表
    result = []
    for point in points:
        result.extend([
            max(0.0, min(1.0, point[0])),
            max(0.0, min(1.0, point[1]))
        ])

    return result