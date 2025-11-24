"""
常量定义模块
定义项目中使用的各种常量
"""

# 支持的图像格式
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']

# YOLO相关常量
YOLO_LABEL_EXTENSION = '.txt'
YOLO_CONFIG_FILE = 'dataset.yaml'

# 默认参数
DEFAULT_OVERLAP_RATIO = 0.5
DEFAULT_WINDOW_SIZE = 640
DEFAULT_JPEG_QUALITY = 95
DEFAULT_TRAIN_VAL_SPLIT = 0.8

# 最小尺寸阈值
MIN_BBOX_RATIO = 0.01
MIN_POLYGON_AREA_RATIO = 0.001
MIN_POLYGON_POINTS = 3

# 增强模式
ENHANCE_MODES = ['original', 'windowing']

# 标签格式
LABEL_FORMATS = ['labelme', 'yolo']
LABEL_MODES = ['det', 'seg']