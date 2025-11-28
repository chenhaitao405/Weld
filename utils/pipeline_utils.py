import os
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    import matplotlib
    from matplotlib import font_manager
    MATPLOTLIB_AVAILABLE = True
    DEFAULT_FONT_FAMILIES = ['AR PL UKai CN', 'Noto Sans CJK JP', 'DejaVu Sans']
    matplotlib.rcParams['font.sans-serif'] = DEFAULT_FONT_FAMILIES
    matplotlib.rcParams['axes.unicode_minus'] = False
except ImportError:  # pragma: no cover
    font_manager = None
    MATPLOTLIB_AVAILABLE = False
    DEFAULT_FONT_FAMILIES = ['DejaVu Sans']

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:  # pragma: no cover
    Image = ImageDraw = ImageFont = None
    PIL_AVAILABLE = False


COLOR_PALETTE = [
    (255, 99, 71),
    (102, 204, 255),
    (144, 238, 144),
    (255, 215, 0),
    (226, 110, 168),
    (0, 191, 255),
    (255, 140, 0),
    (138, 43, 226),
    (60, 179, 113),
    (255, 20, 147),
]


def load_image(image_path: Path) -> np.ndarray:
    """读取图像并标准化通道数"""
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    image = normalize_input_image(image)
    return ensure_color(image)


def normalize_input_image(image: np.ndarray) -> np.ndarray:
    """去除多余通道，便于后续处理"""
    if image.ndim == 2:
        return image

    channels = image.shape[2]
    if channels == 1:
        return image[:, :, 0]
    if channels == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if channels > 4:
        return image[:, :, :3]
    return image


def ensure_color(image: np.ndarray) -> np.ndarray:
    """确保图像为3通道BGR"""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    channels = image.shape[2]
    if channels == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if channels == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if channels > 3:
        return image[:, :, :3]
    return image


def prepare_seg_input(enhanced_roi: np.ndarray) -> np.ndarray:
    """转换增强后的ROI以符合YOLO分割模型输入"""
    seg_input = enhanced_roi

    if seg_input.dtype != np.uint8:
        seg_input = cv2.normalize(seg_input, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if seg_input.ndim == 2:
        seg_input = np.repeat(seg_input[:, :, None], 3, axis=2)
    elif seg_input.ndim == 3:
        channels = seg_input.shape[2]
        if channels == 1:
            seg_input = np.repeat(seg_input, 3, axis=2)
        elif channels == 4:
            seg_input = cv2.cvtColor(seg_input, cv2.COLOR_BGRA2BGR)
        elif channels > 3:
            seg_input = seg_input[:, :, :3]

    return np.ascontiguousarray(seg_input)


class FontRenderer:
    """基于Pillow的文字渲染器，支持自动字体检测"""

    def __init__(self,
                 font_path: Optional[str] = None,
                 font_size: int = 20,
                 preferred_families: Optional[Sequence[str]] = None):
        self.font_size = font_size
        self.preferred_families = preferred_families or DEFAULT_FONT_FAMILIES
        self.font_path = font_path or self._auto_detect_font()
        self._font = self._load_font(self.font_path, font_size)

    @property
    def available(self) -> bool:
        return self._font is not None and PIL_AVAILABLE

    def draw(self, canvas: np.ndarray, text: str,
             origin: Tuple[int, int], bgr_color: Tuple[int, int, int]):
        """在画布上绘制文本，若字体不可用则回退到OpenCV"""
        if not self.available:
            cv2.putText(canvas, text, origin,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        bgr_color, 2, lineType=cv2.LINE_AA)
            return

        canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(canvas_rgb)
        draw_ctx = ImageDraw.Draw(pil_img)
        rgb_color = (bgr_color[2], bgr_color[1], bgr_color[0])
        draw_ctx.text(origin, text, font=self._font, fill=rgb_color)
        updated = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        canvas[:, :, :] = updated

    def _auto_detect_font(self) -> Optional[str]:
        if not (MATPLOTLIB_AVAILABLE and font_manager):
            return None

        for family in self.preferred_families:
            try:
                font_prop = font_manager.FontProperties(family=family)
                font_path = font_manager.findfont(font_prop, fallback_to_default=False)
            except Exception:
                continue
            if font_path and os.path.exists(font_path):
                return font_path
        return None

    @staticmethod
    def _load_font(font_path: Optional[str], font_size: int):
        if not (PIL_AVAILABLE and font_path):
            return None
        try:
            return ImageFont.truetype(font_path, font_size)
        except OSError:
            return None


def _draw_text(canvas: np.ndarray,
               text: str,
               origin: Tuple[int, int],
               bgr_color: Tuple[int, int, int],
               font_renderer: Optional[FontRenderer]):
    if font_renderer and font_renderer.available:
        font_renderer.draw(canvas, text, origin, bgr_color)
    else:
        cv2.putText(canvas, text, origin,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    bgr_color, 2, lineType=cv2.LINE_AA)


def draw_detection_instance(canvas: np.ndarray,
                            bbox: Sequence[float],
                            label: str,
                            score: Optional[float],
                            class_id: int,
                            font_renderer: Optional[FontRenderer] = None,
                            polygon: Optional[Sequence[Sequence[float]]] = None,
                            color_palette: Optional[Sequence[Tuple[int, int, int]]] = None,
                            mask_alpha: float = 0.3):
    """在画布上绘制检测/分割实例（自动处理中文字体）。"""
    if color_palette is None or len(color_palette) == 0:
        color_palette = COLOR_PALETTE
    color = color_palette[class_id % len(color_palette)]

    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    pt1 = (x1, y1)
    pt2 = (x2, y2)

    if polygon and len(polygon) >= 3:
        pts = np.array([
            (int(round(px)), int(round(py)))
            for px, py in polygon
        ], dtype=np.int32)
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, mask_alpha, canvas, 1 - mask_alpha, 0, canvas)
        cv2.polylines(canvas, [pts], True, color, 2)
    else:
        cv2.rectangle(canvas, pt1, pt2, color, 2)

    caption = label if score is None else f"{label}:{score:.2f}"
    text_org = (pt1[0], max(0, pt1[1] - 5))
    _draw_text(canvas, caption, text_org, color, font_renderer)
