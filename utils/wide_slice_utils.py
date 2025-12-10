from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np


@dataclass(frozen=True)
class WideSliceParams:
    """Parameter set describing how wide slices should be generated."""

    aspect_ratio_threshold: float = 4.0
    window_ratio: float = 2.0
    overlap: float = 0.3
    target_size: int = 1120


@dataclass
class WideSlicePatch:
    """A single horizontally cropped slice of the ROI."""

    image: np.ndarray
    x_offset: float
    width: int
    height: int
    is_full_window: bool


@dataclass
class WideSlicePlan:
    """Planning artifact that captures how an ROI should be sliced."""

    is_wide: bool
    aspect_ratio: float
    roi_height: int
    roi_width: int
    patches: List[WideSlicePatch]


def build_wide_slice_plan(image: np.ndarray, params: WideSliceParams) -> WideSlicePlan:
    """Return slicing plan for the given image based on the provided params."""
    roi_height, roi_width = image.shape[:2]
    if roi_height == 0 or roi_width == 0:
        return WideSlicePlan(
            is_wide=False,
            aspect_ratio=0.0,
            roi_height=roi_height,
            roi_width=roi_width,
            patches=[]
        )

    aspect_ratio = roi_width / max(1, roi_height)
    patches: List[WideSlicePatch] = []

    if aspect_ratio < params.aspect_ratio_threshold:
        patches.append(
            WideSlicePatch(
                image=image.copy(),
                x_offset=0.0,
                width=roi_width,
                height=roi_height,
                is_full_window=False
            )
        )
        return WideSlicePlan(
            is_wide=False,
            aspect_ratio=aspect_ratio,
            roi_height=roi_height,
            roi_width=roi_width,
            patches=patches
        )

    window_width = max(1, int(round(roi_height * params.window_ratio)))
    window_width = min(window_width, roi_width)
    stride = max(1, int(round(window_width * (1 - params.overlap))))

    start_x = 0
    while start_x < roi_width:
        end_x = min(start_x + window_width, roi_width)
        width = end_x - start_x
        patch_image = image[:, start_x:end_x].copy()
        patches.append(
            WideSlicePatch(
                image=patch_image,
                x_offset=float(start_x),
                width=int(width),
                height=roi_height,
                is_full_window=(width == window_width)
            )
        )
        if end_x >= roi_width:
            break
        start_x += stride

    return WideSlicePlan(
        is_wide=True,
        aspect_ratio=aspect_ratio,
        roi_height=roi_height,
        roi_width=roi_width,
        patches=patches
    )


def resize_slice_to_square(image: np.ndarray, target_size: int) -> np.ndarray:
    """Resize slice to a square patch compatible with downstream detectors."""
    if target_size <= 0 or image.size == 0:
        return image
    return cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_AREA)


def stack_wide_slice_pair(top_patch: WideSlicePatch, bottom_patch: WideSlicePatch) -> np.ndarray:
    """Vertically stack two full-width slices into a single buffer."""
    if top_patch.width != bottom_patch.width:
        raise ValueError("Full-width slices must share the same width before stacking")
    return np.vstack([top_patch.image, bottom_patch.image])


__all__ = [
    "WideSliceParams",
    "WideSlicePatch",
    "WideSlicePlan",
    "build_wide_slice_plan",
    "resize_slice_to_square",
    "stack_wide_slice_pair"
]
