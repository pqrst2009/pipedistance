"""颜色掩膜：浅绿管线 / 红色五角星。

设计：
- HSV 阈值是可调参数，默认覆盖常见图纸里的"浅绿管线"和"红色失效点星标"。
- 阈值不固化在算法里，而是通过 ``GreenThresholds`` / ``RedThresholds`` 数据类传入，
  方便后续 UI 暴露给用户微调。
- 红色在 HSV 上跨越 0°，需用两段拼接。

输出统一为 uint8 单通道掩膜（0/255）。
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class GreenThresholds:
    """浅绿管线 HSV 阈值。

    OpenCV HSV: H ∈ [0, 180], S ∈ [0, 255], V ∈ [0, 255]。
    浅绿覆盖：H 约 35-95，S 不强求高饱和（淡绿），V 偏高（线条本身亮）。
    """
    h_min: int = 35
    h_max: int = 95
    s_min: int = 40
    s_max: int = 255
    v_min: int = 80
    v_max: int = 255


@dataclass(frozen=True)
class RedThresholds:
    """红色五角星 HSV 阈值，跨 0° 取两段。"""
    h_low_max: int = 10       # 第一段 [0, h_low_max]
    h_high_min: int = 170     # 第二段 [h_high_min, 180]
    s_min: int = 80
    v_min: int = 60


def green_mask(image_bgr: np.ndarray, t: GreenThresholds | None = None) -> np.ndarray:
    """返回浅绿掩膜（0/255）。"""
    t = t or GreenThresholds()
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([t.h_min, t.s_min, t.v_min], dtype=np.uint8)
    upper = np.array([t.h_max, t.s_max, t.v_max], dtype=np.uint8)
    return cv2.inRange(hsv, lower, upper)


def red_mask(image_bgr: np.ndarray, t: RedThresholds | None = None) -> np.ndarray:
    """返回红色掩膜（0/255），HSV 双段合并。"""
    t = t or RedThresholds()
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    low = cv2.inRange(
        hsv,
        np.array([0, t.s_min, t.v_min], dtype=np.uint8),
        np.array([t.h_low_max, 255, 255], dtype=np.uint8),
    )
    high = cv2.inRange(
        hsv,
        np.array([t.h_high_min, t.s_min, t.v_min], dtype=np.uint8),
        np.array([180, 255, 255], dtype=np.uint8),
    )
    return cv2.bitwise_or(low, high)


def clean_mask(
    mask: np.ndarray,
    open_ksize: int = 3,
    close_ksize: int = 3,
    min_area: int = 0,
) -> np.ndarray:
    """形态学清理：先开后闭，去掉孤立噪点、闭合断裂。

    ``min_area > 0`` 时进一步移除面积过小的连通域。
    """
    if open_ksize > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    if close_ksize > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    if min_area > 0:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        keep = np.zeros_like(mask)
        for i in range(1, n):  # 0 = background
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                keep[labels == i] = 255
        mask = keep
    return mask
