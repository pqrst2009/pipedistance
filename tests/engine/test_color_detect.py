"""color_detect 单元测试（用合成图）。"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.engine.color_detect import (
    GreenThresholds,
    RedThresholds,
    clean_mask,
    green_mask,
    red_mask,
)


@pytest.fixture
def synthetic_drawing() -> np.ndarray:
    """合成 200x200 白底，画一条浅绿水平线 + 两个红色实心圆（代表星）。"""
    img = np.full((200, 200, 3), 255, dtype=np.uint8)
    # BGR 浅绿
    cv2.line(img, (20, 100), (180, 100), (160, 230, 180), thickness=3)
    # BGR 红
    cv2.circle(img, (60, 100), 6, (40, 40, 220), thickness=-1)
    cv2.circle(img, (150, 100), 6, (40, 40, 220), thickness=-1)
    return img


def test_green_mask_hits_line(synthetic_drawing: np.ndarray):
    m = green_mask(synthetic_drawing)
    # 线条中段必命中，线外大部分像素为 0
    assert m[100, 100] == 255
    assert m[10, 10] == 0
    # 命中数量应远大于阈值（线长 160，粗 3）
    assert int((m > 0).sum()) > 200


def test_red_mask_hits_circles(synthetic_drawing: np.ndarray):
    m = red_mask(synthetic_drawing)
    assert m[100, 60] == 255
    assert m[100, 150] == 255
    # 绿线区域不应被误判为红
    assert m[100, 100] == 0


def test_red_mask_handles_wraparound():
    """纯红 (0,0,255) 的 H = 0，须被低段命中。"""
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[:] = (0, 0, 255)
    assert (red_mask(img) == 255).all()


def test_thresholds_are_tunable():
    """收紧阈值后浅淡颜色应被排除。"""
    img = np.full((10, 10, 3), 255, dtype=np.uint8)
    img[:] = (200, 240, 210)  # 极淡绿
    loose = green_mask(img, GreenThresholds(s_min=10))
    tight = green_mask(img, GreenThresholds(s_min=200))
    assert (loose > 0).sum() > (tight > 0).sum()


def test_clean_mask_removes_small_components():
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(mask, (10, 10), (50, 50), 255, thickness=-1)  # 大块
    mask[80, 80] = 255  # 孤立噪点
    cleaned = clean_mask(mask, open_ksize=3, close_ksize=0, min_area=100)
    assert cleaned[30, 30] == 255  # 大块保留
    assert cleaned[80, 80] == 0    # 噪点被清除
