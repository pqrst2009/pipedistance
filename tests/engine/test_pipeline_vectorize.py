"""pipeline_vectorize 单元测试。"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.engine.pipeline_vectorize import (
    VectorizeParams,
    bridge_through_points,
    vectorize,
)


def _draw_line(mask: np.ndarray, p1, p2, thickness: int = 3) -> np.ndarray:
    cv2.line(mask, p1, p2, 255, thickness=thickness)
    return mask


def test_straight_line_yields_single_branch():
    mask = np.zeros((200, 400), dtype=np.uint8)
    _draw_line(mask, (20, 100), (380, 100), thickness=3)
    polylines = vectorize(mask, VectorizeParams(rdp_epsilon=2.0))
    assert len(polylines) == 1
    pts = polylines[0]
    # 起止应近似水平线两端
    xs = sorted(p[0] for p in pts)
    assert xs[0] < 30 and xs[-1] > 370
    # 直线经 RDP 简化后应只剩 2-3 个点
    assert len(pts) <= 4


def test_l_shape_yields_one_branch_with_corner():
    mask = np.zeros((300, 300), dtype=np.uint8)
    _draw_line(mask, (50, 50), (250, 50), thickness=3)   # 水平
    _draw_line(mask, (250, 50), (250, 250), thickness=3)  # 垂直
    polylines = vectorize(mask, VectorizeParams(rdp_epsilon=2.0))
    # 拐角处 L 形是单分支，应保留拐点
    assert len(polylines) == 1
    pts = polylines[0]
    assert any(220 < x < 270 and 30 < y < 70 for x, y in pts)


def test_t_junction_yields_three_branches():
    mask = np.zeros((300, 300), dtype=np.uint8)
    _draw_line(mask, (50, 150), (250, 150), thickness=3)  # 横
    _draw_line(mask, (150, 30), (150, 150), thickness=3)  # 竖
    polylines = vectorize(mask, VectorizeParams(rdp_epsilon=2.0, min_branch_px=20))
    # T 字应被切成 3 段
    assert len(polylines) == 3


def test_short_noise_branches_are_dropped():
    mask = np.zeros((100, 100), dtype=np.uint8)
    _draw_line(mask, (10, 50), (90, 50), thickness=3)  # 主干
    mask[50, 50] = 255                                  # 单像素噪声不连
    polylines = vectorize(mask, VectorizeParams(min_branch_px=10))
    assert len(polylines) == 1


def test_empty_mask_returns_empty_list():
    mask = np.zeros((50, 50), dtype=np.uint8)
    assert vectorize(mask) == []


def test_bridge_merges_two_polylines_via_star():
    # 两条水平折线，端点在 (100, 50) 和 (140, 50)，星心 (120, 50)
    polys = [
        [(10.0, 50.0), (100.0, 50.0)],
        [(140.0, 50.0), (230.0, 50.0)],
    ]
    merged = bridge_through_points(polys, [(120.0, 50.0)], max_gap_px=30.0)
    assert len(merged) == 1
    pts = merged[0]
    assert pts[0] == (10.0, 50.0)
    assert pts[-1] == (230.0, 50.0)
    assert (120.0, 50.0) in pts  # 星心成为中转点


def test_bridge_skips_when_only_one_endpoint_near():
    polys = [[(0.0, 0.0), (100.0, 0.0)]]
    merged = bridge_through_points(polys, [(110.0, 0.0)], max_gap_px=20.0)
    # 单端点不能配对，原样返回
    assert merged == polys


def test_bridge_does_not_self_merge():
    """同一折线的两端都接近星心也不应触发自合并（环路）。"""
    polys = [[(100.0, 50.0), (50.0, 50.0), (100.0, 50.0)]]  # 极端情形：两端都靠近
    merged = bridge_through_points(polys, [(100.0, 50.0)], max_gap_px=20.0)
    assert len(merged) == 1
