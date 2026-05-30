"""star_detect 单元测试（用合成红色五角星）。"""
from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from app.engine.star_detect import StarDetectParams, detect_stars


def _draw_filled_star(img_mask: np.ndarray, cx: int, cy: int, r_outer: int, r_inner: int) -> None:
    pts = []
    for k in range(10):
        angle = -math.pi / 2 + k * math.pi / 5
        r = r_outer if k % 2 == 0 else r_inner
        pts.append((int(cx + r * math.cos(angle)), int(cy + r * math.sin(angle))))
    cv2.fillPoly(img_mask, [np.array(pts, dtype=np.int32)], 255)


def test_detects_single_star():
    mask = np.zeros((300, 300), dtype=np.uint8)
    _draw_filled_star(mask, 150, 150, 40, 18)
    stars = detect_stars(mask)
    assert len(stars) == 1
    s = stars[0]
    assert abs(s.x - 150) < 3 and abs(s.y - 150) < 3
    # 五角星应有 ~5 个凸缺陷
    assert 3 <= s.concavity_count <= 7


def test_detects_multiple_stars():
    mask = np.zeros((400, 600), dtype=np.uint8)
    centers = [(100, 100), (300, 150), (500, 300)]
    for cx, cy in centers:
        _draw_filled_star(mask, cx, cy, 30, 13)
    stars = detect_stars(mask)
    assert len(stars) == 3
    # 检测出的位置应与画的位置一一对应（允许 5px 误差）
    detected = {(round(s.x), round(s.y)) for s in stars}
    for cx, cy in centers:
        assert any(abs(dx - cx) < 5 and abs(dy - cy) < 5 for dx, dy in detected)


def test_rejects_circle_when_concavity_required():
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(mask, (100, 100), 40, 255, thickness=-1)
    stars = detect_stars(mask, StarDetectParams(use_concavity=True))
    assert stars == []


def test_accepts_circle_when_all_shape_filters_off():
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(mask, (100, 100), 40, 255, thickness=-1)
    stars = detect_stars(
        mask, StarDetectParams(use_concavity=False, use_solidity=False)
    )
    assert len(stars) == 1


def test_rejects_circle_by_solidity_alone():
    """空心 / 数字 / 圆形都因 solidity 接近 1.0 被排除，无需依赖凸缺陷数。"""
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(mask, (100, 100), 40, 255, thickness=-1)
    stars = detect_stars(mask, StarDetectParams(use_concavity=False))
    assert stars == []


def test_detects_hollow_star():
    """实际图纸里失效点常是空心红五角星：仅外轮廓有色。"""
    mask = np.zeros((300, 300), dtype=np.uint8)
    pts = []
    for k in range(10):
        angle = -math.pi / 2 + k * math.pi / 5
        r = 40 if k % 2 == 0 else 18
        pts.append((int(150 + r * math.cos(angle)), int(150 + r * math.sin(angle))))
    cv2.polylines(mask, [np.array(pts, dtype=np.int32)], isClosed=True,
                  color=255, thickness=2)
    stars = detect_stars(mask)
    assert len(stars) == 1
    s = stars[0]
    assert abs(s.x - 150) < 3 and abs(s.y - 150) < 3
    # 实心化后 solidity 应落在五角星范围
    assert 0.35 <= s.solidity <= 0.85


def test_detects_hollow_star_with_text_inside():
    """星内有数字（如 '30m'）不影响中心定位：填实步骤覆盖内部。"""
    mask = np.zeros((300, 300), dtype=np.uint8)
    pts = []
    for k in range(10):
        angle = -math.pi / 2 + k * math.pi / 5
        r = 40 if k % 2 == 0 else 18
        pts.append((int(150 + r * math.cos(angle)), int(150 + r * math.sin(angle))))
    cv2.polylines(mask, [np.array(pts, dtype=np.int32)], isClosed=True,
                  color=255, thickness=2)
    # 内部画一些红色文字像素
    cv2.putText(mask, "30m", (130, 158), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, 255, 1, cv2.LINE_AA)
    stars = detect_stars(mask)
    assert len(stars) == 1
    s = stars[0]
    assert abs(s.x - 150) < 5 and abs(s.y - 150) < 5


def test_detects_two_overlapping_hollow_stars():
    """两颗空心星明显重叠：连通域合体，应靠距离变换峰值各自定位。"""
    mask = np.zeros((300, 500), dtype=np.uint8)
    centers = [(200, 150), (290, 150)]   # 间距 90 < 2*outer_radius(40), 有重叠
    for cx, cy in centers:
        pts = []
        for k in range(10):
            angle = -math.pi / 2 + k * math.pi / 5
            r = 40 if k % 2 == 0 else 18
            pts.append((int(cx + r * math.cos(angle)), int(cy + r * math.sin(angle))))
        cv2.polylines(mask, [np.array(pts, dtype=np.int32)], isClosed=True,
                      color=255, thickness=2)
    stars = detect_stars(mask)
    assert len(stars) == 2, f"应识别 2 颗重叠星，实际 {len(stars)}"
    detected = sorted((round(s.x), round(s.y)) for s in stars)
    for got, want in zip(detected, sorted(centers)):
        assert abs(got[0] - want[0]) <= 6 and abs(got[1] - want[1]) <= 6, (
            f"中心 {got} 偏离预期 {want} 过多"
        )


def test_detects_two_vertically_stacked_overlapping_stars():
    """两颗星垂直堆叠重叠，aspect ratio 接近 1 也要靠峰值分出两个。"""
    mask = np.zeros((400, 300), dtype=np.uint8)
    centers = [(150, 130), (150, 200)]  # 垂直堆叠，间距 70，r_outer 40
    for cx, cy in centers:
        pts = []
        for k in range(10):
            angle = -math.pi / 2 + k * math.pi / 5
            r = 40 if k % 2 == 0 else 18
            pts.append((int(cx + r * math.cos(angle)), int(cy + r * math.sin(angle))))
        cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 255)
    stars = detect_stars(mask)
    assert len(stars) == 2, f"应识别 2 颗垂直重叠星，实际 {len(stars)}"
    # y 应该排在 130 和 200 附近
    ys = sorted(round(s.y) for s in stars)
    assert abs(ys[0] - 130) <= 6 and abs(ys[1] - 200) <= 6


def test_detects_two_overlapping_stars_with_unimodal_distance():
    """中等重叠（中心距 ≈ outer_radius）时距离变换可能只有 1 个峰，
    这时要靠主轴拆分兜底，仍应给出两颗星。"""
    mask = np.zeros((300, 400), dtype=np.uint8)
    centers = [(180, 150), (210, 150)]   # 中心距 30，r_outer=30，重叠较深
    for cx, cy in centers:
        pts = []
        for k in range(10):
            angle = -math.pi / 2 + k * math.pi / 5
            r = 30 if k % 2 == 0 else 13
            pts.append((int(cx + r * math.cos(angle)), int(cy + r * math.sin(angle))))
        cv2.polylines(mask, [np.array(pts, dtype=np.int32)], isClosed=True,
                      color=255, thickness=2)
    stars = detect_stars(mask)
    assert len(stars) == 2, f"中等重叠应给出 2 颗，实际 {len(stars)}"


def test_detects_two_moderately_overlapping_filled_stars():
    """实心星中度重叠（中心间距 ≈ outer_radius），靠峰值分离仍应给出 2 个中心。"""
    mask = np.zeros((300, 500), dtype=np.uint8)
    centers = [(200, 150), (260, 150)]   # 相距 60，r_outer 40 → 有明显重叠但仍有谷值
    for cx, cy in centers:
        pts = []
        for k in range(10):
            angle = -math.pi / 2 + k * math.pi / 5
            r = 40 if k % 2 == 0 else 18
            pts.append((int(cx + r * math.cos(angle)), int(cy + r * math.sin(angle))))
        cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 255)
    stars = detect_stars(mask)
    assert len(stars) == 2, f"应识别 2 颗重叠实心星，实际 {len(stars)}"


def test_rejects_isolated_text():
    """孤立红文字（如 'P3.5' 标签）不应被识别成星。"""
    mask = np.zeros((200, 400), dtype=np.uint8)
    cv2.putText(mask, "P3.5", (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, 255, 2, cv2.LINE_AA)
    stars = detect_stars(mask)
    assert stars == []


def test_area_filter():
    mask = np.zeros((400, 400), dtype=np.uint8)
    _draw_filled_star(mask, 100, 100, 8, 3)    # 太小
    _draw_filled_star(mask, 300, 300, 40, 18)  # 正常
    stars = detect_stars(mask, StarDetectParams(min_area=200))
    assert len(stars) == 1
    assert abs(stars[0].x - 300) < 3
