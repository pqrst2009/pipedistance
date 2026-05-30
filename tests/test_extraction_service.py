"""ExtractionService 集成测试：用合成图跑完整流程。"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
import pytest

from app.domain.models import LengthLabel, ReviewState, new_id
from app.services.extraction_service import ExtractionService


def _draw_filled_star(img: np.ndarray, cx: int, cy: int, r_outer: int, r_inner: int) -> None:
    pts = []
    for k in range(10):
        angle = -math.pi / 2 + k * math.pi / 5
        r = r_outer if k % 2 == 0 else r_inner
        pts.append((int(cx + r * math.cos(angle)), int(cy + r * math.sin(angle))))
    cv2.fillPoly(img, [np.array(pts, dtype=np.int32)], (40, 40, 220))


@pytest.fixture
def synthetic_image() -> np.ndarray:
    """600x400 白底，画水平浅绿管线 + 3 颗红星。"""
    img = np.full((400, 600, 3), 255, dtype=np.uint8)
    # 浅绿水平管线，500 px 长
    cv2.line(img, (50, 200), (550, 200), (160, 230, 180), thickness=4)
    # 三颗五角星
    for cx in (120, 300, 480):
        _draw_filled_star(img, cx, 200, 18, 8)
    return img


def _make_label(text_value_m: float, bbox) -> LengthLabel:
    return LengthLabel(
        id=new_id("len_"),
        drawing_id="d1",
        text=f"{text_value_m}m",
        value_m=text_value_m,
        bbox=bbox,
        confidence=0.95,
    )


def test_full_pipeline(synthetic_image: np.ndarray):
    # OCR 标注：标注位置不重要，只要能匹配到唯一管线即可
    labels = [_make_label(1000.0, (250, 220, 20, 12))]  # 假设全长 1000m → 2 m/px
    svc = ExtractionService()
    result = svc.run(synthetic_image, "d1", labels)

    # 至少 1 条管线
    assert len(result.pipelines) >= 1
    # 3 颗星
    assert len(result.failure_points) == 3
    codes = sorted(fp.code for fp in result.failure_points)
    assert codes == ["F1", "F2", "F3"]
    # 三星两两 = 3 对（同管线）
    assert len(result.measures) == 3
    # 比例尺应在 ~2 m/px 附近（管线长 500px，标注 1000m）
    scale = result.global_scale_m_per_px
    assert scale is not None
    assert scale == pytest.approx(2.0, rel=0.1)
    # 任一对距离应 > 0
    for m in result.measures:
        assert m.along_pipe_px > 0
        assert m.along_pipe_m is not None and m.along_pipe_m > 0


def test_no_labels_still_emits_pixel_measures(synthetic_image: np.ndarray):
    svc = ExtractionService()
    result = svc.run(synthetic_image, "d1", length_labels=[])
    assert result.global_scale_m_per_px is None
    # 没标定时米值为 None
    assert all(m.along_pipe_m is None for m in result.measures)
    # 像素值仍可用
    assert all(m.along_pipe_px > 0 for m in result.measures)
    # 都置 need_review
    assert all(m.need_review for m in result.measures)


def test_star_far_from_pipeline_marked_review():
    """实际图纸场景：空心红五角星 + 星内数字 + 绿管线压在星心。"""
    img = np.full((400, 600, 3), 255, dtype=np.uint8)
    # 浅绿水平管线
    cv2.line(img, (50, 200), (550, 200), (160, 230, 180), thickness=4)
    # 三颗空心星
    for cx in (120, 300, 480):
        _draw_hollow_star(img, cx, 200, 22, 10, thickness=2)
        # 星心附近写红色 "30m"
        cv2.putText(img, "30m", (cx - 18, 207), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (40, 40, 220), 1, cv2.LINE_AA)

    svc = ExtractionService()
    result = svc.run(img, "d1", length_labels=[])
    assert len(result.failure_points) == 3, (
        f"应识别出 3 颗空心星，实际 {len(result.failure_points)}"
    )
    # 中心位置应靠近预设坐标（±5 像素）
    detected_x = sorted(round(fp.raw_px[0]) for fp in result.failure_points)
    for got, want in zip(detected_x, [120, 300, 480]):
        assert abs(got - want) <= 5
    # 三颗星均在同一条管线上 → 3 对距离
    assert len(result.measures) == 3
    img = np.full((400, 600, 3), 255, dtype=np.uint8)
    cv2.line(img, (50, 200), (550, 200), (160, 230, 180), thickness=4)
    # 一颗星离管线很远（y=350，管线在 y=200）
    _draw_filled_star(img, 300, 350, 18, 8)
    svc = ExtractionService()
    result = svc.run(img, "d1", length_labels=[])
    assert len(result.failure_points) == 1
    fp = result.failure_points[0]
    # 偏离 150px > 默认 max_failure_offset_px(60) → 投影失败
    assert fp.state == ReviewState.NEED_REVIEW


def _draw_hollow_star(img, cx: int, cy: int, r_outer: int, r_inner: int, thickness: int = 2):
    pts = []
    for k in range(10):
        angle = -math.pi / 2 + k * math.pi / 5
        r = r_outer if k % 2 == 0 else r_inner
        pts.append((int(cx + r * math.cos(angle)), int(cy + r * math.sin(angle))))
    cv2.polylines(img, [np.array(pts, dtype=np.int32)], isClosed=True,
                  color=(40, 40, 220), thickness=thickness)


def test_full_pipeline_hollow_stars_with_text():
    """实际图纸场景：空心红五角星 + 星内数字 + 绿管线压在星心。"""
    img = np.full((400, 600, 3), 255, dtype=np.uint8)
    # 浅绿水平管线
    cv2.line(img, (50, 200), (550, 200), (160, 230, 180), thickness=4)
    # 三颗空心星 + 星内红色 "30m"
    for cx in (120, 300, 480):
        _draw_hollow_star(img, cx, 200, 22, 10, thickness=2)
        cv2.putText(img, "30m", (cx - 18, 207), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (40, 40, 220), 1, cv2.LINE_AA)

    svc = ExtractionService()
    result = svc.run(img, "d1", length_labels=[])
    assert len(result.failure_points) == 3, (
        f"应识别出 3 颗空心星，实际 {len(result.failure_points)}"
    )
    detected_x = sorted(round(fp.raw_px[0]) for fp in result.failure_points)
    for got, want in zip(detected_x, [120, 300, 480]):
        assert abs(got - want) <= 5
    # 三颗星均在同一条管线上 → 3 对距离
    assert len(result.measures) == 3
