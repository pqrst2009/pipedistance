"""calibration 单元测试。"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.engine.calibration import (
    global_scale,
    match_labels_to_pipelines,
    per_pipeline_scale,
)


@dataclass
class FakeLabel:
    id: str
    value_m: float
    bbox: tuple   # (x, y, w, h)


def test_match_single_label_single_pipeline():
    # 管线水平线长 100 像素
    pipelines = [[(0.0, 50.0), (100.0, 50.0)]]
    labels = [FakeLabel("l1", 200.0, (40, 30, 20, 10))]  # 中心 (50, 35)，离线 15px
    samples = match_labels_to_pipelines(labels, pipelines, max_match_dist_px=50)
    assert len(samples) == 1
    s = samples[0]
    assert s.pipeline_index == 0
    assert s.pipeline_px_len == pytest.approx(100.0)
    assert s.scale_m_per_px == pytest.approx(2.0)
    assert s.distance_px == pytest.approx(15.0)


def test_label_too_far_is_dropped():
    pipelines = [[(0.0, 0.0), (100.0, 0.0)]]
    labels = [FakeLabel("l1", 200.0, (0, 200, 20, 10))]  # 离线 ~205px
    assert match_labels_to_pipelines(labels, pipelines, max_match_dist_px=50) == []


def test_picks_nearest_of_multiple_pipelines():
    pipelines = [
        [(0.0, 0.0), (100.0, 0.0)],     # 顶部
        [(0.0, 200.0), (100.0, 200.0)],  # 底部
    ]
    labels = [FakeLabel("l1", 50.0, (40, 190, 20, 10))]  # 靠近底部
    samples = match_labels_to_pipelines(labels, pipelines)
    assert len(samples) == 1
    assert samples[0].pipeline_index == 1


def test_per_pipeline_scale_uses_median():
    pipelines = [[(0.0, 0.0), (100.0, 0.0)]]
    # 三个标注命中同一管线，比例分别 2.0 / 2.1 / 10.0 (异常值)
    labels = [
        FakeLabel("a", 200.0, (50, 5, 10, 10)),    # 2.0
        FakeLabel("b", 210.0, (50, 5, 10, 10)),    # 2.1
        FakeLabel("c", 1000.0, (50, 5, 10, 10)),  # 10.0
    ]
    samples = match_labels_to_pipelines(labels, pipelines)
    [scale] = per_pipeline_scale(samples, pipeline_count=1)
    assert scale is not None
    # 中位数应为 2.1，不被异常值带歪
    assert scale.scale_m_per_px == pytest.approx(2.1)
    assert scale.sample_count == 3


def test_per_pipeline_scale_none_when_no_match():
    pipelines = [
        [(0.0, 0.0), (100.0, 0.0)],
        [(200.0, 200.0), (300.0, 200.0)],
    ]
    labels = [FakeLabel("a", 200.0, (50, 5, 10, 10))]  # 只命中第一条
    samples = match_labels_to_pipelines(labels, pipelines)
    scales = per_pipeline_scale(samples, pipeline_count=2)
    assert scales[0] is not None
    assert scales[1] is None


def test_global_scale():
    pipelines = [
        [(0.0, 0.0), (100.0, 0.0)],
        [(0.0, 200.0), (50.0, 200.0)],
    ]
    labels = [
        FakeLabel("a", 200.0, (50, 5, 10, 10)),     # 2.0 m/px
        FakeLabel("b", 100.0, (25, 195, 10, 10)),   # 2.0 m/px
    ]
    samples = match_labels_to_pipelines(labels, pipelines)
    assert global_scale(samples) == pytest.approx(2.0)
    assert global_scale([]) is None
