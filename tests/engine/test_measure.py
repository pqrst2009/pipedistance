"""measure 单元测试。"""
from __future__ import annotations

import pytest

from app.engine.measure import (
    measure_all,
    pairwise_along_pipe,
    project_points,
)


def test_project_to_horizontal_line():
    pipelines = [[(0.0, 0.0), (100.0, 0.0)]]
    proj = project_points([(30.0, 5.0), (70.0, -3.0)], pipelines)
    assert proj[0] is not None
    assert proj[1] is not None
    assert proj[0].pipeline_index == 0
    assert proj[0].proj_xy == pytest.approx((30.0, 0.0))
    assert proj[0].along_param_px == pytest.approx(30.0)
    assert proj[0].offset_px == pytest.approx(5.0)
    assert proj[1].along_param_px == pytest.approx(70.0)


def test_project_picks_nearest_pipeline():
    pipelines = [
        [(0.0, 0.0), (100.0, 0.0)],
        [(0.0, 200.0), (100.0, 200.0)],
    ]
    proj = project_points([(50.0, 190.0)], pipelines)
    assert proj[0].pipeline_index == 1


def test_offset_filter():
    pipelines = [[(0.0, 0.0), (100.0, 0.0)]]
    proj = project_points([(50.0, 999.0)], pipelines, max_offset_px=50.0)
    assert proj == [None]


def test_pairwise_distances_along_horizontal():
    pipelines = [[(0.0, 0.0), (100.0, 0.0)]]
    points = [(10.0, 1.0), (40.0, -1.0), (90.0, 0.0)]
    proj = project_points(points, pipelines)
    pairs = pairwise_along_pipe(proj, scales=[2.0])
    # 3 个点 → 3 对
    assert len(pairs) == 3
    # 点 0 ↔ 1：沿管 30px，直线 sqrt(30^2 + 2^2)≈30.066px；按 2.0m/px
    p01 = next(p for p in pairs if p.from_index == 0 and p.to_index == 1)
    assert p01.along_pipe_px == pytest.approx(30.0)
    assert p01.along_pipe_m == pytest.approx(60.0)
    assert p01.straight_m == pytest.approx(p01.straight_px * 2.0)


def test_pairs_only_within_same_pipeline():
    """include_cross_pipeline=False 时只保留同管线对。"""
    pipelines = [
        [(0.0, 0.0), (100.0, 0.0)],
        [(0.0, 200.0), (100.0, 200.0)],
    ]
    points = [(10.0, 5.0), (50.0, 5.0), (50.0, 205.0)]
    _, pairs = measure_all(
        points, pipelines, scales=[1.0, 0.5], include_cross_pipeline=False
    )
    assert len(pairs) == 1
    p = pairs[0]
    assert p.pipeline_index == 0
    assert {p.from_index, p.to_index} == {0, 1}


def test_pairs_include_cross_pipeline_by_default():
    """include_cross_pipeline=True（默认）：所有两两都给；跨管线只给直线。"""
    pipelines = [
        [(0.0, 0.0), (100.0, 0.0)],
        [(0.0, 200.0), (100.0, 200.0)],
    ]
    points = [(10.0, 5.0), (50.0, 5.0), (50.0, 205.0)]
    # 跨管线对用 fallback_scale 算米值
    _, pairs = measure_all(
        points, pipelines, scales=[1.0, 0.5], fallback_scale=1.0,
    )
    assert len(pairs) == 3
    same = [p for p in pairs if p.pipeline_index >= 0]
    cross = [p for p in pairs if p.pipeline_index < 0]
    assert len(same) == 1
    assert len(cross) == 2
    for p in cross:
        assert p.along_pipe_m is None        # 跨管线没有沿管距离
        assert p.straight_px > 0
        assert p.straight_m is not None      # 但用 fallback_scale 仍能给米值


def test_along_pipe_follows_corner():
    """L 形管线：折角点之后的距离应沿折线算，而不是直线。"""
    pipelines = [[(0.0, 0.0), (100.0, 0.0), (100.0, 100.0)]]
    points = [(10.0, 5.0), (105.0, 80.0)]  # 一点在水平段，一点在垂直段
    proj = project_points(points, pipelines)
    pairs = pairwise_along_pipe(proj)
    p = pairs[0]
    # 沿管距离 ≈ (100-10) + 80 = 170，远大于直线 ≈ sqrt(95^2 + 75^2)≈121
    assert p.along_pipe_px == pytest.approx(170.0, abs=1.0)
    assert p.straight_px < p.along_pipe_px


def test_fallback_scale():
    pipelines = [[(0.0, 0.0), (100.0, 0.0)]]
    proj = project_points([(10.0, 0.0), (90.0, 0.0)], pipelines)
    # 该管线没单独标定，scales=[None]；回退到 fallback
    pairs = pairwise_along_pipe(proj, scales=[None], fallback_scale=3.0)
    assert pairs[0].along_pipe_m == pytest.approx(80.0 * 3.0)


def test_no_scale_returns_none_meters():
    pipelines = [[(0.0, 0.0), (100.0, 0.0)]]
    proj = project_points([(10.0, 0.0), (90.0, 0.0)], pipelines)
    pairs = pairwise_along_pipe(proj)  # 没有 scales, 没有 fallback
    assert pairs[0].along_pipe_m is None
    assert pairs[0].straight_m is None
    assert pairs[0].along_pipe_px == pytest.approx(80.0)
