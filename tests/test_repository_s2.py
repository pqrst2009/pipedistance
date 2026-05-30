"""Repository S2 持久化往返测试（内存 SQLite）。"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.domain.models import (
    Drawing,
    FailurePoint,
    MeasureResult,
    Pipeline,
    ReviewState,
    new_id,
)
from app.persistence.repository import Repository


SCHEMA_PATH = Path(__file__).resolve().parent.parent / "app" / "persistence" / "schema.sql"


@pytest.fixture
def repo(tmp_path) -> Repository:
    r = Repository(tmp_path / "t.sqlite")
    r.init_schema(SCHEMA_PATH)
    # 必须先插入一张 drawing 满足外键
    r.insert_drawing(Drawing(id="d1", image_path="images/x.png", width_px=100, height_px=100))
    return r


def test_pipeline_roundtrip(repo: Repository):
    pl = Pipeline.from_polyline("d1", "P1", [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)])
    repo.replace_pipelines("d1", [pl])
    [back] = repo.get_pipelines("d1")
    assert back.name == "P1"
    assert back.points_px == [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]
    assert back.state == ReviewState.AUTO


def test_failure_point_roundtrip(repo: Repository):
    fp = FailurePoint(
        id=new_id("fp_"),
        drawing_id="d1",
        code="F1",
        raw_px=(50.5, 60.5),
        projected_px=(50.5, 0.0),
        pipeline_id="pip_x",
        offset_px=60.5,
        marker_type="red_star",
        confidence=0.9,
    )
    repo.replace_failure_points("d1", [fp])
    [back] = repo.get_failure_points("d1")
    assert back.code == "F1"
    assert back.raw_px == (50.5, 60.5)
    assert back.projected_px == (50.5, 0.0)
    assert back.offset_px == 60.5


def test_measure_result_roundtrip(repo: Repository):
    mr = MeasureResult(
        id=new_id("mr_"),
        drawing_id="d1",
        from_code="F1",
        to_code="F2",
        pipeline_name="P1",
        straight_px=80.0,
        along_pipe_px=85.0,
        straight_m=160.0,
        along_pipe_m=170.0,
        basis="auto",
        confidence=0.9,
        need_review=False,
    )
    repo.replace_measure_results("d1", [mr])
    [back] = repo.get_measure_results("d1")
    assert (back.from_code, back.to_code) == ("F1", "F2")
    assert back.along_pipe_m == 170.0
    assert back.need_review is False


def test_replace_clears_previous(repo: Repository):
    p1 = Pipeline.from_polyline("d1", "P1", [(0.0, 0.0), (5.0, 0.0)])
    repo.replace_pipelines("d1", [p1])
    repo.replace_pipelines("d1", [])
    assert repo.get_pipelines("d1") == []
