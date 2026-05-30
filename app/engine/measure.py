"""失效点 → 管线投影 + 星-星沿管距离矩阵。

公开三个函数：
- ``project_points``: 失效点 → 最近管线 + 投影点 + 沿管参数 + 偏离距离。
- ``pairwise_along_pipe``: 同一管线上的星两两之间沿管线距离（按像素）。
- ``measure_all``: 一体化入口，吃失效点+管线+比例尺，吐成对结果（米）。

约定：
- 像素坐标系一致（原图坐标）。
- ``scales`` 与管线列表一一对应；某条管线没有标定时回退到 ``fallback_scale``，
  若也没有，则该管线上的成对距离仅有像素值，米值为 None。
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from shapely.geometry import LineString, Point


@dataclass(frozen=True)
class PointProjection:
    point_index: int          # 原 failure point 在输入列表里的下标
    raw_xy: tuple[float, float]
    pipeline_index: int       # 投影到的管线下标
    proj_xy: tuple[float, float]
    along_param_px: float     # 投影点距管线起点沿线像素距离
    offset_px: float           # 失效点到管线的偏离距离


@dataclass(frozen=True)
class PairMeasure:
    from_index: int
    to_index: int
    pipeline_index: int
    straight_px: float
    along_pipe_px: float
    straight_m: float | None
    along_pipe_m: float | None


def project_points(
    points: list[tuple[float, float]],
    pipelines: list[list[tuple[float, float]]],
    max_offset_px: float = float("inf"),
) -> list[PointProjection | None]:
    """每个点投影到最近的管线；若距离超 ``max_offset_px``，返回 None。"""
    lines = [LineString(p) for p in pipelines]
    out: list[PointProjection | None] = []
    for idx, (px, py) in enumerate(points):
        pt = Point(px, py)
        best = -1
        best_d = float("inf")
        for i, line in enumerate(lines):
            d = line.distance(pt)
            if d < best_d:
                best_d, best = d, i
        if best < 0 or best_d > max_offset_px:
            out.append(None)
            continue
        line = lines[best]
        param = line.project(pt)
        proj = line.interpolate(param)
        out.append(
            PointProjection(
                point_index=idx,
                raw_xy=(float(px), float(py)),
                pipeline_index=best,
                proj_xy=(float(proj.x), float(proj.y)),
                along_param_px=float(param),
                offset_px=float(best_d),
            )
        )
    return out


def pairwise_all(
    points: list[tuple[float, float]],
    projections: list[PointProjection | None],
    scales: list[float | None] | None = None,
    fallback_scale: float | None = None,
) -> list[PairMeasure]:
    """所有失效点两两配对：

    - 同管线 → 直线 + 沿管距离都给出。
    - 跨管线（或任一未投影）→ 仅直线距离；``along_pipe_*`` 字段为 0/None。
    - 当任一管线/全局比例可用时给米值；否则只给像素值。
    """
    results: list[PairMeasure] = []
    n = len(points)
    for a, b in combinations(range(n), 2):
        pa, pb = points[a], points[b]
        dx, dy = pa[0] - pb[0], pa[1] - pb[1]
        straight_px = (dx * dx + dy * dy) ** 0.5

        proj_a = projections[a] if a < len(projections) else None
        proj_b = projections[b] if b < len(projections) else None
        same_pipe = (
            proj_a is not None
            and proj_b is not None
            and proj_a.pipeline_index == proj_b.pipeline_index
        )

        if same_pipe:
            along_px = abs(proj_a.along_param_px - proj_b.along_param_px)
            scale = _resolve_scale(scales, proj_a.pipeline_index, fallback_scale)
            pipe_idx = proj_a.pipeline_index
        else:
            along_px = 0.0
            scale = fallback_scale
            pipe_idx = -1   # 跨管线占位

        results.append(
            PairMeasure(
                from_index=a,
                to_index=b,
                pipeline_index=pipe_idx,
                straight_px=straight_px,
                along_pipe_px=along_px if same_pipe else 0.0,
                straight_m=straight_px * scale if scale else None,
                along_pipe_m=(along_px * scale) if same_pipe and scale else None,
            )
        )
    results.sort(key=lambda r: (r.pipeline_index, r.from_index, r.to_index))
    return results


def pairwise_along_pipe(
    projections: list[PointProjection | None],
    scales: list[float | None] | None = None,
    fallback_scale: float | None = None,
) -> list[PairMeasure]:
    """组合同一管线上的失效点为两两测距记录。

    ``straight_px`` = 原始失效点之间的欧氏像素距离；
    ``along_pipe_px`` = 两点投影参数之差的绝对值。
    """
    valid = [p for p in projections if p is not None]
    by_pipeline: dict[int, list[PointProjection]] = {}
    for p in valid:
        by_pipeline.setdefault(p.pipeline_index, []).append(p)

    results: list[PairMeasure] = []
    for pipe_idx, group in by_pipeline.items():
        scale = _resolve_scale(scales, pipe_idx, fallback_scale)
        for a, b in combinations(group, 2):
            dx = a.raw_xy[0] - b.raw_xy[0]
            dy = a.raw_xy[1] - b.raw_xy[1]
            straight = (dx * dx + dy * dy) ** 0.5
            along = abs(a.along_param_px - b.along_param_px)
            results.append(
                PairMeasure(
                    from_index=a.point_index,
                    to_index=b.point_index,
                    pipeline_index=pipe_idx,
                    straight_px=straight,
                    along_pipe_px=along,
                    straight_m=straight * scale if scale else None,
                    along_pipe_m=along * scale if scale else None,
                )
            )
    # 输出按 (pipeline, from, to) 排序，便于对比
    results.sort(key=lambda r: (r.pipeline_index, r.from_index, r.to_index))
    return results


def measure_all(
    points: list[tuple[float, float]],
    pipelines: list[list[tuple[float, float]]],
    scales: list[float | None] | None = None,
    fallback_scale: float | None = None,
    max_offset_px: float = float("inf"),
    include_cross_pipeline: bool = True,
) -> tuple[list[PointProjection | None], list[PairMeasure]]:
    """一体化入口：返回 (投影结果, 成对测距)。

    ``include_cross_pipeline=True``（默认）：所有两两配对都给（跨管线只给直线）。
    ``False``：仅同管线配对（旧 S2 行为）。
    """
    proj = project_points(points, pipelines, max_offset_px=max_offset_px)
    if include_cross_pipeline:
        pairs = pairwise_all(points, proj, scales=scales, fallback_scale=fallback_scale)
    else:
        pairs = pairwise_along_pipe(proj, scales=scales, fallback_scale=fallback_scale)
    return proj, pairs


def _resolve_scale(
    scales: list[float | None] | None,
    pipeline_index: int,
    fallback: float | None,
) -> float | None:
    if scales is not None and pipeline_index < len(scales):
        s = scales[pipeline_index]
        if s is not None:
            return s
    return fallback
