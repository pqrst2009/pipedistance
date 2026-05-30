"""比例标定：长度标注 → 像素↔米。

思路：
- 一个长度标注 ``120m`` 贴在一段管线旁。把标注 bbox 中心匹配到 **最近的管段**，
  令该段管段像素长度为 ``L_px``，则该样本贡献比例 ``s = 120 / L_px``（m/px）。
- 对每条管线（pipeline）汇总所有贡献的 ``s``，加权中位数 / 平均，得到这条
  管线的 ``scale_m_per_px``。
- 整张图通常一个全局比例就够用，对外也暴露 ``global_scale`` 入口。

边界：
- 若某条管线没有任何匹配上的长度标注，返回 None；上层可决定用全局比例兜底。
- 标注的"最近管段距离"超过 ``max_match_dist_px`` 视为不属于任何管线（图上其它文字）。
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import median

from shapely.geometry import LineString, Point


@dataclass(frozen=True)
class LengthSample:
    """一条标注对应一次比例采样。"""
    label_id: str
    pipeline_index: int     # 命中的管线在输入列表中的下标
    value_m: float
    pipeline_px_len: float
    scale_m_per_px: float
    distance_px: float      # 标注中心到管线的最短距离（用于诊断）


@dataclass(frozen=True)
class PipelineScale:
    pipeline_index: int
    scale_m_per_px: float
    sample_count: int
    samples: list[LengthSample]


def _bbox_center(bbox: tuple) -> tuple[float, float]:
    x, y, w, h = bbox
    return (x + w / 2.0, y + h / 2.0)


def _to_linestring(poly: list[tuple[float, float]]) -> LineString:
    return LineString(poly)


def match_labels_to_pipelines(
    labels: list,
    pipelines: list[list[tuple[float, float]]],
    max_match_dist_px: float = 50.0,
) -> list[LengthSample]:
    """把长度标注按最近距离匹配到管线。

    ``labels`` 元素需有 ``id`` / ``value_m`` / ``bbox`` 三个属性
    （兼容 ``app.domain.models.LengthLabel`` 与轻量测试 stub）。
    """
    lines = [_to_linestring(p) for p in pipelines]
    samples: list[LengthSample] = []
    for lb in labels:
        cx, cy = _bbox_center(lb.bbox)
        center = Point(cx, cy)
        best_i = -1
        best_d = float("inf")
        for i, line in enumerate(lines):
            d = line.distance(center)
            if d < best_d:
                best_d, best_i = d, i
        if best_i < 0 or best_d > max_match_dist_px:
            continue
        line = lines[best_i]
        if line.length <= 0:
            continue
        scale = lb.value_m / line.length
        samples.append(
            LengthSample(
                label_id=lb.id,
                pipeline_index=best_i,
                value_m=lb.value_m,
                pipeline_px_len=line.length,
                scale_m_per_px=scale,
                distance_px=best_d,
            )
        )
    return samples


def per_pipeline_scale(samples: list[LengthSample], pipeline_count: int) -> list[PipelineScale | None]:
    """每条管线一个比例：多样本取中位数（鲁棒抗异常值）。"""
    buckets: dict[int, list[LengthSample]] = {}
    for s in samples:
        buckets.setdefault(s.pipeline_index, []).append(s)
    out: list[PipelineScale | None] = []
    for i in range(pipeline_count):
        bucket = buckets.get(i, [])
        if not bucket:
            out.append(None)
            continue
        med = median(s.scale_m_per_px for s in bucket)
        out.append(
            PipelineScale(
                pipeline_index=i,
                scale_m_per_px=med,
                sample_count=len(bucket),
                samples=bucket,
            )
        )
    return out


def global_scale(samples: list[LengthSample]) -> float | None:
    """全局比例：所有样本的中位数；样本太少时返回 None。"""
    if not samples:
        return None
    return median(s.scale_m_per_px for s in samples)
