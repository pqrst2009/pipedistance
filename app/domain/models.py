"""领域模型（dataclass）。

约定：
- 坐标统一使用图像像素坐标系，原点在左上角，单位 px。
- 实际工程长度统一使用米 (m)。
- 所有"识别"结果先以候选 + 置信度形式存在，经人工复核置为 CONFIRMED。
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum


def new_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:12]}"


class ReviewState(str, Enum):
    AUTO = "auto"              # 自动生成，未复核
    NEED_REVIEW = "need_review"  # 低置信度，需人工确认
    CONFIRMED = "confirmed"    # 已人工确认


@dataclass
class Drawing:
    id: str
    image_path: str               # 项目内相对路径，例如 images/dwg_xxx.png
    width_px: int
    height_px: int
    crop_rect: tuple | None = None  # (x, y, w, h) 有效区域，None 表示整图


@dataclass
class PolylineSegment:
    """管线中心线的一段折线。"""
    points_px: list[tuple[float, float]] = field(default_factory=list)
    pixel_length: float = 0.0


@dataclass
class Pipeline:
    id: str
    drawing_id: str
    name: str = ""
    points_px: list[tuple[float, float]] = field(default_factory=list)  # 折线顶点
    segments: list[PolylineSegment] = field(default_factory=list)
    geometry_wkt: str = ""        # LINESTRING(...)，便于 Shapely 计算
    state: ReviewState = ReviewState.AUTO

    @classmethod
    def from_polyline(cls, drawing_id: str, name: str, points: list[tuple[float, float]]) -> "Pipeline":
        wkt = "LINESTRING(" + ", ".join(f"{x} {y}" for x, y in points) + ")" if len(points) >= 2 else ""
        return cls(
            id=new_id("pip_"),
            drawing_id=drawing_id,
            name=name,
            points_px=[(float(x), float(y)) for x, y in points],
            geometry_wkt=wkt,
        )


@dataclass
class LengthLabel:
    """OCR 识别出的长度标注。"""
    id: str
    drawing_id: str
    text: str                     # 原始文本，如 "L=120m"
    value_m: float                # 归一化为米
    bbox: tuple                   # (x, y, w, h)
    confidence: float
    matched_segment_id: str | None = None
    state: ReviewState = ReviewState.AUTO


@dataclass
class Calibration:
    """像素 ↔ 实际长度换算。"""
    id: str
    pipeline_id: str
    pixel_length: float
    real_length_m: float
    scale_m_per_px: float
    source: str                   # "管段L=120m" / "多段加权"
    confidence: float


@dataclass
class FailurePoint:
    id: str
    drawing_id: str
    code: str                     # F01 / 泄漏点1
    raw_px: tuple                 # 原始标记位置 (x, y)
    projected_px: tuple | None = None
    pipeline_id: str | None = None
    offset_px: float | None = None
    marker_type: str = ""         # red_circle / red_cross / yellow / arrow / manual
    confidence: float = 1.0
    state: ReviewState = ReviewState.AUTO


@dataclass
class MeasureResult:
    id: str
    drawing_id: str
    from_code: str
    to_code: str
    pipeline_name: str
    straight_px: float
    along_pipe_px: float
    straight_m: float | None
    along_pipe_m: float | None
    basis: str = ""
    confidence: float = 1.0
    need_review: bool = False
