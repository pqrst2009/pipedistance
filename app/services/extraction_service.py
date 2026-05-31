"""自动提取编排：图像 → 管线 + 失效点 + 标定 + 成对测距。

把引擎层 5 个模块串成一次调用，输出领域模型，便于 UI 直接使用 / 持久化。

调用顺序：
  green_mask → clean → vectorize         → list[Pipeline]
  red_mask   → clean → detect_stars      → list[FailurePoint]
  length_labels + pipelines → calibration → 每条管线的 m/px（中位数）
  points + pipelines + scales → measure   → 投影 + 成对距离

不依赖 Qt；UI 层只需调用 ``ExtractionService.run`` 即可。
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.domain.models import (
    FailurePoint,
    LengthLabel,
    MeasureResult,
    Pipeline,
    ReviewState,
    new_id,
)
from app.engine.calibration import (
    global_scale,
    match_labels_to_pipelines,
    per_pipeline_scale,
)
from app.engine.color_detect import (
    GreenThresholds,
    RedThresholds,
    clean_mask,
    green_mask,
    red_mask,
)
from app.engine.measure import measure_all
from app.engine.pipeline_vectorize import (
    VectorizeParams,
    bridge_through_points,
    vectorize,
)
from app.engine.star_detect import StarDetectParams, detect_stars

# 失效点偏离管线过远（像素）则不参与测距，仅保留 raw 位置供复核
DEFAULT_MAX_OFFSET_PX = 60.0


@dataclass
class ExtractionParams:
    green: GreenThresholds = GreenThresholds()
    red: RedThresholds = RedThresholds()
    green_min_area: int = 50          # 绿掩膜小连通域清理阈值（像素）
    bridge_star_gap_px: float = 40.0  # 跨星缝合：折线端点距星心 ≤ 此值则缝合
    vectorize: VectorizeParams = VectorizeParams()
    star: StarDetectParams = StarDetectParams()
    max_label_match_dist_px: float = 60.0
    max_failure_offset_px: float = DEFAULT_MAX_OFFSET_PX
    failure_review_offset_px: float = 25.0   # 偏离大于此值置 NEED_REVIEW


@dataclass
class ExtractionResult:
    pipelines: list[Pipeline]
    failure_points: list[FailurePoint]
    measures: list[MeasureResult]
    global_scale_m_per_px: float | None
    per_pipeline_scale: list[float | None]


class ExtractionService:
    """一次自动提取的编排入口；无状态，可在 worker 线程里反复调用。"""

    def __init__(self, params: ExtractionParams | None = None):
        self.params = params or ExtractionParams()

    def run(
        self,
        image_bgr: np.ndarray,
        drawing_id: str,
        length_labels: list[LengthLabel],
    ) -> ExtractionResult:
        p = self.params

        # 1) 颜色掩膜与失效点（先检测星，缝合时要用星心做中转点）
        gmask_raw = green_mask(image_bgr, p.green)
        rmask = red_mask(image_bgr, p.red)
        gmask = clean_mask(gmask_raw, open_ksize=3, close_ksize=3, min_area=p.green_min_area)
        # 红 mask 仅去单像素噪声，不做 open——open 会蚀掉爆炸图标的 spike 和小图标。
        # detect_stars 内部已经做 close + fill_outline 的形状重建，且自带 min_area。
        rmask_clean = clean_mask(rmask, open_ksize=0, close_ksize=0, min_area=0)
        stars = detect_stars(rmask_clean, p.star)
        star_centers = [(s.x, s.y) for s in stars]

        # 2) 矢量化 + 跨星缝合：红星把绿管打断后，用星心把附近端点缝回去
        polylines = vectorize(gmask, p.vectorize)
        if p.bridge_star_gap_px > 0 and star_centers:
            polylines = bridge_through_points(
                polylines, star_centers, max_gap_px=p.bridge_star_gap_px
            )
        pipelines = [
            Pipeline.from_polyline(drawing_id, f"P{i + 1}", poly)
            for i, poly in enumerate(polylines)
        ]
        # 编号按检测顺序 F1..Fn
        star_points = star_centers

        # 3) 标定（用 OCR 的长度标注）
        samples = match_labels_to_pipelines(
            length_labels, polylines, max_match_dist_px=p.max_label_match_dist_px
        )
        per_pipe = per_pipeline_scale(samples, pipeline_count=len(polylines))
        scales = [ps.scale_m_per_px if ps else None for ps in per_pipe]
        g_scale = global_scale(samples)

        # 4) 投影 + 成对测距
        proj, pairs = measure_all(
            star_points,
            polylines,
            scales=scales,
            fallback_scale=g_scale,
            max_offset_px=p.max_failure_offset_px,
        )

        # 5) 装配领域模型
        failure_points = self._build_failure_points(
            drawing_id, stars, proj, pipelines
        )
        measures = self._build_measures(drawing_id, failure_points, pairs, pipelines)

        return ExtractionResult(
            pipelines=pipelines,
            failure_points=failure_points,
            measures=measures,
            global_scale_m_per_px=g_scale,
            per_pipeline_scale=scales,
        )

    # ---- 内部 ----
    def _build_failure_points(
        self,
        drawing_id: str,
        stars,
        projections,
        pipelines: list[Pipeline],
    ) -> list[FailurePoint]:
        out: list[FailurePoint] = []
        for i, star in enumerate(stars):
            pj = projections[i]
            pip_id = None
            projected = None
            offset = None
            state = ReviewState.AUTO
            if pj is not None:
                pip_id = pipelines[pj.pipeline_index].id
                projected = pj.proj_xy
                offset = pj.offset_px
                if offset > self.params.failure_review_offset_px:
                    state = ReviewState.NEED_REVIEW
            else:
                state = ReviewState.NEED_REVIEW
            out.append(
                FailurePoint(
                    id=new_id("fp_"),
                    drawing_id=drawing_id,
                    code=f"F{i + 1}",
                    raw_px=(star.x, star.y),
                    projected_px=projected,
                    pipeline_id=pip_id,
                    offset_px=offset,
                    marker_type="red_star",
                    confidence=0.9,
                    state=state,
                )
            )
        return out

    def _build_measures(
        self,
        drawing_id: str,
        failure_points: list[FailurePoint],
        pairs,
        pipelines: list[Pipeline],
    ) -> list[MeasureResult]:
        out: list[MeasureResult] = []
        for pair in pairs:
            fp_from = failure_points[pair.from_index]
            fp_to = failure_points[pair.to_index]
            same_pipe = pair.pipeline_index >= 0
            need_review = (
                fp_from.state != ReviewState.AUTO
                or fp_to.state != ReviewState.AUTO
                or pair.straight_m is None
            )
            pipe_name = (
                pipelines[pair.pipeline_index].name if same_pipe else "—（跨管线）"
            )
            out.append(
                MeasureResult(
                    id=new_id("mr_"),
                    drawing_id=drawing_id,
                    from_code=fp_from.code,
                    to_code=fp_to.code,
                    pipeline_name=pipe_name,
                    straight_px=pair.straight_px,
                    along_pipe_px=pair.along_pipe_px,
                    straight_m=pair.straight_m,
                    along_pipe_m=pair.along_pipe_m,
                    basis="auto" if same_pipe else "cross_pipeline",
                    confidence=0.9,
                    need_review=need_review,
                )
            )
        return out
