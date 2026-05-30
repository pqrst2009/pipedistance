"""数据持久层：SQLite 读写。

Repository 只负责"对象 <-> 表"的映射，不含业务逻辑。
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path


def _wkt_to_points(wkt: str | None) -> list[tuple[float, float]]:
    """LINESTRING(x1 y1, x2 y2, ...) → [(x1, y1), ...]；空/异常返回 []。"""
    if not wkt:
        return []
    m = re.search(r"LINESTRING\s*\(([^)]+)\)", wkt, re.IGNORECASE)
    if not m:
        return []
    out: list[tuple[float, float]] = []
    for chunk in m.group(1).split(","):
        nums = chunk.strip().split()
        if len(nums) >= 2:
            try:
                out.append((float(nums[0]), float(nums[1])))
            except ValueError:
                continue
    return out

from app.domain.models import (
    Drawing,
    FailurePoint,
    LengthLabel,
    MeasureResult,
    Pipeline,
    ReviewState,
)


class Repository:
    def __init__(self, db_path: str | Path):
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")

    def init_schema(self, schema_sql_path: str | Path) -> None:
        sql = Path(schema_sql_path).read_text(encoding="utf-8")
        self.conn.executescript(sql)
        self.conn.commit()

    def commit(self) -> None:
        self.conn.commit()

    def close(self) -> None:
        try:
            self.conn.commit()
        finally:
            self.conn.close()

    # ----- drawing -----
    def insert_drawing(self, d: Drawing) -> None:
        cx = cy = cw = ch = None
        if d.crop_rect:
            cx, cy, cw, ch = d.crop_rect
        self.conn.execute(
            "INSERT OR REPLACE INTO drawing "
            "(id, image_path, width_px, height_px, crop_x, crop_y, crop_w, crop_h) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (d.id, d.image_path, d.width_px, d.height_px, cx, cy, cw, ch),
        )
        self.conn.commit()

    def update_crop(self, drawing_id: str, rect: tuple | None) -> None:
        if rect:
            x, y, w, h = rect
        else:
            x = y = w = h = None
        self.conn.execute(
            "UPDATE drawing SET crop_x=?, crop_y=?, crop_w=?, crop_h=? WHERE id=?",
            (x, y, w, h, drawing_id),
        )
        self.conn.commit()

    def get_drawings(self) -> list[Drawing]:
        rows = self.conn.execute("SELECT * FROM drawing").fetchall()
        out = []
        for r in rows:
            rect = None
            if r["crop_w"] is not None:
                rect = (r["crop_x"], r["crop_y"], r["crop_w"], r["crop_h"])
            out.append(
                Drawing(
                    id=r["id"],
                    image_path=r["image_path"],
                    width_px=r["width_px"],
                    height_px=r["height_px"],
                    crop_rect=rect,
                )
            )
        return out

    # ----- length_label -----
    def replace_length_labels(self, drawing_id: str, labels: list[LengthLabel]) -> None:
        self.conn.execute("DELETE FROM length_label WHERE drawing_id=?", (drawing_id,))
        for lb in labels:
            x, y, w, h = lb.bbox
            self.conn.execute(
                "INSERT OR REPLACE INTO length_label "
                "(id, drawing_id, text, value_m, bbox_x, bbox_y, bbox_w, bbox_h, "
                " confidence, matched_segment_id, state) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (lb.id, lb.drawing_id, lb.text, lb.value_m, x, y, w, h,
                 lb.confidence, lb.matched_segment_id, lb.state.value),
            )
        self.conn.commit()

    def get_length_labels(self, drawing_id: str) -> list[LengthLabel]:
        rows = self.conn.execute(
            "SELECT * FROM length_label WHERE drawing_id=?", (drawing_id,)
        ).fetchall()
        return [
            LengthLabel(
                id=r["id"],
                drawing_id=r["drawing_id"],
                text=r["text"],
                value_m=r["value_m"],
                bbox=(r["bbox_x"], r["bbox_y"], r["bbox_w"], r["bbox_h"]),
                confidence=r["confidence"],
                matched_segment_id=r["matched_segment_id"],
                state=ReviewState(r["state"]),
            )
            for r in rows
        ]

    # ----- pipeline -----
    def replace_pipelines(self, drawing_id: str, pipelines: list[Pipeline]) -> None:
        self.conn.execute("DELETE FROM pipeline WHERE drawing_id=?", (drawing_id,))
        for p in pipelines:
            self.conn.execute(
                "INSERT OR REPLACE INTO pipeline "
                "(id, drawing_id, name, geometry_wkt, state) VALUES (?,?,?,?,?)",
                (p.id, p.drawing_id, p.name, p.geometry_wkt, p.state.value),
            )
        self.conn.commit()

    def get_pipelines(self, drawing_id: str) -> list[Pipeline]:
        rows = self.conn.execute(
            "SELECT * FROM pipeline WHERE drawing_id=?", (drawing_id,)
        ).fetchall()
        return [
            Pipeline(
                id=r["id"],
                drawing_id=r["drawing_id"],
                name=r["name"] or "",
                points_px=_wkt_to_points(r["geometry_wkt"]),
                geometry_wkt=r["geometry_wkt"] or "",
                state=ReviewState(r["state"]),
            )
            for r in rows
        ]

    # ----- failure_point -----
    def replace_failure_points(self, drawing_id: str, points: list[FailurePoint]) -> None:
        self.conn.execute("DELETE FROM failure_point WHERE drawing_id=?", (drawing_id,))
        for fp in points:
            rx, ry = fp.raw_px
            px, py = fp.projected_px if fp.projected_px else (None, None)
            self.conn.execute(
                "INSERT OR REPLACE INTO failure_point "
                "(id, drawing_id, code, raw_x, raw_y, proj_x, proj_y, "
                " pipeline_id, offset_px, marker_type, confidence, state) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    fp.id, fp.drawing_id, fp.code, rx, ry, px, py,
                    fp.pipeline_id, fp.offset_px, fp.marker_type,
                    fp.confidence, fp.state.value,
                ),
            )
        self.conn.commit()

    def get_failure_points(self, drawing_id: str) -> list[FailurePoint]:
        rows = self.conn.execute(
            "SELECT * FROM failure_point WHERE drawing_id=?", (drawing_id,)
        ).fetchall()
        out: list[FailurePoint] = []
        for r in rows:
            proj = (r["proj_x"], r["proj_y"]) if r["proj_x"] is not None else None
            out.append(
                FailurePoint(
                    id=r["id"],
                    drawing_id=r["drawing_id"],
                    code=r["code"] or "",
                    raw_px=(r["raw_x"], r["raw_y"]),
                    projected_px=proj,
                    pipeline_id=r["pipeline_id"],
                    offset_px=r["offset_px"],
                    marker_type=r["marker_type"] or "",
                    confidence=r["confidence"] or 1.0,
                    state=ReviewState(r["state"]),
                )
            )
        return out

    # ----- measure_result -----
    def replace_measure_results(self, drawing_id: str, results: list[MeasureResult]) -> None:
        self.conn.execute("DELETE FROM measure_result WHERE drawing_id=?", (drawing_id,))
        for m in results:
            self.conn.execute(
                "INSERT OR REPLACE INTO measure_result "
                "(id, drawing_id, from_code, to_code, pipeline_name, "
                " straight_px, along_pipe_px, straight_m, along_pipe_m, "
                " basis, confidence, need_review) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    m.id, m.drawing_id, m.from_code, m.to_code, m.pipeline_name,
                    m.straight_px, m.along_pipe_px, m.straight_m, m.along_pipe_m,
                    m.basis, m.confidence, int(m.need_review),
                ),
            )
        self.conn.commit()

    def get_measure_results(self, drawing_id: str) -> list[MeasureResult]:
        rows = self.conn.execute(
            "SELECT * FROM measure_result WHERE drawing_id=?", (drawing_id,)
        ).fetchall()
        return [
            MeasureResult(
                id=r["id"],
                drawing_id=r["drawing_id"],
                from_code=r["from_code"],
                to_code=r["to_code"],
                pipeline_name=r["pipeline_name"],
                straight_px=r["straight_px"],
                along_pipe_px=r["along_pipe_px"],
                straight_m=r["straight_m"],
                along_pipe_m=r["along_pipe_m"],
                basis=r["basis"] or "",
                confidence=r["confidence"] or 1.0,
                need_review=bool(r["need_review"]),
            )
            for r in rows
        ]

    # ----- review log -----
    def log(self, entity_type: str, entity_id: str, action: str, detail: str = "") -> None:
        from datetime import datetime, timezone

        self.conn.execute(
            "INSERT INTO review_log (ts, entity_type, entity_id, action, detail) "
            "VALUES (?,?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(), entity_type, entity_id, action, detail),
        )
        self.conn.commit()
