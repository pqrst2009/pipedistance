"""画布叠加层：管线折线 / 失效点星标，支持手工拖动修正。

设计：
- ``OverlayLayer`` 持有当前 Pipeline / FailurePoint 模型副本，作为 QObject 暴露信号
  ``model_changed``。MainWindow 监听后重新计算成对距离并刷新右侧表。
- 所有图形项轻量化、可选中、可拖动；拖动结束自动写回模型。
- 不依赖 Qt 之外的 app 包，便于复用。
"""
from __future__ import annotations

import math

from PySide6.QtCore import QObject, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsPathItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
)

CALIB_COLOR = QColor(0, 0, 0)

PIPELINE_COLOR = QColor(0, 170, 80)
PIPELINE_WIDTH = 2.5
HANDLE_RADIUS = 5.0
HANDLE_BRUSH = QColor(0, 170, 80, 220)
HANDLE_PEN = QColor(20, 80, 40)

STAR_COLOR = QColor(220, 30, 40)
STAR_OUTER = 12.0
STAR_INNER = 5.0
LABEL_COLOR = QColor(0, 0, 0)               # 醒目的黑色
LABEL_BG_COLOR = QColor(255, 255, 255, 230)  # 半透明白底，避免压住背景图
LABEL_BORDER_COLOR = QColor(0, 0, 0, 110)
# 文字相对星心的偏移（星右上方），数值在屏幕像素上恒定（ItemIgnoresTransformations）
LABEL_OFFSET_X = STAR_OUTER + 8
LABEL_OFFSET_Y = -STAR_OUTER - 18
LABEL_PAD = 3


def _safe_remove_from_scene(item, scene: QGraphicsScene) -> None:
    """从 scene 移除 item，已被 Qt 删的 C++ wrapper 静默吞掉。"""
    try:
        if item.scene() is scene:
            scene.removeItem(item)
    except RuntimeError:
        # "Internal C++ object already deleted" —— scene.clear() 已经吞了它
        pass


def _build_star_path(outer: float = STAR_OUTER, inner: float = STAR_INNER) -> QPainterPath:
    pts = []
    for k in range(10):
        a = -math.pi / 2 + k * math.pi / 5
        r = outer if k % 2 == 0 else inner
        pts.append(QPointF(r * math.cos(a), r * math.sin(a)))
    path = QPainterPath()
    path.addPolygon(QPolygonF(pts))
    path.closeSubpath()
    return path


class _VertexHandle(QGraphicsEllipseItem):
    """单个折线顶点的可拖动控制点。"""

    def __init__(self, x: float, y: float, layer: "OverlayLayer",
                 pipeline_id: str, vertex_index: int):
        r = HANDLE_RADIUS
        super().__init__(-r, -r, 2 * r, 2 * r)
        # itemChange 在 setPos / setFlags 期间可能被回调，必须先设好属性
        self._layer = layer
        self._pipeline_id = pipeline_id
        self._vertex_index = vertex_index
        self.setBrush(QBrush(HANDLE_BRUSH))
        pen = QPen(HANDLE_PEN, 0)
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setFlags(
            QGraphicsItem.ItemIsMovable
            | QGraphicsItem.ItemIsSelectable
            | QGraphicsItem.ItemSendsGeometryChanges
            | QGraphicsItem.ItemIgnoresTransformations
        )
        self.setZValue(20)
        self.setPos(x, y)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionHasChanged:
            new_pos = self.pos()
            self._layer._on_vertex_moved(
                self._pipeline_id, self._vertex_index, new_pos.x(), new_pos.y()
            )
        return super().itemChange(change, value)


class _PipelinePath(QGraphicsPathItem):
    """折线主体（不直接捕获拖动）。"""

    def __init__(self, points: list[tuple[float, float]]):
        super().__init__()
        pen = QPen(PIPELINE_COLOR, PIPELINE_WIDTH)
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setZValue(10)
        self.set_points(points)

    def set_points(self, points: list[tuple[float, float]]):
        path = QPainterPath()
        if points:
            path.moveTo(points[0][0], points[0][1])
            for x, y in points[1:]:
                path.lineTo(x, y)
        self.setPath(path)


class _FailurePointItem(QGraphicsPathItem):
    """红色五角星 + 编号标签，整体可拖动。"""

    def __init__(self, code: str, x: float, y: float, layer: "OverlayLayer", fp_id: str):
        super().__init__(_build_star_path())
        # itemChange 在 setPos / setFlags 期间可能被回调，必须先设好属性
        self._layer = layer
        self._fp_id = fp_id
        self.setBrush(QBrush(STAR_COLOR))
        pen = QPen(QColor(120, 0, 0), 0)
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setFlags(
            QGraphicsItem.ItemIsMovable
            | QGraphicsItem.ItemIsSelectable
            | QGraphicsItem.ItemSendsGeometryChanges
            | QGraphicsItem.ItemIgnoresTransformations
        )
        self.setZValue(30)
        self.setPos(x, y)

        # 标签文字（黑色加粗），放在星的右上方
        self._label = QGraphicsSimpleTextItem(code, parent=self)
        f = QFont()
        f.setBold(True)
        f.setPointSize(11)
        self._label.setFont(f)
        self._label.setBrush(QBrush(LABEL_COLOR))
        self._label.setPos(LABEL_OFFSET_X, LABEL_OFFSET_Y)

        # 白色半透明底（贴在文字后面），保证压在地图任何颜色上都清晰
        text_rect = self._label.boundingRect()
        bg_rect = QRectF(
            LABEL_OFFSET_X - LABEL_PAD,
            LABEL_OFFSET_Y - LABEL_PAD,
            text_rect.width() + 2 * LABEL_PAD,
            text_rect.height() + 2 * LABEL_PAD,
        )
        self._label_bg = QGraphicsRectItem(bg_rect, parent=self)
        self._label_bg.setBrush(QBrush(LABEL_BG_COLOR))
        bp = QPen(LABEL_BORDER_COLOR, 0)
        bp.setCosmetic(True)
        self._label_bg.setPen(bp)
        # 把底色画在文字之下
        self._label_bg.setZValue(self._label.zValue() - 1)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionHasChanged:
            p = self.pos()
            self._layer._on_failure_moved(self._fp_id, p.x(), p.y())
        return super().itemChange(change, value)


class CalibrationRuler:
    """画布上的永久标定标尺：黑色实线 + 两个黑色端点 + 中点距离文字。"""

    def __init__(self, scene: QGraphicsScene, p1: QPointF, p2: QPointF, label_text: str):
        self._scene = scene
        self._items: list[QGraphicsItem] = []

        pen = QPen(CALIB_COLOR, 2)
        pen.setCosmetic(True)
        line = QGraphicsLineItem(p1.x(), p1.y(), p2.x(), p2.y())
        line.setPen(pen)
        line.setZValue(60)
        scene.addItem(line)
        self._items.append(line)

        for p in (p1, p2):
            r = 5
            dot = QGraphicsEllipseItem(-r, -r, 2 * r, 2 * r)
            dot.setBrush(QBrush(CALIB_COLOR))
            ep = QPen(CALIB_COLOR, 0)
            ep.setCosmetic(True)
            dot.setPen(ep)
            dot.setFlag(QGraphicsItem.ItemIgnoresTransformations)
            dot.setZValue(61)
            dot.setPos(p)
            scene.addItem(dot)
            self._items.append(dot)

        label = QGraphicsSimpleTextItem(label_text)
        f = QFont()
        f.setBold(True)
        f.setPointSize(11)
        label.setFont(f)
        label.setBrush(QBrush(CALIB_COLOR))
        # 中点附近 + 略微上偏，避免压住直线
        mx = (p1.x() + p2.x()) / 2
        my = (p1.y() + p2.y()) / 2
        label.setPos(mx + 6, my - 18)
        label.setFlag(QGraphicsItem.ItemIgnoresTransformations)
        label.setZValue(62)
        scene.addItem(label)
        self._items.append(label)

    def remove(self) -> None:
        for item in self._items:
            _safe_remove_from_scene(item, self._scene)
        self._items.clear()


class OverlayLayer(QObject):
    """叠加层模型 + 视图同步管理器。

    持有 Pipeline / FailurePoint 当前状态（仅 UI 副本），通过信号通知 MainWindow
    重新计算成对测距。
    """

    model_changed = Signal()

    def __init__(self, scene: QGraphicsScene):
        super().__init__()
        self._scene = scene
        self._pipeline_paths: dict[str, _PipelinePath] = {}
        self._pipeline_handles: dict[str, list[_VertexHandle]] = {}
        self._failure_items: dict[str, _FailurePointItem] = {}
        # 模型副本（外部用 set_* 接口写入）
        self._pipelines: list = []
        self._failure_points: list = []

    # ---- 写入模型 / 重建图形 ----
    def set_pipelines(self, pipelines: list) -> None:
        self._clear_pipelines()
        self._pipelines = list(pipelines)
        for pl in self._pipelines:
            path = _PipelinePath(pl.points_px)
            self._scene.addItem(path)
            self._pipeline_paths[pl.id] = path
            handles: list[_VertexHandle] = []
            for idx, (x, y) in enumerate(pl.points_px):
                h = _VertexHandle(x, y, self, pl.id, idx)
                self._scene.addItem(h)
                handles.append(h)
            self._pipeline_handles[pl.id] = handles

    def set_failure_points(self, points: list) -> None:
        self._clear_failure_points()
        self._failure_points = list(points)
        for fp in self._failure_points:
            x, y = fp.raw_px
            item = _FailurePointItem(fp.code or "?", x, y, self, fp.id)
            self._scene.addItem(item)
            self._failure_items[fp.id] = item

    def clear(self) -> None:
        self._clear_pipelines()
        self._clear_failure_points()
        self._pipelines = []
        self._failure_points = []

    # ---- 只读访问 ----
    @property
    def pipelines(self) -> list:
        return self._pipelines

    @property
    def failure_points(self) -> list:
        return self._failure_points

    # ---- 内部 ----
    def _clear_pipelines(self) -> None:
        # 上游可能已经 scene.clear() —— Qt 把 C++ 对象删了但 Python wrapper 还在，
        # 任何方法调用都会 RuntimeError("Internal C++ object already deleted")。
        # 这里只清字典，不再去碰可能已悬挂的 QGraphicsItem。
        for path in self._pipeline_paths.values():
            _safe_remove_from_scene(path, self._scene)
        for handles in self._pipeline_handles.values():
            for h in handles:
                _safe_remove_from_scene(h, self._scene)
        self._pipeline_paths.clear()
        self._pipeline_handles.clear()

    def _clear_failure_points(self) -> None:
        for item in self._failure_items.values():
            _safe_remove_from_scene(item, self._scene)
        self._failure_items.clear()

    def _on_vertex_moved(self, pipeline_id: str, vertex_index: int, x: float, y: float) -> None:
        pl = next((p for p in self._pipelines if p.id == pipeline_id), None)
        if pl is None or vertex_index >= len(pl.points_px):
            return
        pts = list(pl.points_px)
        pts[vertex_index] = (float(x), float(y))
        pl.points_px = pts
        # 同步几何
        pl.geometry_wkt = "LINESTRING(" + ", ".join(f"{a} {b}" for a, b in pts) + ")"
        self._pipeline_paths[pipeline_id].set_points(pts)
        self.model_changed.emit()

    def _on_failure_moved(self, fp_id: str, x: float, y: float) -> None:
        fp = next((f for f in self._failure_points if f.id == fp_id), None)
        if fp is None:
            return
        fp.raw_px = (float(x), float(y))
        self.model_changed.emit()
