"""图纸画布：QGraphicsView 子类。

特性：
- 滚轮缩放、空格/中键拖拽平移。
- 裁剪模式：拖拽出矩形选区，释放后发出 cropSelected(QRectF)，
  坐标即图像像素坐标（pixmap 置于场景原点，无额外变换）。
"""
from __future__ import annotations

from PySide6.QtCore import Qt, QRectF, QPointF, Signal
from PySide6.QtGui import QBrush, QPainter, QPen, QColor, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
)


class DrawingView(QGraphicsView):
    cropSelected = Signal(QRectF)            # 图像像素坐标系下的矩形
    calibrationCompleted = Signal(QPointF, QPointF)  # 两点都落地后发：(p1, p2)
    addFailurePointRequested = Signal(QPointF)  # 手动添加失效点：图像像素坐标

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        # 关闭内置滚动条：避免在 macOS / 某些平台下条出现在画布里把图"分栏"。
        # 缩放走 wheelEvent，平移走 ScrollHandDrag，已经够用。
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.NoFrame)
        # macOS overlay scrollbar 即使设了 AlwaysOff 也会留 1 像素轨道槽，
        # 直接通过 stylesheet 把两条 scrollbar 物理宽度压成 0。
        self.setStyleSheet(
            "QGraphicsView { border: 0px; padding: 0px; margin: 0px; } "
            "QGraphicsView > QScrollBar { width: 0px; height: 0px; }"
        )
        self.horizontalScrollBar().setFixedHeight(0)
        self.verticalScrollBar().setFixedWidth(0)

        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._crop_mode = False
        self._calibrate_mode = False
        self._add_failure_mode = False
        self._rubber: QGraphicsRectItem | None = None
        self._origin: QPointF | None = None
        # 标定中间状态：第一点坐标 + 视觉指示
        self._calib_p1: QPointF | None = None
        self._calib_dot: QGraphicsEllipseItem | None = None
        self._calib_rubber: QGraphicsLineItem | None = None
        # ScrollHandDrag 会抢走 left-press 启动平移，落在 movable item 上时
        # 临时切到 NoDrag 让 item 接管；mouseRelease 时恢复。
        self._dragmode_before_item: QGraphicsView.DragMode | None = None

    # ---- 图像 ----
    def load_pixmap(self, pixmap: QPixmap) -> None:
        self._scene.clear()
        self._rubber = None
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        self.resetTransform()
        self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)

    def clear(self) -> None:
        self._scene.clear()
        self._pixmap_item = None
        self._rubber = None

    def has_image(self) -> bool:
        return self._pixmap_item is not None

    # ---- 裁剪模式 ----
    def set_crop_mode(self, on: bool) -> None:
        self._crop_mode = on
        self.setDragMode(QGraphicsView.NoDrag if on else QGraphicsView.ScrollHandDrag)
        self.setCursor(Qt.CrossCursor if on else Qt.ArrowCursor)

    # ---- 手动添加失效点：单次点击落星 ----
    def set_add_failure_mode(self, on: bool) -> None:
        self._add_failure_mode = on
        self.setDragMode(QGraphicsView.NoDrag if on else QGraphicsView.ScrollHandDrag)
        self.setCursor(Qt.CrossCursor if on else Qt.ArrowCursor)

    # ---- 手动标定模式：两次点击 + 橡皮筋预览 ----
    def set_calibrate_mode(self, on: bool) -> None:
        self._calibrate_mode = on
        self.setDragMode(QGraphicsView.NoDrag if on else QGraphicsView.ScrollHandDrag)
        self.setCursor(Qt.CrossCursor if on else Qt.ArrowCursor)
        self.setMouseTracking(on)
        if not on:
            self._reset_calib_preview()

    def draw_crop_overlay(self, rect: tuple | None) -> None:
        """加载已保存的裁剪区域作为可视提示。"""
        if rect is None or self._pixmap_item is None:
            return
        x, y, w, h = rect
        self._ensure_rubber()
        self._rubber.setRect(QRectF(x, y, w, h))

    # ---- 缩放 ----
    def wheelEvent(self, event) -> None:
        if self._pixmap_item is None:
            return
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)

    # ---- 裁剪交互 ----
    def mousePressEvent(self, event) -> None:
        if self._add_failure_mode and event.button() == Qt.LeftButton and self._pixmap_item:
            p = self.mapToScene(event.position().toPoint())
            # 只在图像范围内才发信号，避免在画布空白区点击造成误添加
            if self._scene.sceneRect().contains(p):
                self.addFailurePointRequested.emit(p)
            return
        if self._calibrate_mode and event.button() == Qt.LeftButton and self._pixmap_item:
            p = self.mapToScene(event.position().toPoint())
            if self._calib_p1 is None:
                # 第一点落地：显示黑色端点，准备拉橡皮筋
                self._calib_p1 = p
                self._calib_dot = self._make_endpoint_marker(p)
                self._scene.addItem(self._calib_dot)
            else:
                # 第二点落地：清掉预览，发出完成信号
                p1 = self._calib_p1
                self._reset_calib_preview()
                self.calibrationCompleted.emit(p1, p)
            return
        if self._crop_mode and event.button() == Qt.LeftButton and self._pixmap_item:
            self._origin = self.mapToScene(event.position().toPoint())
            self._ensure_rubber()
            self._rubber.setRect(QRectF(self._origin, self._origin))
            return
        # 普通模式下，若 left-press 落在 movable item（失效点 / 折线顶点）上，
        # 临时切 NoDrag 让 item 接管拖动；否则保持 ScrollHandDrag 平移。
        if event.button() == Qt.LeftButton and self.dragMode() == QGraphicsView.ScrollHandDrag:
            item = self.itemAt(event.position().toPoint())
            if item is not None and (item.flags() & QGraphicsItem.ItemIsMovable):
                self._dragmode_before_item = self.dragMode()
                self.setDragMode(QGraphicsView.NoDrag)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._calibrate_mode and self._calib_p1 is not None:
            cur = self.mapToScene(event.position().toPoint())
            self._update_calib_rubber(cur)
            return
        if self._crop_mode and self._origin is not None and self._rubber is not None:
            cur = self.mapToScene(event.position().toPoint())
            self._rubber.setRect(QRectF(self._origin, cur).normalized())
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._crop_mode and self._origin is not None and self._rubber is not None:
            rect = self._rubber.rect()
            self._origin = None
            # 约束到图像范围内
            bounds = self._scene.sceneRect()
            rect = rect.intersected(bounds)
            if rect.width() > 3 and rect.height() > 3:
                self.cropSelected.emit(rect)
            # 仍要恢复可能的临时拖动模式
            if self._dragmode_before_item is not None:
                self.setDragMode(self._dragmode_before_item)
                self._dragmode_before_item = None
            return
        super().mouseReleaseEvent(event)
        # 普通模式下 press 时为 item 临时切了 NoDrag，在这里恢复
        if self._dragmode_before_item is not None:
            self.setDragMode(self._dragmode_before_item)
            self._dragmode_before_item = None

    def _ensure_rubber(self) -> None:
        if self._rubber is None:
            self._rubber = QGraphicsRectItem()
            pen = QPen(QColor(0, 150, 255), 0)  # cosmetic pen，缩放下保持线宽
            pen.setCosmetic(True)
            self._rubber.setPen(pen)
            self._rubber.setBrush(QColor(0, 150, 255, 40))
            self._scene.addItem(self._rubber)

    # ---- 标定预览（仅交互期间的临时图元）----
    def _make_endpoint_marker(self, pos: QPointF) -> QGraphicsEllipseItem:
        r = 5
        dot = QGraphicsEllipseItem(-r, -r, 2 * r, 2 * r)
        dot.setBrush(QBrush(QColor(0, 0, 0)))
        pen = QPen(QColor(0, 0, 0), 0)
        pen.setCosmetic(True)
        dot.setPen(pen)
        dot.setFlag(QGraphicsItem.ItemIgnoresTransformations)
        dot.setZValue(80)
        dot.setPos(pos)
        return dot

    def _update_calib_rubber(self, cur: QPointF) -> None:
        if self._calib_p1 is None:
            return
        if self._calib_rubber is None:
            line = QGraphicsLineItem()
            pen = QPen(QColor(0, 0, 0), 2)
            pen.setCosmetic(True)
            pen.setStyle(Qt.DashLine)
            line.setPen(pen)
            line.setZValue(79)
            self._scene.addItem(line)
            self._calib_rubber = line
        self._calib_rubber.setLine(
            self._calib_p1.x(), self._calib_p1.y(), cur.x(), cur.y()
        )

    def _reset_calib_preview(self) -> None:
        self._calib_p1 = None
        for item in (self._calib_dot, self._calib_rubber):
            if item is not None and item.scene() is self._scene:
                self._scene.removeItem(item)
        self._calib_dot = None
        self._calib_rubber = None
