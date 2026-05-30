"""测距结果面板：星-星沿管 / 直线距离表。"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class MeasurePanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self._title = QLabel("失效点距离表")
        self._title.setStyleSheet("font-weight: 600;")
        layout.addWidget(self._title)

        self._summary = QLabel("尚未提取。")
        self._summary.setStyleSheet("color: #666;")
        layout.addWidget(self._summary)

        self.table = QTableWidget(0, 5, self)
        self.table.setHorizontalHeaderLabels(
            ["起点", "终点", "管线", "沿管 (m)", "直线 (m)"]
        )
        header = self.table.horizontalHeader()
        for c in range(5):
            header.setSectionResizeMode(c, QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        layout.addWidget(self.table)

    def set_summary(self, text: str) -> None:
        self._summary.setText(text)

    def populate(self, measures: list, scale_m_per_px: float | None) -> None:
        self.table.setRowCount(0)
        for m in measures:
            row = self.table.rowCount()
            self.table.insertRow(row)

            self.table.setItem(row, 0, _center(m.from_code))
            self.table.setItem(row, 1, _center(m.to_code))
            self.table.setItem(row, 2, _center(m.pipeline_name))
            self.table.setItem(row, 3, _num(m.along_pipe_m, m.along_pipe_px))
            self.table.setItem(row, 4, _num(m.straight_m, m.straight_px))

            if m.need_review:
                color = QColor(255, 240, 210)
                for c in range(5):
                    self.table.item(row, c).setBackground(color)

        scale_text = f"{scale_m_per_px:.4f} m/px" if scale_m_per_px else "未标定"
        self._title.setText(f"失效点距离表（共 {len(measures)} 对 · 比例 {scale_text}）")


def _center(text: str) -> QTableWidgetItem:
    it = QTableWidgetItem(str(text or ""))
    it.setTextAlignment(Qt.AlignCenter)
    return it


def _num(meters: float | None, pixels: float) -> QTableWidgetItem:
    if meters is None:
        text = f"{pixels:.1f} px"
    else:
        text = f"{meters:.2f}"
    it = QTableWidgetItem(text)
    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
    return it
