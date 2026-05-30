"""OCR 结果面板：展示识别文本与长度标注候选，支持人工查看。

S1 仅做只读展示 + 长度高亮；S2 起开放在表内直接修正文本/数值。
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
)

from app.engine.label_parse import parse_length_text


class OcrPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self._title = QLabel("OCR 结果")
        self._title.setStyleSheet("font-weight: 600;")
        layout.addWidget(self._title)

        self.table = QTableWidget(0, 4, self)
        self.table.setHorizontalHeaderLabels(["文本", "置信度", "长度?", "归一化(m)"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        layout.addWidget(self.table)

    def populate(self, items) -> None:
        self.table.setRowCount(0)
        length_count = 0
        for it in items:
            parsed = parse_length_text(it.text)
            row = self.table.rowCount()
            self.table.insertRow(row)

            self.table.setItem(row, 0, QTableWidgetItem(it.text))
            conf = QTableWidgetItem(f"{it.confidence:.2f}")
            conf.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, 1, conf)

            is_len = parsed is not None
            flag = QTableWidgetItem("是" if is_len else "")
            flag.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, 2, flag)

            val = QTableWidgetItem(f"{parsed.value_m:g}" if is_len else "")
            val.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.table.setItem(row, 3, val)

            if is_len:
                length_count += 1
                color = QColor(225, 245, 225) if it.confidence >= 0.8 else QColor(255, 240, 210)
                for c in range(4):
                    self.table.item(row, c).setBackground(color)

        self._title.setText(f"OCR 结果（共 {len(items)} 项，长度标注 {length_count} 项）")
