"""图像 I/O 辅助（UI 侧）。

Windows 上 cv2.imread 不支持非 ASCII 路径，统一用 np.fromfile + imdecode。
"""
from __future__ import annotations

import numpy as np
import cv2


def imread_unicode(path: str) -> np.ndarray:
    """读取为 BGR ndarray，支持中文/Unicode 路径。"""
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"无法解码图像：{path}")
    return img


def imwrite_unicode(path: str, bgr: np.ndarray, ext: str = ".png") -> None:
    ok, buf = cv2.imencode(ext, bgr)
    if not ok:
        raise IOError(f"无法编码图像：{path}")
    buf.tofile(path)


def bgr_to_qpixmap(bgr: np.ndarray):
    """转换为 QPixmap（延迟导入 Qt，避免引擎层依赖）。"""
    from PySide6.QtGui import QImage, QPixmap

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)
