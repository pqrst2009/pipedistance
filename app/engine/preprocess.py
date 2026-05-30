"""图像预处理（引擎层，无 Qt 依赖）。

S1 仅实现裁剪与基础增强，后续 Sprint 扩展颜色分层 / 线条增强 / 去噪。
"""
from __future__ import annotations

import cv2
import numpy as np


def crop(image: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = (int(v) for v in rect)
    x = max(0, x)
    y = max(0, y)
    return image[y:y + h, x:x + w].copy()


def to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def binarize(gray: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10
    )


def enhance_for_ocr(image: np.ndarray) -> np.ndarray:
    """轻度增强：去噪 + 锐化，提升小字号 OCR 命中率。"""
    den = cv2.fastNlMeansDenoisingColored(image, None, 5, 5, 7, 21)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(den, -1, kernel)
