"""OCR 引擎封装（RapidOCR / ONNX）。

设计目标：
1. 兼容多版本 RapidOCR：较新的统一包 `rapidocr`（`from rapidocr import RapidOCR`）
   与较旧的 `rapidocr_onnxruntime`（返回 (result, elapse)）。
2. 引擎未安装时不崩溃：available=False，调用 recognize 抛出可读错误，
   由 UI 友好提示用户安装。
3. 归一化不同返回结构为统一的 OcrItem 列表，供上层使用。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class OcrItem:
    text: str
    bbox: tuple[int, int, int, int]   # 轴对齐 (x, y, w, h)
    confidence: float
    quad: list = field(default_factory=list)  # 原始四点框


class OcrEngine:
    def __init__(self, lazy: bool = True):
        self._engine = None
        self._api = None
        self.available = False
        self._error = ""
        self._loaded = False
        if not lazy:
            self._load()

    # ---- 生命周期 ----
    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        # 优先较新的统一包
        try:
            from rapidocr import RapidOCR  # type: ignore

            self._engine = RapidOCR()
            self._api = "rapidocr"
            self.available = True
            return
        except Exception as e:  # noqa: BLE001
            self._error = f"rapidocr 不可用: {e}"
        # 回退较旧的 onnxruntime 版
        try:
            from rapidocr_onnxruntime import RapidOCR  # type: ignore

            self._engine = RapidOCR()
            self._api = "rapidocr_onnxruntime"
            self.available = True
            return
        except Exception as e:  # noqa: BLE001
            self._error += f" | rapidocr_onnxruntime 不可用: {e}"
            self.available = False

    def ensure_loaded(self) -> None:
        if not self._loaded:
            self._load()

    @property
    def backend(self) -> str:
        return self._api or "none"

    @property
    def last_error(self) -> str:
        return self._error

    # ---- 推理 ----
    def recognize(self, image_bgr: np.ndarray) -> list[OcrItem]:
        self.ensure_loaded()
        if not self.available:
            raise RuntimeError(
                "未检测到可用的 OCR 引擎，请先安装：pip install rapidocr\n详情：" + self._error
            )
        raw = self._engine(image_bgr)
        return self._normalize(raw)

    # ---- 返回结构归一化 ----
    def _normalize(self, raw) -> list[OcrItem]:
        items: list[OcrItem] = []
        result = raw

        # 旧版返回 (result, elapse)
        if isinstance(raw, tuple) and raw:
            result = raw[0]

        if result is None:
            return items

        # 新版对象：具有 .txts / .boxes / .scores
        if hasattr(result, "txts"):
            txts = getattr(result, "txts", None)
            boxes = getattr(result, "boxes", None)
            scores = getattr(result, "scores", None)
            if txts is not None:
                for i, t in enumerate(txts):
                    quad = self._to_list(boxes[i]) if boxes is not None else []
                    sc = float(scores[i]) if scores is not None else 0.0
                    items.append(self._make(t, quad, sc))
                return items

        # 列表型：[[box, text, score], ...]
        try:
            for row in result:
                box, text, score = row[0], row[1], row[2]
                items.append(self._make(text, self._to_list(box), float(score)))
        except Exception:  # noqa: BLE001
            # 未知结构：尽量保底，不抛错
            pass
        return items

    @staticmethod
    def _to_list(quad) -> list:
        if quad is None:
            return []
        try:
            return [[float(p[0]), float(p[1])] for p in quad]
        except Exception:  # noqa: BLE001
            return []

    def _make(self, text, quad: list, score: float) -> OcrItem:
        return OcrItem(text=str(text), bbox=self._quad_to_bbox(quad),
                       confidence=score, quad=quad)

    @staticmethod
    def _quad_to_bbox(quad: list) -> tuple[int, int, int, int]:
        if not quad:
            return (0, 0, 0, 0)
        xs = [p[0] for p in quad]
        ys = [p[1] for p in quad]
        x, y = min(xs), min(ys)
        return (int(x), int(y), int(max(xs) - x), int(max(ys) - y))
