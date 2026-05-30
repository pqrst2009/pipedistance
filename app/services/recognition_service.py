"""应用服务层：识别编排。

负责把"引擎 OCR 结果"转成"长度标注候选"并持久化。
不做任何 Qt 调用，便于在测试或批处理中复用。
"""
from __future__ import annotations

import numpy as np

from app.domain.models import LengthLabel, ReviewState, new_id
from app.engine import preprocess
from app.engine.label_parse import parse_length_text
from app.engine.ocr import OcrEngine, OcrItem

# 低于该置信度的 OCR 结果标记为需复核
OCR_REVIEW_THRESHOLD = 0.80


class RecognitionService:
    def __init__(self, ocr_engine: OcrEngine):
        self.ocr = ocr_engine

    def run_ocr(
        self,
        image_bgr: np.ndarray,
        drawing_id: str,
        crop_rect: tuple | None = None,
        enhance: bool = True,
    ) -> tuple[list[OcrItem], list[LengthLabel]]:
        """对（可裁剪的）图纸执行 OCR，返回 (全部文本项, 长度标注候选)。

        注意：crop_rect 下的坐标需要平移回原图坐标系。
        """
        img = image_bgr
        offset_x = offset_y = 0
        if crop_rect:
            img = preprocess.crop(image_bgr, crop_rect)
            offset_x, offset_y = int(crop_rect[0]), int(crop_rect[1])
        if enhance:
            img = preprocess.enhance_for_ocr(img)

        items = self.ocr.recognize(img)

        labels: list[LengthLabel] = []
        for it in items:
            # 平移 bbox 回原图坐标系
            x, y, w, h = it.bbox
            it.bbox = (x + offset_x, y + offset_y, w, h)

            parsed = parse_length_text(it.text)
            if parsed is None:
                continue
            state = (
                ReviewState.AUTO
                if it.confidence >= OCR_REVIEW_THRESHOLD
                else ReviewState.NEED_REVIEW
            )
            labels.append(
                LengthLabel(
                    id=new_id("len_"),
                    drawing_id=drawing_id,
                    text=it.text,
                    value_m=parsed.value_m,
                    bbox=it.bbox,
                    confidence=it.confidence,
                    state=state,
                )
            )
        return items, labels
