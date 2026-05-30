"""后台线程：长耗时 OCR 不阻塞 UI。"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QThread, Signal

from app.services.extraction_service import ExtractionResult, ExtractionService
from app.services.recognition_service import RecognitionService


class OcrWorker(QThread):
    finished_ok = Signal(list, list)   # (items, labels)
    failed = Signal(str)

    def __init__(self, recog: RecognitionService, image_bgr: np.ndarray,
                 drawing_id: str, crop_rect: tuple | None):
        super().__init__()
        self._recog = recog
        self._image = image_bgr
        self._drawing_id = drawing_id
        self._crop = crop_rect

    def run(self) -> None:
        try:
            items, labels = self._recog.run_ocr(
                self._image, self._drawing_id, self._crop
            )
            self.finished_ok.emit(items, labels)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class ExtractionWorker(QThread):
    """自动提取（管线 / 失效点 / 测距）后台线程。"""

    finished_ok = Signal(object)   # ExtractionResult
    failed = Signal(str)

    def __init__(
        self,
        service: ExtractionService,
        image_bgr: np.ndarray,
        drawing_id: str,
        length_labels: list,
    ):
        super().__init__()
        self._service = service
        self._image = image_bgr
        self._drawing_id = drawing_id
        self._labels = length_labels

    def run(self) -> None:
        try:
            result: ExtractionResult = self._service.run(
                self._image, self._drawing_id, self._labels
            )
            self.finished_ok.emit(result)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"{type(e).__name__}: {e}")
