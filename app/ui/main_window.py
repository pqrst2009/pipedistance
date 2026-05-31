"""主窗口：连接服务层、画布与面板。

S1 提供的闭环入口：
  新建/打开/保存项目 → 导入图纸 → （可选）裁剪有效区域 → 运行 OCR → 查看长度标注。
后续 Sprint 在此基础上接入：管线绘制、失效点标注、测距、导出。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QMainWindow,
    QFileDialog,
    QInputDialog,
    QMessageBox,
    QDockWidget,
    QToolBar,
    QStatusBar,
)

from app.engine.measure import measure_all
from app.engine.ocr import OcrEngine
from app.services.export_service import export_to_xlsx
from app.services.extraction_service import (
    ExtractionParams,
    ExtractionResult,
    ExtractionService,
)
from app.services.project_service import ProjectService
from app.services.recognition_service import RecognitionService
from app.domain.models import MeasureResult, ReviewState, new_id
from app.ui.canvas.graphics_view import DrawingView
from app.ui.canvas.overlays import CalibrationRuler, OverlayLayer
from app.ui.panels.measure_panel import MeasurePanel
from app.ui.panels.ocr_panel import OcrPanel
from app.ui.workers import ExtractionWorker, OcrWorker
from app.utils.imaging import imread_unicode, bgr_to_qpixmap

PROJECT_FILTER = "图纸测距项目 (*.fdproj)"
IMAGE_FILTER = "图纸 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.pdf)"


DEFAULT_WINDOW_TITLE = "离线图纸失效点测距软件 — MVP"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(DEFAULT_WINDOW_TITLE)
        self.resize(1280, 820)

        # 服务
        self.project = ProjectService()
        self.ocr_engine = OcrEngine(lazy=True)
        self.recognition = RecognitionService(self.ocr_engine)
        self.extraction = ExtractionService(ExtractionParams())

        # 当前状态
        self.current_drawing = None
        self.current_image_bgr: np.ndarray | None = None
        self.current_crop: tuple | None = None
        self._worker: OcrWorker | None = None
        self._extract_worker: ExtractionWorker | None = None
        # 标定与提取结果（用于拖动后重算）
        self._per_pipeline_scale: list[float | None] = []
        self._global_scale: float | None = None
        self._latest_measures: list[MeasureResult] = []
        # 标定标尺（点对 + 米数）
        self._calibration_ruler: CalibrationRuler | None = None

        # UI
        self.view = DrawingView(self)
        self.setCentralWidget(self.view)
        self.view.cropSelected.connect(self._on_crop_selected)
        self.view.calibrationCompleted.connect(self._on_calibration_completed)
        self.view.addFailurePointRequested.connect(self._on_add_failure_clicked)
        self.overlay = OverlayLayer(self.view.scene())
        self.overlay.model_changed.connect(self._on_overlay_edited)

        self.ocr_panel = OcrPanel(self)
        self._ocr_dock = QDockWidget("OCR 结果", self)
        self._ocr_dock.setWidget(self.ocr_panel)
        # 仅允许停在右侧，避免误拖到中央把图"分栏"
        self._ocr_dock.setAllowedAreas(Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.RightDockWidgetArea, self._ocr_dock)

        self.measure_panel = MeasurePanel(self)
        self._measure_dock = QDockWidget("失效点距离", self)
        self._measure_dock.setWidget(self.measure_panel)
        self._measure_dock.setAllowedAreas(Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.RightDockWidgetArea, self._measure_dock)
        # 合并到同一位置，用 tab 切换，避免把画布挤窄
        self.tabifyDockWidget(self._ocr_dock, self._measure_dock)
        self._measure_dock.raise_()

        self.setStatusBar(QStatusBar(self))
        self._build_actions()
        self._update_actions_enabled()
        self._set_status("就绪。请新建或打开项目。")

    # ---- 菜单/工具栏 ----
    def _build_actions(self):
        tb = QToolBar("主工具栏", self)
        tb.setMovable(False)
        self.addToolBar(tb)
        menu = self.menuBar().addMenu("文件(&F)")

        self.act_new = QAction("新建项目", self, shortcut=QKeySequence.New,
                               triggered=self.on_new)
        self.act_open = QAction("打开项目", self, shortcut=QKeySequence.Open,
                                triggered=self.on_open)
        self.act_save = QAction("保存", self, shortcut=QKeySequence.Save,
                                triggered=self.on_save)
        self.act_save_as = QAction("另存为…", self, triggered=self.on_save_as)
        self.act_import = QAction("导入图纸…", self, triggered=self.on_import)
        self.act_crop = QAction("裁剪有效区域", self, checkable=True,
                                triggered=self.on_toggle_crop)
        self.act_uncrop = QAction("恢复完整图纸", self, triggered=self.on_uncrop)
        self.act_ocr = QAction("运行 OCR", self, triggered=self.on_run_ocr)
        self.act_extract = QAction("自动提取（管线+失效点）", self,
                                   triggered=self.on_run_extraction)
        self.act_calibrate = QAction("手动标定（点两点）", self, checkable=True,
                                     triggered=self.on_toggle_calibrate)
        self.act_add_failure = QAction("手动添加失效点", self, checkable=True,
                                       triggered=self.on_toggle_add_failure)
        self.act_export = QAction("导出 Excel…", self, triggered=self.on_export_xlsx)
        self.act_clear_overlay = QAction("清除叠加层", self,
                                        triggered=self.on_clear_overlay)
        self.act_quit = QAction("退出", self, triggered=self.close)

        for a in (self.act_new, self.act_open, self.act_save, self.act_save_as):
            menu.addAction(a)
        menu.addSeparator()
        menu.addAction(self.act_import)
        menu.addSeparator()
        menu.addAction(self.act_quit)

        for a in (self.act_new, self.act_open, self.act_save, self.act_import,
                  self.act_crop, self.act_uncrop, self.act_ocr, self.act_extract,
                  self.act_calibrate, self.act_add_failure,
                  self.act_export, self.act_clear_overlay):
            tb.addAction(a)

    def _update_actions_enabled(self):
        has_proj = self.project.workdir is not None
        has_img = self.current_image_bgr is not None
        self.act_save.setEnabled(has_proj)
        self.act_save_as.setEnabled(has_proj)
        self.act_import.setEnabled(has_proj)
        self.act_crop.setEnabled(has_img)
        self.act_uncrop.setEnabled(
            has_img
            and self.current_drawing is not None
            and self.current_drawing.crop_rect is not None
        )
        self.act_ocr.setEnabled(has_img)
        self.act_extract.setEnabled(has_img)
        self.act_calibrate.setEnabled(has_img)
        self.act_add_failure.setEnabled(has_img)
        self.act_export.setEnabled(
            has_img and bool(self.overlay.failure_points)
        )
        self.act_clear_overlay.setEnabled(has_img)

    # ---- 项目操作 ----
    def on_new(self):
        # 先掐掉旧项目的后台 worker，避免 stale 结果回灌到新项目
        self._detach_workers()
        try:
            self.project.new_project()
        except Exception as e:  # noqa: BLE001
            self._error("新建项目失败", e)
            return
        self._reset_view()
        self._set_status("已新建项目（未保存）。")
        self._update_actions_enabled()

    def on_open(self):
        path, _ = QFileDialog.getOpenFileName(self, "打开项目", "", PROJECT_FILTER)
        if not path:
            return
        self._detach_workers()
        try:
            self.project.open_project(path)
        except Exception as e:  # noqa: BLE001
            self._error("打开项目失败", e)
            return
        self._reset_view()
        self.setWindowTitle(f"离线图纸失效点测距软件 — {Path(path).name}")
        # 载入第一张图纸（若有）
        drawings = self.project.repo.get_drawings()
        if drawings:
            self._load_drawing(drawings[0])
        self._set_status(f"已打开：{Path(path).name}")
        self._update_actions_enabled()

    def on_save(self):
        if self.project.project_path is None:
            self.on_save_as()
            return
        self._do_save(str(self.project.project_path))

    def on_save_as(self):
        path, _ = QFileDialog.getSaveFileName(self, "另存为", "未命名.fdproj", PROJECT_FILTER)
        if not path:
            return
        if not path.endswith(".fdproj"):
            path += ".fdproj"
        self._do_save(path)

    def _do_save(self, path: str):
        try:
            self.project.save_project(path)
        except Exception as e:  # noqa: BLE001
            self._error("保存失败", e)
            return
        self.setWindowTitle(f"离线图纸失效点测距软件 — {Path(path).name}")
        self._set_status(f"已保存：{Path(path).name}")

    # ---- 图纸/裁剪/OCR ----
    def on_import(self):
        path, _ = QFileDialog.getOpenFileName(self, "导入图纸", "", IMAGE_FILTER)
        if not path:
            return
        try:
            drawing = self.project.import_drawing(path)
        except Exception as e:  # noqa: BLE001
            self._error("导入失败", e)
            return
        self._load_drawing(drawing)
        self._set_status(f"已导入：{Path(path).name}")
        self._update_actions_enabled()

    def _load_drawing(self, drawing):
        # 先释放叠加层 / 标尺的旧 QGraphicsItem，避免 scene.clear 后悬挂引用
        self.overlay.clear()
        if self._calibration_ruler is not None:
            self._calibration_ruler.remove()
            self._calibration_ruler = None
        self.current_drawing = drawing
        abs_path = self.project.abs_image_path(drawing.image_path)
        img = imread_unicode(abs_path)
        # 累积裁剪：DB 里 crop_rect 是相对原图的绝对坐标，应用后丢掉外侧像素
        if drawing.crop_rect:
            x, y, w, h = drawing.crop_rect
            x, y = max(0, int(x)), max(0, int(y))
            w = min(int(w), img.shape[1] - x)
            h = min(int(h), img.shape[0] - y)
            img = img[y:y + h, x:x + w].copy()
        self.current_image_bgr = img
        # 显示后的画布坐标系就是裁剪后的图像，后续 OCR/提取都不再二次裁剪
        self.current_crop = None
        self.view.load_pixmap(bgr_to_qpixmap(self.current_image_bgr))
        self.ocr_panel.populate([])
        # 若数据库里已有提取/测距结果，恢复显示
        if self.project.repo is not None:
            pipelines = self.project.repo.get_pipelines(drawing.id)
            fpoints = self.project.repo.get_failure_points(drawing.id)
            measures = self.project.repo.get_measure_results(drawing.id)
            if pipelines:
                self.overlay.set_pipelines(pipelines)
            if fpoints:
                self.overlay.set_failure_points(fpoints)
            self._latest_measures = list(measures)
            self.measure_panel.populate(measures, self._global_scale)
        else:
            self.measure_panel.populate([], None)
        self._update_actions_enabled()

    def on_toggle_crop(self, checked: bool):
        self.view.set_crop_mode(checked)
        self._set_status(
            "裁剪模式：拖拽框选有效区域，松手即刻切掉框外像素。"
            if checked else "已退出裁剪模式。"
        )

    def _on_crop_selected(self, rect: QRectF):
        if self.current_image_bgr is None or self.current_drawing is None:
            return
        H, W = self.current_image_bgr.shape[:2]
        lx = max(0, int(rect.x()))
        ly = max(0, int(rect.y()))
        lw = max(1, min(int(rect.width()), W - lx))
        lh = max(1, min(int(rect.height()), H - ly))
        if lw < 4 or lh < 4:
            self._set_status("裁剪框太小，已忽略。")
            return

        # 立即切除外侧
        cropped = self.current_image_bgr[ly:ly + lh, lx:lx + lw].copy()

        # 把本次相对裁剪合成到"原图坐标系"的绝对裁剪存档
        prev = self.current_drawing.crop_rect
        abs_x = (prev[0] if prev else 0) + lx
        abs_y = (prev[1] if prev else 0) + ly
        abs_rect = (abs_x, abs_y, lw, lh)
        if self.project.repo is not None:
            self.project.repo.update_crop(self.current_drawing.id, abs_rect)
        self.current_drawing.crop_rect = abs_rect
        self.project.dirty = True

        # 替换显示图像 + 清空旧叠加层（坐标系已改变）
        self.current_image_bgr = cropped
        self.current_crop = None
        self.view.load_pixmap(bgr_to_qpixmap(cropped))
        self.overlay.clear()
        self._latest_measures = []
        self._per_pipeline_scale = []
        self._global_scale = None
        self.measure_panel.populate([], None)

        # 退出裁剪模式
        self.act_crop.setChecked(False)
        self.view.set_crop_mode(False)
        self._update_actions_enabled()
        self._set_status(
            f"已裁剪为 {lw}×{lh} 像素，原图外侧已丢弃。请重新运行 OCR / 自动提取。"
        )

    def on_uncrop(self):
        """撤销所有裁剪，恢复完整原图。"""
        if self.current_drawing is None:
            return
        if self.current_drawing.crop_rect is None:
            self._set_status("当前已是完整图纸，无需撤销。")
            return
        if self.project.repo is not None:
            self.project.repo.update_crop(self.current_drawing.id, None)
        self.current_drawing.crop_rect = None
        self.project.dirty = True
        # 既然回到原图坐标系，之前提取的折线/失效点（在裁剪图坐标系里）已经失效
        self.overlay.clear()
        self._latest_measures = []
        self._per_pipeline_scale = []
        self._global_scale = None
        self.measure_panel.populate([], None)
        self._load_drawing(self.current_drawing)
        self._set_status("已恢复完整图纸。请重新运行 OCR / 自动提取。")

    def on_run_ocr(self):
        if self.current_image_bgr is None or self.current_drawing is None:
            return
        self.act_ocr.setEnabled(False)
        self._set_status(f"正在识别…（OCR 后端：{self.ocr_engine.backend}）")
        self._worker = OcrWorker(
            self.recognition, self.current_image_bgr,
            self.current_drawing.id, self.current_crop,
        )
        self._worker.finished_ok.connect(self._on_ocr_done)
        self._worker.failed.connect(self._on_ocr_failed)
        self._worker.start()

    def _on_ocr_done(self, items, labels):
        self.ocr_panel.populate(items)
        if self.project.repo is not None and self.current_drawing is not None:
            self.project.repo.replace_length_labels(self.current_drawing.id, labels)
            self.project.dirty = True
        self.act_ocr.setEnabled(True)
        self._set_status(
            f"识别完成：{len(items)} 个文本，{len(labels)} 个长度标注（后端 {self.ocr_engine.backend}）。"
        )

    def _on_ocr_failed(self, msg: str):
        self.act_ocr.setEnabled(True)
        self._set_status("OCR 失败。")
        QMessageBox.warning(
            self, "OCR 不可用",
            f"{msg}\n\n提示：MVP 骨架未内置模型时需先安装 OCR 引擎：\n"
            f"    pip install rapidocr\n首次运行会在本地加载 ONNX 模型，全程离线。",
        )

    # ---- 自动提取（管线 + 失效点 + 测距）----
    def on_run_extraction(self):
        if self.current_image_bgr is None or self.current_drawing is None:
            return
        labels = []
        if self.project.repo is not None:
            labels = self.project.repo.get_length_labels(self.current_drawing.id)
        self.act_extract.setEnabled(False)
        self._set_status(
            "正在自动提取管线 / 失效点 / 测距…（输入长度标注 "
            f"{len(labels)} 条）"
        )
        self._extract_worker = ExtractionWorker(
            self.extraction,
            self.current_image_bgr,
            self.current_drawing.id,
            labels,
        )
        self._extract_worker.finished_ok.connect(self._on_extract_done)
        self._extract_worker.failed.connect(self._on_extract_failed)
        self._extract_worker.start()

    def _on_extract_done(self, result: ExtractionResult):
        self.act_extract.setEnabled(True)
        # 标定
        self._per_pipeline_scale = list(result.per_pipeline_scale)
        self._global_scale = result.global_scale_m_per_px
        # 显示
        self.overlay.set_pipelines(result.pipelines)
        self.overlay.set_failure_points(result.failure_points)
        self._latest_measures = list(result.measures)
        self.measure_panel.populate(result.measures, self._global_scale)
        # 持久化
        if self.project.repo is not None and self.current_drawing is not None:
            did = self.current_drawing.id
            self.project.repo.replace_pipelines(did, result.pipelines)
            self.project.repo.replace_failure_points(did, result.failure_points)
            self.project.repo.replace_measure_results(did, result.measures)
            self.project.dirty = True
        scale = (
            f"{self._global_scale:.3f} m/px"
            if self._global_scale else "未标定（仅像素值）"
        )
        self._update_actions_enabled()
        self._set_status(
            f"自动提取完成：{len(result.pipelines)} 条管线、"
            f"{len(result.failure_points)} 颗失效点、"
            f"{len(result.measures)} 对距离 · 比例 {scale}。可拖动控制点 / 星标修正。"
        )

    def _on_extract_failed(self, msg: str):
        self.act_extract.setEnabled(True)
        self._set_status("自动提取失败。")
        QMessageBox.warning(self, "自动提取失败", msg)

    def on_clear_overlay(self):
        self.overlay.clear()
        self._latest_measures = []
        self.measure_panel.populate([], None)
        self._update_actions_enabled()
        self._set_status("已清除叠加层。")

    # ---- 导出 Excel ----
    def on_export_xlsx(self):
        if not self.overlay.failure_points:
            QMessageBox.information(self, "无可导出数据",
                                    "尚未提取失效点。请先运行自动提取。")
            return
        if not self._latest_measures:
            # 没有测距记录时也允许导出（只导出失效点）
            ans = QMessageBox.question(
                self, "确认导出",
                "当前距离表为空，是否仍要导出失效点列表？",
            )
            if ans != QMessageBox.Yes:
                return
        default_name = "测距结果.xlsx"
        if self.project.project_path is not None:
            default_name = self.project.project_path.stem + "_测距结果.xlsx"
        path, _ = QFileDialog.getSaveFileName(
            self, "导出 Excel", default_name, "Excel 工作簿 (*.xlsx)"
        )
        if not path:
            return
        if not path.lower().endswith(".xlsx"):
            path += ".xlsx"
        try:
            out = export_to_xlsx(
                path,
                self.overlay.failure_points,
                self._latest_measures,
                self.overlay.pipelines,
                global_scale_m_per_px=self._global_scale,
                per_pipeline_scale=self._per_pipeline_scale,
                project_name=(
                    self.project.project_path.stem
                    if self.project.project_path else ""
                ),
                drawing_name=(
                    self.current_drawing.image_path
                    if self.current_drawing else ""
                ),
            )
        except Exception as e:  # noqa: BLE001
            self._error("导出失败", e)
            return
        self._set_status(
            f"已导出：{out.name}（失效点 {len(self.overlay.failure_points)}，"
            f"距离 {len(self._latest_measures)} 对）"
        )

    # ---- 手动标定（点两点 + 输入米数 → 全局 m/px）----
    def on_toggle_calibrate(self, checked: bool):
        self.view.set_calibrate_mode(checked)
        if checked:
            self._set_status(
                "手动标定：在画布上点击 **第一个** 端点，移动鼠标可看到拉出的预览线。"
            )
        else:
            self._set_status("已退出手动标定。")

    def on_toggle_add_failure(self, checked: bool):
        self.view.set_add_failure_mode(checked)
        if checked:
            self._set_status(
                "添加失效点模式：每次在画布上点击都会落下一颗带编号的红五角星，"
                "可直接拖动调整位置。再次点击工具栏按钮（或按 Esc）退出。"
            )
        else:
            self._set_status("已退出手动添加失效点。")

    def _on_add_failure_clicked(self, pt: QPointF):
        if self.current_drawing is None:
            return
        self.overlay.add_failure_point(
            self.current_drawing.id, float(pt.x()), float(pt.y())
        )
        # 添加后距离重算靠 overlay.model_changed → _on_overlay_edited 自动跑

    def keyPressEvent(self, event):
        # Esc 退出手动添加 / 标定模式
        if event.key() == Qt.Key_Escape:
            if self.act_add_failure.isChecked():
                self.act_add_failure.setChecked(False)
                self.on_toggle_add_failure(False)
                return
            if self.act_calibrate.isChecked():
                self.act_calibrate.setChecked(False)
                self.on_toggle_calibrate(False)
                return
        # Delete / Backspace 删除选中的失效点
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            selected = self.overlay.selected_failure_ids()
            if selected:
                for fid in selected:
                    self.overlay.remove_failure_point(fid)
                self._set_status(f"已删除 {len(selected)} 个失效点。")
                self._update_actions_enabled()
                return
        super().keyPressEvent(event)

    def _on_calibration_completed(self, p1: QPointF, p2: QPointF):
        dx, dy = p2.x() - p1.x(), p2.y() - p1.y()
        pixel_dist = (dx * dx + dy * dy) ** 0.5
        if pixel_dist < 1.0:
            self._set_status("两点过近，标定已取消。")
            self.act_calibrate.setChecked(False)
            self.view.set_calibrate_mode(False)
            return
        meters, ok = QInputDialog.getDouble(
            self, "输入实际长度",
            f"两点像素距离 {pixel_dist:.2f} px\n请输入对应的实际长度（米）：",
            value=100.0, minValue=0.001, maxValue=1e7, decimals=3,
        )
        # 不论是否确认，先退出标定模式（预览已经清空）
        self.act_calibrate.setChecked(False)
        self.view.set_calibrate_mode(False)
        if not ok:
            self._set_status("已取消标定。")
            return
        scale = meters / pixel_dist
        self._global_scale = scale
        n = len(self._per_pipeline_scale)
        self._per_pipeline_scale = [scale] * n if n else []
        # 替换画布上的永久标尺（黑色线 + 端点 + 距离文字）
        if self._calibration_ruler is not None:
            self._calibration_ruler.remove()
        self._calibration_ruler = CalibrationRuler(
            self.view.scene(), p1, p2, _format_meters(meters)
        )
        self._set_status(
            f"标定完成：{scale:.4f} m/px（{meters} m ÷ {pixel_dist:.2f} px）。"
        )
        self._on_overlay_edited()

    # ---- 拖动后增量重算 ----
    def _on_overlay_edited(self):
        pipelines = self.overlay.pipelines
        failures = self.overlay.failure_points
        if not pipelines or not failures:
            self.measure_panel.populate([], self._global_scale)
            return
        polylines = [p.points_px for p in pipelines]
        points = [fp.raw_px for fp in failures]
        scales = self._per_pipeline_scale[: len(polylines)]
        proj, pairs = measure_all(
            points,
            polylines,
            scales=scales,
            fallback_scale=self._global_scale,
            max_offset_px=self.extraction.params.max_failure_offset_px,
        )
        # 同步 failure point 的 projected / pipeline / offset / state
        for i, fp in enumerate(failures):
            pj = proj[i]
            if pj is None:
                fp.projected_px = None
                fp.pipeline_id = None
                fp.offset_px = None
                fp.state = ReviewState.NEED_REVIEW
            else:
                fp.projected_px = pj.proj_xy
                fp.pipeline_id = pipelines[pj.pipeline_index].id
                fp.offset_px = pj.offset_px
                fp.state = (
                    ReviewState.NEED_REVIEW
                    if pj.offset_px > self.extraction.params.failure_review_offset_px
                    else ReviewState.AUTO
                )

        measures = []
        did = self.current_drawing.id if self.current_drawing else ""
        for pair in pairs:
            from_fp = failures[pair.from_index]
            to_fp = failures[pair.to_index]
            same_pipe = pair.pipeline_index >= 0
            need_review = (
                from_fp.state != ReviewState.AUTO
                or to_fp.state != ReviewState.AUTO
                or pair.straight_m is None
            )
            pipe_name = (
                pipelines[pair.pipeline_index].name if same_pipe else "—（跨管线）"
            )
            measures.append(
                MeasureResult(
                    id=new_id("mr_"),
                    drawing_id=did,
                    from_code=from_fp.code,
                    to_code=to_fp.code,
                    pipeline_name=pipe_name,
                    straight_px=pair.straight_px,
                    along_pipe_px=pair.along_pipe_px,
                    straight_m=pair.straight_m,
                    along_pipe_m=pair.along_pipe_m,
                    basis="manual_edit" if same_pipe else "cross_pipeline",
                    confidence=0.9,
                    need_review=need_review,
                )
            )
        self._latest_measures = measures
        self.measure_panel.populate(measures, self._global_scale)
        if self.project.repo is not None and self.current_drawing is not None:
            self.project.repo.replace_pipelines(did, pipelines)
            self.project.repo.replace_failure_points(did, failures)
            self.project.repo.replace_measure_results(did, measures)
            self.project.dirty = True

    # ---- 辅助 ----
    def _reset_view(self):
        self.current_drawing = None
        self.current_image_bgr = None
        self.current_crop = None
        # 顺序关键：先让 overlay / ruler 主动放下 QGraphicsItem，再 scene.clear()。
        # 反过来会让 overlay 持有已被 Qt 删掉的 C++ 对象，下次访问 path.scene()
        # 抛 RuntimeError 中断 _reset_view，结果就是再点新建+导入图纸无法显示。
        self.overlay.clear()
        if self._calibration_ruler is not None:
            self._calibration_ruler.remove()
            self._calibration_ruler = None
        self.view.clear()
        self.ocr_panel.populate([])
        self.measure_panel.reset()
        self._per_pipeline_scale = []
        self._global_scale = None
        self._latest_measures = []
        self.act_crop.setChecked(False)
        self.act_calibrate.setChecked(False)
        self.act_add_failure.setChecked(False)
        self.view.set_calibrate_mode(False)
        self.view.set_add_failure_mode(False)
        self.setWindowTitle(DEFAULT_WINDOW_TITLE)

    def _detach_workers(self):
        """掐断旧 worker 的信号连接：QThread 不能安全 terminate，但断信号能
        让 stale 结果到达时直接丢弃，避免回灌到新项目。"""
        for attr in ("_worker", "_extract_worker"):
            w = getattr(self, attr, None)
            if w is None:
                continue
            try:
                w.finished_ok.disconnect()
            except (TypeError, RuntimeError):
                pass
            try:
                w.failed.disconnect()
            except (TypeError, RuntimeError):
                pass
            setattr(self, attr, None)

    def _set_status(self, text: str):
        self.statusBar().showMessage(text)

    def _error(self, title: str, e: Exception):
        QMessageBox.critical(self, title, str(e))


def _format_meters(m: float) -> str:
    if m >= 1000:
        return f"{m / 1000:g} km"
    if m >= 1:
        return f"{m:g} m"
    return f"{m * 100:g} cm"

    def closeEvent(self, event):
        if self.project.dirty:
            ret = QMessageBox.question(
                self, "退出", "项目有未保存的更改，仍要退出吗？",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            )
            if ret == QMessageBox.Cancel:
                event.ignore()
                return
            if ret == QMessageBox.Save:
                self.on_save()
        event.accept()
