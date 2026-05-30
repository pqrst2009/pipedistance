"""应用服务层：项目生命周期管理。

项目文件 .fdproj 是一个 zip 容器：
    meta.json        项目元数据
    project.sqlite   全部识别/标注/测距数据
    images/          图纸资源

打开时解压到临时工作目录；保存时回写 SQLite 并重新打包。
"""
from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from app.domain.models import Drawing, new_id
from app.persistence.repository import Repository
from app.utils.resource import resource_path

META_NAME = "meta.json"
DB_NAME = "project.sqlite"
IMAGES_DIR = "images"
APP_VERSION = "0.1.0"
RASTER_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class ProjectService:
    def __init__(self):
        self.workdir: Path | None = None
        self.project_path: Path | None = None    # None 表示尚未保存
        self.repo: Repository | None = None
        self.meta: dict = {}
        self.dirty: bool = False

    # ---- 生命周期 ----
    def new_project(self) -> None:
        self._cleanup()
        self.workdir = Path(tempfile.mkdtemp(prefix="fdproj_"))
        (self.workdir / IMAGES_DIR).mkdir(parents=True, exist_ok=True)
        self.repo = Repository(self.workdir / DB_NAME)
        self.repo.init_schema(resource_path("app/persistence/schema.sql"))
        self.meta = {
            "app_version": APP_VERSION,
            "unit": "m",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._write_meta()
        self.project_path = None
        self.dirty = True

    def open_project(self, path: str) -> None:
        self._cleanup()
        self.workdir = Path(tempfile.mkdtemp(prefix="fdproj_"))
        with zipfile.ZipFile(path) as z:
            z.extractall(self.workdir)
        db = self.workdir / DB_NAME
        if not db.exists():
            raise IOError("项目文件损坏：缺少 project.sqlite")
        self.repo = Repository(db)
        meta_path = self.workdir / META_NAME
        self.meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        self.project_path = Path(path)
        self.dirty = False

    def save_project(self, path: str | None = None) -> None:
        if self.workdir is None or self.repo is None:
            raise RuntimeError("没有打开的项目")
        if path:
            self.project_path = Path(path)
        if self.project_path is None:
            raise ValueError("未指定保存路径（请使用另存为）")
        self.repo.commit()
        self._write_meta()
        tmp = self.project_path.with_suffix(self.project_path.suffix + ".tmp")
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(self.workdir.rglob("*")):
                if f.is_file():
                    z.write(f, f.relative_to(self.workdir))
        tmp.replace(self.project_path)
        self.dirty = False

    # ---- 图纸导入 ----
    def import_drawing(self, src_path: str) -> Drawing:
        if self.workdir is None or self.repo is None:
            raise RuntimeError("请先新建或打开项目")
        src = Path(src_path)
        ext = src.suffix.lower()
        did = new_id("dwg_")
        dst = self.workdir / IMAGES_DIR / f"{did}.png"

        if ext in RASTER_EXTS:
            w, h = self._copy_raster(src, dst)
        elif ext == ".pdf":
            w, h = self._render_pdf_first_page(src, dst)
        else:
            raise ValueError(f"不支持的格式：{ext}")

        drawing = Drawing(
            id=did,
            image_path=f"{IMAGES_DIR}/{dst.name}",
            width_px=w,
            height_px=h,
        )
        self.repo.insert_drawing(drawing)
        self.repo.log("drawing", did, "import", src.name)
        self.dirty = True
        return drawing

    def abs_image_path(self, rel: str) -> str:
        assert self.workdir is not None
        return str(self.workdir / rel)

    # ---- 内部 ----
    def _copy_raster(self, src: Path, dst: Path) -> tuple[int, int]:
        from PIL import Image

        with Image.open(src) as im:
            im = im.convert("RGB")
            w, h = im.size
            im.save(dst, format="PNG")
        return w, h

    def _render_pdf_first_page(self, src: Path, dst: Path) -> tuple[int, int]:
        try:
            import fitz  # PyMuPDF
        except ImportError as e:  # noqa: F841
            raise RuntimeError("导入 PDF 需要 PyMuPDF：pip install PyMuPDF")
        doc = fitz.open(str(src))
        page = doc.load_page(0)
        pix = page.get_pixmap(dpi=200)
        pix.save(str(dst))
        return pix.width, pix.height

    def _write_meta(self) -> None:
        assert self.workdir is not None
        (self.workdir / META_NAME).write_text(
            json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _cleanup(self) -> None:
        if self.repo is not None:
            try:
                self.repo.close()
            except Exception:  # noqa: BLE001
                pass
            self.repo = None
        if self.workdir is not None and self.workdir.exists():
            shutil.rmtree(self.workdir, ignore_errors=True)
        self.workdir = None
