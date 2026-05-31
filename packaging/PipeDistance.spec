# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 配置：把 PipeDistance 全部依赖打成单目录可执行包。

用法（在目标平台上分别构建）：
    pyinstaller --noconfirm packaging/PipeDistance.spec

跨平台均可：Windows 走 .bat、macOS 走 .sh。spec 本身与平台无关，
依赖收集由 PyInstaller 钩子在当前平台抓正确的 .dll / .so / .dylib。
"""
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_all,
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)


ROOT = Path(SPECPATH).parent.resolve()
APP_NAME = "PipeDistance"

# 主依赖一律 collect_all 抓全（子模块 + 数据文件 + C 动态库），
# 避免点名遗漏：
#   - shapely 内部 GEOS C 库
#   - PIL/Pillow 各种 _imaging / _imagingft 扩展
#   - PyMuPDF (fitz) C 扩展 + 字体资源
#   - skimage / scipy 海量懒加载子模块
#   - openpyxl 各种可选写入后端
HIDDEN_IMPORTS: list[str] = []
DATAS: list[tuple[str, str]] = [
    (str(ROOT / "app" / "persistence" / "schema.sql"), "app/persistence"),
]
BINARIES: list[tuple[str, str]] = []

for pkg in ("skimage", "scipy", "shapely", "PIL", "fitz", "rapidocr", "onnxruntime"):
    try:
        pkg_datas, pkg_binaries, pkg_hidden = collect_all(pkg)
        DATAS += pkg_datas
        BINARIES += pkg_binaries
        HIDDEN_IMPORTS += pkg_hidden
    except Exception as e:
        # 某些包（如可选的 rapidocr_onnxruntime）可能未安装，跳过即可
        print(f"[spec] collect_all({pkg!r}) skipped: {e}")

# 兼容包（旧 rapidocr_onnxruntime API），装了就一起带
try:
    rdo_datas, rdo_binaries, rdo_hidden = collect_all("rapidocr_onnxruntime")
    DATAS += rdo_datas
    BINARIES += rdo_binaries
    HIDDEN_IMPORTS += rdo_hidden
except Exception:
    pass

# 顺手把 openpyxl 全子模块带上（写 xlsx 走 lxml.etree 等不同分支）
HIDDEN_IMPORTS += collect_submodules("openpyxl")

# PySide6 的关键子模块（PyInstaller 内置 hook 通常会处理，列出以防）
HIDDEN_IMPORTS += [
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
]

# 排除明确用不到的大型模块，缩小体积
EXCLUDES = [
    "tkinter",
    # 注意：不要排除 unittest / test —— skimage / scipy 等科学库会偷偷 import
    # unittest 做类型断言，排掉它运行时会 ModuleNotFoundError
    # PySide6 用不到的子模块
    "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineWidgets",
    "PySide6.QtMultimedia",
    "PySide6.QtMultimediaWidgets",
    "PySide6.Qt3DCore",
    "PySide6.Qt3DRender",
    "PySide6.QtCharts",
    "PySide6.QtDataVisualization",
    "PySide6.QtDesigner",
    "PySide6.QtQuick3D",
]


a = Analysis(
    [str(ROOT / "app" / "main.py")],
    pathex=[str(ROOT)],
    binaries=BINARIES,
    datas=DATAS,
    hiddenimports=HIDDEN_IMPORTS,
    hookspath=[str(ROOT / "packaging" / "hooks")],
    runtime_hooks=[],
    excludes=EXCLUDES,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # GUI 应用，不显示控制台窗口
    disable_windowed_traceback=False,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    a.zipfiles,
    strip=False,
    upx=False,
    upx_exclude=[],
    name=APP_NAME,
)
