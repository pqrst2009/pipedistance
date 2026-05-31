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

# 显式列出 PyInstaller 静态分析可能漏掉的隐式导入
HIDDEN_IMPORTS = [
    # OCR
    "rapidocr",
    "rapidocr.main",
    "onnxruntime",
    "onnxruntime.capi",
    "onnxruntime.capi._pybind_state",
    # shapely
    "shapely",
    "shapely.geometry",
    "shapely.ops",
    # PIL
    "PIL._imaging",
    "PIL.Image",
    # PySide6 主要子模块（Analysis 一般能找到，列出以防）
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    # Excel
    "openpyxl",
    "openpyxl.workbook",
]

# 资源文件
DATAS = [
    (str(ROOT / "app" / "persistence" / "schema.sql"), "app/persistence"),
]

# rapidocr 自带模型 / 字典 / 配置（含首次运行后下载到 site-packages 的 onnx）
DATAS += collect_data_files("rapidocr")
# rapidocr_onnxruntime 兼容包（旧 API）若装了，也带上
try:
    DATAS += collect_data_files("rapidocr_onnxruntime")
except Exception:
    pass

# onnxruntime 的动态库（DLL/SO/dylib）
BINARIES = collect_dynamic_libs("onnxruntime")

# scikit-image / scipy：整个包都收（用了 skeletonize、peak_local_max 等
# 间接通过 lazy 加载触发的子模块；如果只点名 skimage.feature 会漏一堆）
for pkg in ("skimage", "scipy"):
    pkg_datas, pkg_binaries, pkg_hidden = collect_all(pkg)
    DATAS += pkg_datas
    BINARIES += pkg_binaries
    HIDDEN_IMPORTS += pkg_hidden

# 顺手把 openpyxl 全子模块也带上（写 xlsx 走 lxml.etree 等不同分支）
HIDDEN_IMPORTS += collect_submodules("openpyxl")

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
