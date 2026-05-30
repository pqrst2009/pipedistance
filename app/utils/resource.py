"""资源路径解析：兼容开发态与 PyInstaller 打包态。

打包后资源被解压到 sys._MEIPASS；开发态以项目根目录为基准。
"""
from __future__ import annotations

import sys
from pathlib import Path


def project_root() -> Path:
    # 本文件位于 app/utils/resource.py -> 上溯两级为项目根
    return Path(__file__).resolve().parents[2]


def resource_path(relative: str) -> str:
    """relative 形如 'app/persistence/schema.sql' 或 'app/resources/models'。"""
    base = getattr(sys, "_MEIPASS", None)
    if base:
        return str(Path(base) / relative)
    return str(project_root() / relative)
