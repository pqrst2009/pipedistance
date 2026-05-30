"""程序入口。

运行：
    python -m app.main
或打包后双击启动。全程离线，不发起任何网络请求。
"""
from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from app.ui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("PipeDistance")
    app.setOrganizationName("PipeDistance")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
