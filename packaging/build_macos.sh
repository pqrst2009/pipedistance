#!/usr/bin/env bash
# macOS 打包脚本（开发期自测用；线上交付参考 BUILD.md 走签名 + 公证）。
set -e

cd "$(dirname "$0")/.."

if [ ! -d .venv ]; then
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

# 预下载 OCR 模型（让模型一并进 app bundle）
python -c "from rapidocr import RapidOCR; import numpy as np; RapidOCR()(np.zeros((10,10,3), dtype='uint8'))"

rm -rf build dist
pyinstaller --noconfirm packaging/PipeDistance.spec

ARCH=$(uname -m)
ZIP_NAME="PipeDistance-macos-${ARCH}.zip"
( cd dist && zip -rq "${ZIP_NAME}" PipeDistance )
echo "Done: dist/PipeDistance/, dist/${ZIP_NAME}"
