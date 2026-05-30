@echo off
REM ============================================================
REM Windows 打包脚本：在 Windows 上运行此文件
REM   1. 创建/激活虚拟环境
REM   2. 装依赖
REM   3. 预下载 OCR 模型（让模型一并进包，最终用户无需联网）
REM   4. 调 PyInstaller 出可执行包到 dist\PipeDistance\
REM   5. 打 zip
REM
REM 用法：双击或 cmd 里执行
REM   packaging\build_windows.bat
REM ============================================================
setlocal enableextensions enabledelayedexpansion

cd /d "%~dp0\.."

if not exist .venv (
    echo [1/5] 创建虚拟环境 .venv
    python -m venv .venv
    if errorlevel 1 goto :err
)

echo [2/5] 激活虚拟环境并安装依赖
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 goto :err

echo [3/5] 预下载 OCR 模型（首次需要联网，之后离线可用）
python -c "from rapidocr import RapidOCR; import numpy as np; RapidOCR()(np.zeros((10,10,3), dtype='uint8'))"
if errorlevel 1 goto :err

echo [4/5] 调用 PyInstaller 打包
if exist build rmdir /s /q build
if exist dist  rmdir /s /q dist
pyinstaller --noconfirm packaging\PipeDistance.spec
if errorlevel 1 goto :err

echo [5/5] 打包成 zip
powershell -NoProfile -Command "Compress-Archive -Path dist\PipeDistance\* -DestinationPath dist\PipeDistance-windows.zip -Force"
if errorlevel 1 goto :err

echo.
echo ===============================
echo 打包成功
echo   程序目录: dist\PipeDistance\
echo   入口:    dist\PipeDistance\PipeDistance.exe
echo   分发 zip: dist\PipeDistance-windows.zip
echo ===============================
goto :eof

:err
echo.
echo 打包失败，错误码 %errorlevel%
exit /b %errorlevel%
