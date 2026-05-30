# 打包说明

> **重要**：PyInstaller 不支持跨平台编译。Windows 包必须在 Windows 上构建，
> macOS 包必须在 macOS 上构建。Mac 用户要出 Windows EXE 用下面的 **方案 A（GitHub Actions）**。

最终交付物运行时完全离线（OCR 模型已内置）；构建过程可联网（下载依赖、模型）。

---

## 方案 A：GitHub Actions（推荐）

零 Windows 本地依赖，push 一个 tag 自动产出 zip 包。

1. 把项目推到 GitHub。
2. 打个 tag 触发：
   ```bash
   git tag v0.2.0
   git push origin v0.2.0
   ```
   或者在 GitHub 仓库 **Actions → Build Windows EXE → Run workflow** 手动跑一次。
3. 等约 10-15 分钟，去 **Actions → 该次运行 → Artifacts** 下载 `PipeDistance-windows.zip`。
4. 解压后双击 `PipeDistance.exe` 即可运行（解压后约 600 MB，含全部 Qt / OpenCV / OCR 模型）。

工作流文件：[.github/workflows/build-windows.yml](../.github/workflows/build-windows.yml)

---

## 方案 B：本地 Windows 机器打包

需要 Windows 10/11 + Python 3.11 / 3.12（推荐）。

```cmd
git clone <repo>
cd pipedistance
packaging\build_windows.bat
```

脚本自动完成：
1. 创建 `.venv` 虚拟环境
2. `pip install -r requirements.txt`
3. 触发一次 `RapidOCR()` 让模型下载到 `.venv\Lib\site-packages\rapidocr\models\`
4. `pyinstaller --noconfirm packaging\PipeDistance.spec`
5. 把 `dist\PipeDistance\` 压缩为 `dist\PipeDistance-windows.zip`

产出：
- 程序目录：`dist\PipeDistance\`
- 入口：`dist\PipeDistance\PipeDistance.exe`
- 分发包：`dist\PipeDistance-windows.zip`（约 200 MB 压缩后）

---

## 方案 C：macOS 自测打包（仅开发用）

```bash
./packaging/build_macos.sh
```

产出 `dist/PipeDistance-macos-arm64.zip`（或 `-x86_64.zip`，依平台架构）。

> 线上发布 macOS 包需要 codesign + notarytool 公证 + dmgbuild 制作 dmg，
> 参考下方 "macOS 签名 / 公证" 段落。

---

## spec 文件做了什么

[packaging/PipeDistance.spec](PipeDistance.spec) 通用配置：

- `app/main.py` 作为入口
- 显式 `hiddenimports` 列出 `rapidocr` / `skimage.feature` / `scipy.ndimage` 等 PyInstaller 静态分析可能漏掉的子包
- `collect_data_files('rapidocr')` 把 `.onnx` 模型 + 字典 + 配置全部打包
- `collect_dynamic_libs('onnxruntime')` 把推理后端 DLL/SO 全部打包
- 通过 hooks 收集 `app/persistence/schema.sql`
- 排除 `QtWebEngine` / `QtMultimedia` / `tkinter` 等用不到的大模块缩小体积
- `console=False` 隐藏命令行窗口

---

## 常见坑

### 1. `onnxruntime_pybind11_state` 加载失败
- 通常因为缺 VC++ 2015-2022 运行库。让用户装 [vc_redist.x64.exe](https://aka.ms/vs/17/release/vc_redist.x64.exe)。
- 或在 `--add-binary` 里强制带上 `msvcp140.dll` / `vcruntime140*.dll`。

### 2. 首次启动卡很久
- 因为 ONNX 模型要载入内存（约 16 MB），冷启动 2-5 秒正常。

### 3. 包体积太大（600 MB+）
- 主要是 PySide6 Qt 库（300 MB）+ OpenCV（100 MB）+ scipy/skimage（150 MB），属于这套技术栈的固定成本。
- 用 UPX 可以再压 20-30%（在 spec 里 `upx=True` 并指定 `upx_dir`），但 macOS 上签名会破坏 UPX 压缩，慎用。

### 4. SmartScreen 拦截 EXE
- 未签名的 EXE 第一次运行会有"未知发布者"提示。商用发布请买代码签名证书并签名。

---

## macOS 签名 / 公证（仅线上发布需要）

```bash
codesign --deep --force --options runtime \
  --sign "Developer ID Application: YOUR NAME (TEAMID)" \
  dist/PipeDistance.app
xcrun notarytool submit PipeDistance.dmg --apple-id ... --team-id ... --wait
xcrun stapler staple PipeDistance.dmg
```

然后用 `dmgbuild` / `create-dmg` 把 .app 打成 .dmg 分发。
