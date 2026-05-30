# 离线图纸失效点测距软件 — 项目骨架（S1 / MVP）

跨平台（Windows / macOS）、**完全离线**的管线图纸失效点智能测距工具。
本仓库为开发方案中"第一阶段（S1）"的可运行骨架，已打通：

**新建/打开/保存项目 → 导入图纸（含 PDF 首页）→ 框选裁剪有效区域 → 运行 OCR → 长度标注归一化与展示**

后续 Sprint 在此架构上叠加：管线中心线绘制、失效点标注、投影、沿管线测距、Excel/图片导出。

## 快速开始

```bash
# 1. 建议使用 Python 3.11 / 3.12
python -m venv .venv
# Windows: .venv\Scripts\activate   macOS/Linux: source .venv/bin/activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 运行
python -m app.main

# 4. 跑单元测试（纯逻辑，无需 OCR/GUI）
pytest
```

> **关于 OCR**：骨架默认依赖 `rapidocr`（ONNX，跨平台、离线）。若未安装，软件仍可正常
> 导入/裁剪/保存，仅"运行 OCR"会弹出友好提示让你 `pip install rapidocr`。
> 选用 RapidOCR 而非 PaddleOCR 是为了规避 paddlepaddle 在 Apple Silicon 上的兼容/打包问题。

## 架构（四层）

```
表现层 (PySide6)     app/ui/        主窗口、画布(QGraphicsView)、OCR面板、后台线程
应用服务层           app/services/  项目生命周期(.fdproj)、识别编排
算法引擎层 (无 Qt)   app/engine/    预处理、OCR封装、长度标注归一化
数据持久层           app/persistence/  SQLite 读写 + schema
领域模型             app/domain/    dataclass 实体
工具                 app/utils/     资源路径(打包兼容)、图像IO(Unicode安全)
```

- 引擎层不依赖 PySide6，可在 pytest 中独立测试。
- 长耗时 OCR 在 `QThread` 中执行，UI 不卡顿。
- 项目文件 `.fdproj` 是 zip 容器：`meta.json` + `project.sqlite` + `images/`。

## 目录

```
app/
  main.py                 入口
  domain/models.py        领域模型
  persistence/            schema.sql / repository.py
  engine/                 preprocess / ocr / label_parse
  services/               project_service / recognition_service
  ui/                     main_window / canvas / panels / workers
  utils/                  resource / imaging
tests/engine/             单元测试
packaging/                PyInstaller hook、打包说明
```

## 已实现 vs 待办

| 能力 | 状态 |
|---|---|
| 项目新建/打开/保存（.fdproj） | ✅ |
| 图纸导入（png/jpg/bmp/tiff/PDF首页） | ✅ |
| 画布缩放/平移/框选裁剪 | ✅ |
| OCR 识别（RapidOCR，后台线程） | ✅ |
| 长度标注正则归一化（排除 DN/PN/φ/MPa） | ✅ |
| SQLite 持久化（drawing / length_label） | ✅ |
| 浅绿管线自动提取（HSV → 骨架 → 折线 + 跨星缝合） | ✅ S2 |
| 红色五角星失效点检测（HSV → 凸缺陷过滤） | ✅ S2 |
| 长度标注 → 比例 m/px（按管线中位数 + 全局兜底） | ✅ S2 |
| 失效点投影 + 星-星沿管距离（Shapely） | ✅ S2 |
| 叠加层：折线顶点 / 星标可拖动手工修正 | ✅ S2 |
| 持久化（pipeline / failure_point / measure_result） | ✅ S2 |
| Excel / 标注图导出 | ⏳ S3 |

## S2 用法

1. 新建或打开项目，导入图纸。
2. （可选）运行 OCR，得到长度标注，自动用于比例标定。
3. 点工具栏「**自动提取（管线+失效点）**」：
   - 浅绿色线条被识别为管线，红色五角星作为失效点测量基准。
   - 右侧「**失效点距离**」面板列出星 ↔ 星 沿管 / 直线距离（米；未标定时显示像素）。
4. 在画布上：
   - 拖动管线上的**绿色控制点**修正折线走向；
   - 拖动**红色星标**改变失效点位置；
   - 移动后距离表即时刷新，状态会自动标 `need_review`。
5. 保存（Ctrl+S）会把管线 / 失效点 / 距离写入 `.fdproj` 中的 SQLite。

## 许可与合规
- 运行时不发起任何网络请求，图纸与数据不出本机。
- 依赖均为商用友好许可（PySide6 LGPL、OpenCV/Shapely/NetworkX BSD/Apache、RapidOCR Apache-2.0）。
