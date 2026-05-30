# PyInstaller hook：确保 onnxruntime 的二进制与 rapidocr 的模型/配置被收集。
# 放置于 packaging/hooks/，打包时通过 --additional-hooks-dir packaging/hooks 引用。
from PyInstaller.utils.hooks import collect_dynamic_libs, collect_data_files

# onnxruntime 的 .dll/.so/.dylib（关键：否则打包后报 onnxruntime_pybind11_state 加载失败）
binaries = collect_dynamic_libs("onnxruntime")

# rapidocr 自带的模型与 yaml 配置
datas = collect_data_files("rapidocr")
datas += collect_data_files("rapidocr_onnxruntime")
