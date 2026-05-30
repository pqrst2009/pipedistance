"""长度标注归一化单元测试（纯逻辑，无需 Qt/OCR）。

运行：pytest tests/engine/test_label_parse.py
"""
from app.engine.label_parse import parse_length_text


def test_basic_meters():
    assert parse_length_text("120m").value_m == 120.0
    assert parse_length_text("230米").value_m == 230.0


def test_prefix_forms():
    assert parse_length_text("L=85m").value_m == 85.0
    assert parse_length_text("长度=120m").value_m == 120.0
    assert parse_length_text("长度 120 m").value_m == 120.0


def test_kilometers():
    assert parse_length_text("0.35km").value_m == 350.0
    assert parse_length_text("1.2公里").value_m == 1200.0


def test_fullwidth():
    # 全角数字与等号
    assert parse_length_text("Ｌ＝１２０ｍ").value_m == 120.0


def test_excludes_process_params():
    assert parse_length_text("DN300") is None
    assert parse_length_text("PN1.6") is None
    assert parse_length_text("φ219") is None
    assert parse_length_text("Φ325") is None
    assert parse_length_text("0.8MPa") is None


def test_non_length_text():
    assert parse_length_text("阀室") is None
    assert parse_length_text("A线") is None
    assert parse_length_text("") is None


def test_unreasonable_values():
    assert parse_length_text("0m") is None
    assert parse_length_text("999999km") is None  # 超大异常值
