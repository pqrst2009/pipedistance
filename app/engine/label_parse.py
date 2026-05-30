"""长度标注文本归一化（引擎层，纯函数，便于单测）。

将 OCR 文本判定为"长度标注"并归一化为米：
  支持： 120m / 230米 / L=85m / 长度=120m / 0.35km / 1.2公里
  排除： DN300 / PN1.6 / φ219 / Φ325 / 0.8MPa 等工艺参数
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# 需排除的非长度工艺参数（出现即整体判为非长度）
_EXCLUDE = re.compile(r"(DN\s*\d+|PN\s*[\d.]+|[φΦ]\s*\d+|[\d.]+\s*MPa)", re.IGNORECASE)

# 长度匹配：可选前缀(L= / 长度=)，数字，单位(km/公里 优先于 m/米)
_LENGTH = re.compile(
    r"(?:L\s*=\s*|长度\s*=?\s*)?(\d+(?:\.\d+)?)\s*(km|公里|m|米)",
    re.IGNORECASE,
)

_UNIT_TO_M = {"m": 1.0, "米": 1.0, "km": 1000.0, "公里": 1000.0}


@dataclass
class ParsedLength:
    value_m: float
    unit: str
    raw_number: float


def normalize_fullwidth(text: str) -> str:
    """全角数字/字母/等号转半角。"""
    out = []
    for ch in text:
        code = ord(ch)
        if code == 0x3000:          # 全角空格
            out.append(" ")
        elif 0xFF01 <= code <= 0xFF5E:  # 全角 ASCII
            out.append(chr(code - 0xFEE0))
        else:
            out.append(ch)
    return "".join(out)


def parse_length_text(text: str) -> ParsedLength | None:
    """返回 ParsedLength 或 None（非长度文本）。"""
    if not text:
        return None
    t = normalize_fullwidth(text).strip()
    if _EXCLUDE.search(t):
        return None
    m = _LENGTH.search(t)
    if not m:
        return None
    number = float(m.group(1))
    unit = m.group(2).lower()
    if number <= 0:
        return None
    value_m = number * _UNIT_TO_M[unit]
    # 合理性过滤：管段长度一般不会出现 0 或超大异常值
    if value_m > 100000:
        return None
    return ParsedLength(value_m=value_m, unit=unit, raw_number=number)
