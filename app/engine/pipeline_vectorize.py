"""管线矢量化：浅绿掩膜 → 骨架 → 折线列表。

流程：
1. 输入二值掩膜（来自 ``color_detect.green_mask`` + ``clean_mask``）。
2. 用 ``skimage.morphology.skeletonize`` 细化为 1 像素宽骨架。
3. 把骨架像素当作 8-邻接图，按"端点 / 分叉点"切段（分支提取）。
4. 每条分支用 Ramer-Douglas-Peucker 简化为折线。
5. 过短分支按 ``min_branch_px`` 丢弃。

不依赖 networkx：对 8-邻接骨架做 BFS 即可，简单稳定。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.morphology import skeletonize


@dataclass(frozen=True)
class VectorizeParams:
    min_branch_px: int = 20      # 折线总像素长度阈值，过短视为噪声
    rdp_epsilon: float = 2.0     # RDP 简化容差（像素）


def skeleton_from_mask(mask: np.ndarray) -> np.ndarray:
    """uint8 0/255 掩膜 → uint8 0/1 骨架。"""
    return skeletonize(mask > 0).astype(np.uint8)


def _neighbors(y: int, x: int, h: int, w: int):
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                yield ny, nx


def _count_neighbors(skel: np.ndarray, y: int, x: int) -> int:
    h, w = skel.shape
    return sum(1 for ny, nx in _neighbors(y, x, h, w) if skel[ny, nx])


def _extract_branches(skel: np.ndarray) -> list[list[tuple[int, int]]]:
    """从骨架提取所有"端点-端点 / 分叉-分叉 / 闭环"分支。

    返回每条分支按顺序排列的像素点 ``(x, y)`` 列表。
    """
    h, w = skel.shape
    skel = skel.copy()
    # 标记端点 (邻居=1) 与分叉点 (邻居>=3) 作为切割种子
    endpoints: list[tuple[int, int]] = []
    junctions: list[tuple[int, int]] = []
    for y in range(h):
        for x in range(w):
            if not skel[y, x]:
                continue
            n = _count_neighbors(skel, y, x)
            if n == 1:
                endpoints.append((y, x))
            elif n >= 3:
                junctions.append((y, x))

    branches: list[list[tuple[int, int]]] = []
    visited_edge: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    def edge_key(a, b):
        return (a, b) if a < b else (b, a)

    def walk(start: tuple[int, int]) -> None:
        # 从 start 沿未访问的边走，遇端点/分叉/已访问则终止
        sy, sx = start
        for ny, nx in _neighbors(sy, sx, h, w):
            if not skel[ny, nx]:
                continue
            key = edge_key((sy, sx), (ny, nx))
            if key in visited_edge:
                continue
            path: list[tuple[int, int]] = [(sx, sy)]
            visited_edge.add(key)
            py, px = sy, sx
            cy, cx = ny, nx
            path.append((cx, cy))
            while True:
                nbrs = [
                    (yy, xx) for yy, xx in _neighbors(cy, cx, h, w)
                    if skel[yy, xx] and (yy, xx) != (py, px)
                ]
                # 分叉/端点：结束
                if len(nbrs) != 1:
                    break
                nyy, nxx = nbrs[0]
                key2 = edge_key((cy, cx), (nyy, nxx))
                if key2 in visited_edge:
                    break
                visited_edge.add(key2)
                path.append((nxx, nyy))
                py, px = cy, cx
                cy, cx = nyy, nxx
                # 走到端点/分叉，停
                deg = _count_neighbors(skel, cy, cx)
                if deg != 2:
                    break
            branches.append(path)

    for ep in endpoints:
        walk(ep)
    for jp in junctions:
        walk(jp)

    # 处理闭环（无端点也无分叉的孤立环）：找尚未访问的骨架点
    visited_pix: set[tuple[int, int]] = set()
    for path in branches:
        for x, y in path:
            visited_pix.add((y, x))
    for y in range(h):
        for x in range(w):
            if not skel[y, x] or (y, x) in visited_pix:
                continue
            # 沿任一方向走一圈
            path = [(x, y)]
            visited_pix.add((y, x))
            py, px = -1, -1
            cy, cx = y, x
            while True:
                nbrs = [
                    (yy, xx) for yy, xx in _neighbors(cy, cx, h, w)
                    if skel[yy, xx] and (yy, xx) != (py, px) and (yy, xx) not in visited_pix
                ]
                if not nbrs:
                    break
                nyy, nxx = nbrs[0]
                path.append((nxx, nyy))
                visited_pix.add((nyy, nxx))
                py, px = cy, cx
                cy, cx = nyy, nxx
            if len(path) > 1:
                branches.append(path)
    return branches


def _rdp(points: list[tuple[float, float]], epsilon: float) -> list[tuple[float, float]]:
    """Ramer-Douglas-Peucker 折线简化（迭代实现，避免大递归栈）。"""
    if len(points) < 3:
        return list(points)
    keep = [False] * len(points)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        # 离 (i, j) 直线最远的点
        x1, y1 = points[i]
        x2, y2 = points[j]
        dx, dy = x2 - x1, y2 - y1
        denom = (dx * dx + dy * dy) ** 0.5
        max_d = -1.0
        max_k = -1
        for k in range(i + 1, j):
            x0, y0 = points[k]
            if denom == 0:
                d = ((x0 - x1) ** 2 + (y0 - y1) ** 2) ** 0.5
            else:
                d = abs(dy * x0 - dx * y0 + x2 * y1 - y2 * x1) / denom
            if d > max_d:
                max_d, max_k = d, k
        if max_d > epsilon:
            keep[max_k] = True
            stack.append((i, max_k))
            stack.append((max_k, j))
    return [points[k] for k, ok in enumerate(keep) if ok]


def _polyline_length(pts: list[tuple[float, float]]) -> float:
    return sum(
        ((pts[i + 1][0] - pts[i][0]) ** 2 + (pts[i + 1][1] - pts[i][1]) ** 2) ** 0.5
        for i in range(len(pts) - 1)
    )


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def bridge_through_points(
    polylines: list[list[tuple[float, float]]],
    via_points: list[tuple[float, float]],
    max_gap_px: float = 40.0,
) -> list[list[tuple[float, float]]]:
    """把端点散落在 ``via_points`` 周围的多条折线缝合成一条。

    用途：红色五角星压在浅绿管线上会把它打断；矢量化出多段后，
    用星心做"中转点"把跨星的两段连回去。

    规则（贪心）：对每个中转点，把附近 (距离 ≤ max_gap_px) 的所有端点按距离
    升序取前 2 个；若来自不同折线，则合并这两条折线，链路上插入中转点。
    多于 2 个的端点继续配对。
    """
    polys: list[list[tuple[float, float]] | None] = [list(p) for p in polylines]

    def endpoint_candidates(via: tuple[float, float]):
        out = []
        for i, poly in enumerate(polys):
            if poly is None:
                continue
            d0 = _dist(poly[0], via)
            d1 = _dist(poly[-1], via)
            if d0 < max_gap_px:
                out.append((d0, i, 0))
            if d1 < max_gap_px:
                out.append((d1, i, -1))
        out.sort()
        return out

    def merge(i_a: int, end_a: int, i_b: int, end_b: int, via: tuple[float, float]):
        a = polys[i_a]
        b = polys[i_b]
        assert a is not None and b is not None
        # 让 a 末端、b 起端靠近 via
        seg_a = a if end_a == -1 else list(reversed(a))
        seg_b = b if end_b == 0 else list(reversed(b))
        polys[i_a] = seg_a + [via] + seg_b
        polys[i_b] = None

    for via in via_points:
        # 一颗星可能缝合多对端点（极少见，但循环处理更稳）
        while True:
            cands = endpoint_candidates(via)
            if len(cands) < 2:
                break
            d0, i0, e0 = cands[0]
            pair = next(
                ((dx, ix, ex) for dx, ix, ex in cands[1:] if ix != i0),
                None,
            )
            if pair is None:
                break
            d1, i1, e1 = pair
            merge(i0, e0, i1, e1, via)

    return [p for p in polys if p is not None]


def vectorize(mask: np.ndarray, params: VectorizeParams | None = None) -> list[list[tuple[float, float]]]:
    """主入口：掩膜 → 折线列表，每条折线为 ``[(x, y), ...]`` 像素坐标。"""
    p = params or VectorizeParams()
    skel = skeleton_from_mask(mask)
    branches = _extract_branches(skel)
    out: list[list[tuple[float, float]]] = []
    for br in branches:
        # 像素长度（按 8-连通对角线算近似就够用，骨架已经按邻接排序）
        floats = [(float(x), float(y)) for x, y in br]
        if _polyline_length(floats) < p.min_branch_px:
            continue
        simplified = _rdp(floats, p.rdp_epsilon)
        if len(simplified) >= 2:
            out.append(simplified)
    return out
