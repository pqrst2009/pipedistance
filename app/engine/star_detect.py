"""红色五角星检测：返回每颗星的中心点。

适配两类星标：
- 实心红五角星（旧规格）
- **空心红五角星**（实际图纸常见）：只有红色轮廓线，内部可能盖了数字/文字。

策略（轻量、无模型）：
1. 红色掩膜（外部已做）。
2. ``_fill_outline``：先用闭运算修补轮廓上的细小断裂，再把外轮廓内部填实，
   把空心星变成实心 silhouette；这一步对实心星几乎无影响。
3. 对每个连通域做面积 / 长宽比 / **solidity** 过滤：
   - 长宽比 ≈ 1（五角星近似正方形）。
   - solidity = 轮廓面积 / 凸包面积，五角星 ≈ 0.45-0.85，圆 ≈ 1.0，数字 ≈ 0.7-1.0。
   - solidity 是分辨"星 vs 圆 / 文字"的主力指标。
4. ``concavity_count``：凸缺陷计数 ≈ 5 作为附加确认；对小星 / 锯齿可关闭。
5. 中心点取连通域质心（对填实后的星 = 几何中心）。
"""
from __future__ import annotations

import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from skimage.feature import peak_local_max


# 诊断日志：Windows 打包后 peak_local_max 静默退化会导致多颗合体星无法拆分，
# 把异常 / 关键模块版本写到用户目录，方便远程排查。
# 仅在异常路径和模块首次加载时写，正常运行不产生 I/O。
_DIAG_LOG = Path(os.environ.get("PIPEDISTANCE_DIAG_LOG") or
                 (Path.home() / "pipedistance_diag.log"))


def _diag_write(msg: str) -> None:
    try:
        with _DIAG_LOG.open("a", encoding="utf-8") as f:
            f.write(msg.rstrip() + "\n")
    except Exception:
        # 诊断本身不能抛错影响主流程
        pass


def _diag_probe_once() -> None:
    """模块加载时跑一次 peak_local_max 自检，记录版本和探针结果。"""
    try:
        import skimage
        import scipy
        probe = np.zeros((9, 9), dtype=np.float32)
        probe[4, 4] = 5.0
        coords = peak_local_max(probe, min_distance=1, threshold_abs=1.0)
        _diag_write(
            f"[probe] platform={sys.platform} frozen={getattr(sys, 'frozen', False)} "
            f"skimage={skimage.__version__} scipy={scipy.__version__} "
            f"cv2={cv2.__version__} numpy={np.__version__} "
            f"probe_peaks={len(coords)}"
        )
    except Exception:
        _diag_write(
            f"[probe] platform={sys.platform} frozen={getattr(sys, 'frozen', False)} "
            f"FAILED:\n{traceback.format_exc()}"
        )


_diag_probe_once()


@dataclass(frozen=True)
class StarCenter:
    x: float
    y: float
    area: float
    concavity_count: int
    solidity: float = 0.0


@dataclass(frozen=True)
class StarDetectParams:
    min_area: int = 80          # 小于此面积当噪声丢弃；实际 burst 标记常在 100-150 px
    max_area: int = 50000
    max_aspect_ratio: float = 1.35  # 单颗星 ≈ 1.0；> 1.35 推断为多星合体或文字
    # 空心星轮廓填实：先 close 再 drawContours FILLED。0 = 跳过（仅旧测试场景）
    fill_outline: bool = True
    # 双通道 close kernel：小 kernel(0) 抓白心红刺爆炸 + 实心星真中心，
    # 大 kernel(5) 抓需要 close 桥接 outline 的空心五角星。
    # **顺序很重要**：先 0 后 5，dedup 按 FIFO，真中心优先于 close 引入的"中点峰"
    # （重叠星被 close=5 融成长条 → 距离变换峰落在两星之间）。
    fill_close_kernels: tuple[int, ...] = (0, 5)
    # 轮廓 bbox 面积超过此值时**不**填实——避免把红色边界框 / 大面积红色
    # 阴影当成"巨型空心星"填满，吞掉里面真正的失效点。
    # 经验：单颗星的 bbox 上限 ~ max_area * 4（含星尖之间的空角）。
    fill_max_bbox_area: int = 200000
    # 两通道结果去重半径（像素）：两通道命中位置距离 ≤ 此值视为同一颗。
    # 实测 close=5 通道在重叠星合体上产生"中点伪峰"距真中心 ~15.13 px，
    # 需要 ≥ 16 才能吃掉。相邻独立失效点的中心距图纸里典型 ≥ 25 px，
    # 仍能安全保留。
    dedup_radius_px: float = 16.0
    # solidity = contourArea / hullArea；五角星 ≈ 0.45-0.80，圆/数字 ≈ ≥0.85
    use_solidity: bool = True
    min_solidity: float = 0.30
    max_solidity: float = 0.85
    # 形状档案：每条 (target_concavity, tolerance) 都接受。默认覆盖：
    #   - 五角星 (5±1) ：图纸上常见的红色五角星（空心或实心）
    #   - 多角爆炸 (10±3) ：白心红刺的"扎啊/爆炸"图标，spike 数 7-13 都算
    # 凸缺陷数 ≥ 14 仍可能是多星合体或文字噪声，留给 split_overlapping 兜底。
    use_concavity: bool = True
    min_concavity: float = 2.0       # 凸缺陷深度阈值（像素）
    concavity_profiles: tuple[tuple[int, int], ...] = ((5, 1), (10, 3))
    # 重叠星分离：单连通域的形状像"多星合体"时，用距离变换峰值切分
    split_overlapping: bool = True
    # 相邻星中心最少像素间距。实测：单孤立星 r∈[22,80] 在 md∈[4,8] 都只给 1 个峰，
    # 把 md 降到 6 才能分离三连重叠星（中心距 25）。再大就会漏 3 颗中的第 3 颗。
    peak_min_distance: int = 6
    peak_min_radius_px: float = 4.0   # 峰值高度（星内接圆半径下限），过滤文字
    # 进入拆分分支的门槛（小连通域 / 过实心 / 太瘦长的形状不会被误拆为多星）
    min_split_area: int = 1200         # 多星合体的连通域面积下限（避开"3.5"等文字合体）
    # 连通域整体守门 solidity：低于此值不进入任何检测路径（含拆分），
    # 用于挡掉巨型边界框 / 大块线描的 hollow ring（solidity 通常 <0.15）。
    split_min_solidity: float = 0.20


def detect_stars(red_mask: np.ndarray, params: StarDetectParams | None = None) -> list[StarCenter]:
    """从红色掩膜检测失效点（五角星 / 白心红刺爆炸）中心。

    输入：uint8 0/255 掩膜。
    输出：``StarCenter`` 列表，按面积降序。

    双通道策略：
    - 通道 1（close_kernel 大）：桥接空心五角星 outline 的 AA 缝隙，
      让 drawContours FILLED 成出完整 silhouette。但同样会把爆炸 spike 抹平。
    - 通道 2（close_kernel=0）：保留爆炸的 spike 特征，但对线描很薄、有缝隙的
      空心五角星可能漏。
    两通道命中按中心距离去重，互不漏掉。
    """
    p = params or StarDetectParams()
    if not p.fill_outline:
        return _detect_once(red_mask, p)

    all_hits: list[StarCenter] = []
    for k in p.fill_close_kernels:
        work_mask = _fill_outline(red_mask, k, p.fill_max_bbox_area)
        all_hits.extend(_detect_once(work_mask, p, _skip_fill=True))
    return _dedup_centers(all_hits, p.dedup_radius_px)


def _detect_once(
    work_mask: np.ndarray,
    p: StarDetectParams,
    _skip_fill: bool = False,
) -> list[StarCenter]:
    """单通道检测：连通域 → 距离变换 + 形状校验。"""
    if not _skip_fill and p.fill_outline:
        work_mask = _fill_outline(work_mask, p.fill_close_kernels[0], p.fill_max_bbox_area)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(work_mask, connectivity=8)
    results: list[StarCenter] = []

    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < p.min_area or area > p.max_area:
            continue
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        if min(w, h) == 0:
            continue
        ratio = max(w, h) / min(w, h)
        cx, cy = float(centroids[i, 0]), float(centroids[i, 1])
        comp = (labels == i).astype(np.uint8) * 255

        # 早期 solidity 门槛：边界框 / 大尺寸线描形成的 hollow ring solidity ≈ 0.03，
        # 不能让它们绕过拆分路径假装"多颗合体星"。真实星簇即使重叠 solidity 也 ≥ 0.25。
        # use_solidity=False 时跳过此守门（旧测试场景）。
        comp_solidity = _solidity(comp) if p.use_solidity else 1.0
        if p.use_solidity and comp_solidity < p.split_min_solidity:
            continue

        # 距离变换 + 局部峰值（始终运行）
        peaks: list[tuple[float, float, float]] = []
        if p.split_overlapping:
            peaks = _split_by_distance_peaks(
                comp,
                min_distance=p.peak_min_distance,
                min_height=p.peak_min_radius_px,
            )

        # 多峰 + 面积足够 → 多颗合体星，直接采用每个峰
        if len(peaks) >= 2 and area >= p.min_split_area:
            for px, py, peak_r in peaks:
                results.append(
                    StarCenter(
                        x=float(px), y=float(py),
                        area=float(peak_r * peak_r * 3.14),
                        concavity_count=0,
                        solidity=0.0,
                    )
                )
            continue

        # 单峰 / 无峰：走严格单星形状校验
        solidity = comp_solidity
        conc = _concavity_count(comp, p.min_concavity) if p.use_concavity else 0
        passes_aspect = ratio <= p.max_aspect_ratio
        passes_solidity = (
            not p.use_solidity
            or (p.min_solidity <= solidity <= p.max_solidity)
        )
        passes_concavity = (
            not p.use_concavity
            or any(abs(conc - t) <= tol for t, tol in p.concavity_profiles)
        )
        if passes_aspect and passes_solidity and passes_concavity:
            results.append(
                StarCenter(
                    x=cx, y=cy, area=float(area),
                    concavity_count=conc, solidity=solidity,
                )
            )
            continue

        # 单星检查失败：若连通域足够大且偏长，按主轴拆成 2 颗（中等重叠星的兜底）
        if (
            p.split_overlapping
            and area >= p.min_split_area
            and ratio > p.max_aspect_ratio  # 必须偏长才推断是两颗合体
        ):
            for px, py, peak_r in _two_centers_along_major_axis(comp):
                results.append(
                    StarCenter(
                        x=px, y=py,
                        area=float(peak_r * peak_r * 3.14),
                        concavity_count=conc,
                        solidity=solidity,
                    )
                )

    results.sort(key=lambda s: s.area, reverse=True)
    return results


def _dedup_centers(hits: list[StarCenter], radius_px: float) -> list[StarCenter]:
    """按中心距离去重：FIFO，先到先得。

    调用方负责把"更可信"的通道排在前面 —— 当前 detect_stars 把 close=0 通道
    排在第一，所以重叠星的真中心优先，close=5 引入的"中点伪峰"被丢弃。
    """
    if not hits:
        return []
    kept: list[StarCenter] = []
    r2 = radius_px * radius_px
    for s in hits:
        ok = True
        for k in kept:
            dx, dy = s.x - k.x, s.y - k.y
            if dx * dx + dy * dy <= r2:
                ok = False
                break
        if ok:
            kept.append(s)
    # 最终结果按面积降序，保持 API 兼容
    kept.sort(key=lambda s: s.area, reverse=True)
    return kept


def _two_centers_along_major_axis(component_mask: np.ndarray) -> list[tuple[float, float, float]]:
    """对偏长的连通域，沿主轴推断"两颗星"的中心。

    用法：当距离变换只给出 1 个峰但连通域明显被拉长（aspect ratio > 阈值且面积够大）时，
    几乎可断定是两颗星紧贴在一起。用二阶矩拟合主轴方向，沿主轴在质心两侧各放一个中心。
    """
    M = cv2.moments(component_mask)
    if M["m00"] <= 0:
        return []
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    mu20 = M["mu20"] / M["m00"]
    mu02 = M["mu02"] / M["m00"]
    mu11 = M["mu11"] / M["m00"]
    cov = np.array([[mu20, mu11], [mu11, mu02]], dtype=float)
    try:
        eigvals, eigvecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return []
    # eigvals 升序：[小, 大]
    minor_var, major_var = float(eigvals[0]), float(eigvals[1])
    if major_var <= 0:
        return []
    major_axis = eigvecs[:, 1]
    # 两中心放在主轴上 ±0.5 * 主轴标准差处（经验值，匹配两个紧贴圆/星的典型位置）
    offset = (major_var ** 0.5) * 0.85
    p1 = (cx - major_axis[0] * offset, cy - major_axis[1] * offset)
    p2 = (cx + major_axis[0] * offset, cy + major_axis[1] * offset)
    # peak_r 用次轴标准差 ≈ 单颗星的内接圆半径
    peak_r = max(1.0, minor_var ** 0.5)
    return [
        (float(p1[0]), float(p1[1]), peak_r),
        (float(p2[0]), float(p2[1]), peak_r),
    ]


def _split_by_distance_peaks(
    component_mask: np.ndarray,
    min_distance: int,
    min_height: float,
) -> list[tuple[float, float, float]]:
    """在单连通域内用距离变换 + 局部峰值找 ≥1 个星中心。

    返回 [(x, y, peak_radius_px), ...]。
    """
    dist = cv2.distanceTransform(component_mask, cv2.DIST_L2, 5)
    try:
        coords = peak_local_max(
            dist,
            min_distance=min_distance,
            threshold_abs=min_height,
            exclude_border=False,
        )
    except Exception:
        # Windows 打包后 skimage / scipy 子模块缺失会在这里抛 ImportError / AttributeError，
        # 落盘后退化为「只走质心 + 单星形状校验」分支，至少独立星仍能识别。
        _diag_write(
            f"[peak_local_max] FAILED mask_shape={component_mask.shape} "
            f"min_distance={min_distance} min_height={min_height}\n"
            f"{traceback.format_exc()}"
        )
        return []
    out: list[tuple[float, float, float]] = []
    for (py, px) in coords:
        out.append((float(px), float(py), float(dist[py, px])))
    return _drop_midpoint_peaks(out)


def _drop_midpoint_peaks(
    peaks: list[tuple[float, float, float]],
    height_margin: float = 0.5,
) -> list[tuple[float, float, float]]:
    """剔除"中点伪峰"——两颗重叠星合体后距离变换在中点处会升高（因覆盖区域比
    单颗星更宽），生成一个比两端真中心都高的虚假峰。

    规则：峰 P 若被另外两峰 Q, R 夹在中间（向量 PQ·PR 为正）且 P 的 inscribed
    radius **严格高于**两端 + ``height_margin``，判为伪峰丢弃。

    真三连/四连等密叠场景里中间峰高度与两端相近或更低，不会被误删。
    """
    if len(peaks) < 3:
        return peaks
    drop = [False] * len(peaks)
    for i, (xi, yi, ri) in enumerate(peaks):
        if drop[i]:
            continue
        for j in range(len(peaks)):
            if j == i or drop[j]:
                continue
            for k in range(j + 1, len(peaks)):
                if k == i or drop[k]:
                    continue
                xj, yj, rj = peaks[j]
                xk, yk, rk = peaks[k]
                # i 是否落在 j-k 连线之间（向量同向）
                v1x, v1y = xi - xj, yi - yj
                v2x, v2y = xk - xi, yk - yi
                if v1x * v2x + v1y * v2y <= 0:
                    continue
                if ri > rj + height_margin and ri > rk + height_margin:
                    drop[i] = True
                    break
            if drop[i]:
                break
    return [p for p, d in zip(peaks, drop) if not d]


def _fill_outline(mask: np.ndarray, close_kernel: int,
                  max_bbox_area: int | None = None) -> np.ndarray:
    """把空心轮廓填实成 silhouette。

    1. 先用闭运算修补轮廓上的细小断裂（如 1-2 像素的裂缝），
       让外轮廓在 ``findContours`` 时变成闭环。
    2. 用 ``drawContours(..., thickness=cv2.FILLED)`` 把外轮廓内部填满。
       对原本就是实心的星，此步几乎无副作用。

    ``max_bbox_area`` 限制只填合理大小的轮廓：图纸里红色边界框 / 区域阴影
    的 bbox 远大于单颗星，若不限制，drawContours FILLED 会把整片区域填实，
    把里面的真实星标全部吞掉。
    """
    if close_kernel > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return mask
    filled = mask.copy()
    for c in contours:
        if max_bbox_area is not None:
            x, y, w, h = cv2.boundingRect(c)
            if w * h > max_bbox_area:
                continue   # 边界框 / 大块阴影，跳过
        cv2.drawContours(filled, [c], -1, 255, thickness=cv2.FILLED)
    return filled


def _solidity(component_mask: np.ndarray) -> float:
    """实际像素数 / 凸包面积。五角星 ≈ 0.45-0.80。无效则返回 0。

    用 ``cv2.countNonZero`` 而非 ``cv2.contourArea``：后者把外轮廓当成填实多边形
    算面积，对 hollow ring（如边界框线描）会错算成"完全填实"（1.0）。
    像素计数才反映实际占用，hollow 形状的低 solidity 才能正确识别。
    """
    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0
    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 3:
        return 0.0
    pixel_area = float(cv2.countNonZero(component_mask))
    hull_area = cv2.contourArea(cv2.convexHull(cnt))
    if hull_area <= 0:
        return 0.0
    return float(pixel_area / hull_area)


def _concavity_count(component_mask: np.ndarray, min_depth_px: float) -> int:
    """计算单个连通域相对其凸包的凹陷数（depth > min_depth_px 才计）。

    cv2.convexityDefects 的 depth 单位是 ``像素 * 256``，按 256 缩放回像素。
    """
    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0
    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 5:
        return 0
    hull = cv2.convexHull(cnt, returnPoints=False)
    if hull is None or len(hull) < 4:
        return 0
    try:
        defects = cv2.convexityDefects(cnt, hull)
    except cv2.error:
        return 0
    if defects is None:
        return 0
    count = 0
    for k in range(defects.shape[0]):
        depth = defects[k, 0, 3] / 256.0
        if depth >= min_depth_px:
            count += 1
    return count
