# -*- coding: utf-8 -*-
"""线框表格检测器（layout_analyzer 的补充模块）。

PP-DocLayout 对细线表格常漏检，而规则书数据表几乎都有线框。
检测原理：形态学提取横向/纵向长线；表格区域 = 横线与竖线都出现
（各自膨胀后取交）的连通块；再以"行线数 ≥ MIN_ROW_LINES"校验，
排除页面边框、横幅、装饰盒子（它们只有一两条线）。
"""
import cv2
import numpy as np

# 处理时把页面缩放到该宽度（提速且阈值尺度稳定）
_WORK_W = 1500
# 形态学核（横向线 / 纵向线 的最小长度，工作尺度下）
_H_KERNEL = 60
_V_KERNEL = 60
# 线膨胀半径（工作尺度下）：横竖线相距该距离内视为相交
_CROSS_DILATE = 6
# 表格最小尺寸（工作尺度下）
_MIN_W, _MIN_H = 200, 120
# 表格内最少行线数（排除边框/横幅，它们只有 1-2 条线）
_MIN_ROW_LINES = 5
# 表格内最少精确交点数（横竖线真实相交的像素数，排除噪声线偶然重叠）
_MIN_CROSS_PIXELS = 20


def line_maps(img_path):
    """提取页面的横线/竖线二值图（工作尺度）。

    返回: (h_lines, v_lines, scale)，线=255。h_lines/v_lines 为工作尺度
    （宽 _WORK_W）下的二值图，scale = 工作宽/原图宽（用于坐标换算）。
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None, 1.0
    scale = _WORK_W / img.shape[1]
    small = cv2.resize(img, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_AREA)
    # 自适应二值化：THRESH_BINARY 把"比局部背景亮"的像素标为 255。
    # 浅底深线：线是暗的 -> 需反转；深底浅线：线是亮的 -> 直接可用。
    bw = cv2.adaptiveThreshold(small, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY, 31, -4)
    lines = 255 - bw if small.mean() > 128 else bw
    h_lines = cv2.morphologyEx(lines, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (_H_KERNEL, 1)))
    v_lines = cv2.morphologyEx(lines, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (1, _V_KERNEL)))
    return h_lines, v_lines, scale


def detect_tables(img_path):
    """检测页面中的线框表格区域。

    返回: 原图坐标下的表格 bbox 列表 [[x0, y0, x1, y1], ...]
    """
    h_lines, v_lines, scale = line_maps(img_path)
    if h_lines is None:
        return []
    exact_cross = cv2.bitwise_and(h_lines, v_lines)  # 精确交点

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (_CROSS_DILATE, _CROSS_DILATE))
    grid = cv2.bitwise_and(cv2.dilate(h_lines, k), cv2.dilate(v_lines, k))
    grid = cv2.dilate(grid, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

    cnts, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tables = []
    for c in cnts:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw < _MIN_W or ch < _MIN_H:
            continue
        n_rows = _count_row_lines(h_lines, x, y, cw, ch)
        if n_rows < _MIN_ROW_LINES:
            continue  # 边框/横幅：行线太少
        # 内部交点校验：真表格交点遍布内部；页面边框/装饰框交点只在边角，
        # 其"内部 90% 区域"几乎没有交点。
        inner = _count_inner_cross(exact_cross, x, y, cw, ch, margin=0.06)
        if inner < _MIN_CROSS_PIXELS:
            continue
        # 转回原图坐标
        tables.append([int(x / scale), int(y / scale),
                       int((x + cw) / scale), int((y + ch) / scale)])
    return _merge_nearby(tables)


def _count_row_lines(h_lines, x, y, w, h):
    """统计 bbox 内横向线的条数（y 投影的连续段数）。"""
    sub = h_lines[y:y + h, x:x + w]
    proj = sub.sum(axis=1)
    rows = proj > w * 60  # 一条完整横线贡献约 w*255
    runs, in_run = 0, False
    for v in rows:
        if v and not in_run:
            runs += 1
            in_run = True
        elif not v:
            in_run = False
    return runs


def _count_inner_cross(exact_cross, x, y, w, h, margin=0.06):
    """统计 bbox 内部（去掉边缘 margin 比例）的精确交点像素数。"""
    mx, my = int(w * margin), int(h * margin)
    if w - 2 * mx <= 0 or h - 2 * my <= 0:
        return 0
    return int(np.count_nonzero(exact_cross[y + my:y + h - my, x + mx:x + w - mx]))


def _merge_nearby(tables):
    """合并互相重叠或贴近的表格 bbox（同一表格被拆成多段时）。"""
    if len(tables) <= 1:
        return tables
    merged = []
    tables = sorted(tables)
    cur = tables[0]
    for t in tables[1:]:
        if (t[0] < cur[2] and cur[0] < t[2]
                and min(t[3], cur[3]) - max(t[1], cur[1]) > -40):
            cur = [min(cur[0], t[0]), min(cur[1], t[1]),
                   max(cur[2], t[2]), max(cur[3], t[3])]
        else:
            merged.append(cur)
            cur = t
    merged.append(cur)
    return merged
