# -*- coding: utf-8 -*-
"""模块四的表格分支：区域网格 -> 单元格 -> 内容填充 -> Markdown 表格。

背景：PP-Structure 表格模型（SLANet 等）的 ONNX 包在官方源不可用（404），
且本机 paddle 原生推理有 bug，故改用确定性混合方法：
- 行线：区域局部极性下的形态学横线检测（规则书表格行线实线可靠）；
- 列分隔：优先图像竖线；失败时用内容（OCR 行/文本层词）的 x 聚类——
  点状引导线（如分值表的 ".........."）在内容 x 间隙中自然显现；
- 单元格内容：OCR 行或文本层词（中心点落格）。
无线框表格留待二期。
"""
import cv2
import numpy as np

# 行线在区域内的相对跨度阈值（容忍断线）
_ROW_SPAN_RATIO = 0.25
# 列线在区域内的相对跨度阈值
_COL_SPAN_RATIO = 0.5
# 合并近邻线位置（像素）
_LINE_MERGE_GAP = 8
# 内容 x 聚类的最小间隙（占区域宽比例）
_COL_GAP_RATIO = 0.06
# 单元格内缩（避免把线像素算进内容）
_CELL_PAD = 8


def extract_table_region(img_path, ocr_lines, region, words=None):
    """在区域 bbox 内提取表格。

    参数:
        img_path: 页面图像路径
        ocr_lines: 整页 OCR 行 [{'text','score','bbox'}, ...]（原图坐标）
        region: [x0, y0, x1, y1]（原图坐标）
        words: 可选，文本层词 [(x0,y0,x1,y1,text), ...]（原图坐标，数字版用，
               列分隔与内容填充优先用词）
    返回:
        (markdown, bbox) 或 (None, None)
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None
    x0, y0, x1, y1 = region
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(img.shape[1], int(x1)), min(img.shape[0], int(y1))
    if x1 - x0 < 80 or y1 - y0 < 80:
        return None, None
    crop = img[y0:y1, x0:x1]
    # 区域局部极性：深色底板（规则书深色表格）上线条是亮的
    bw = cv2.adaptiveThreshold(crop, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY, 31, -4)
    lines = 255 - bw if crop.mean() > 128 else bw
    h_lines = cv2.morphologyEx(lines, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (60, 1)))
    v_lines = cv2.morphologyEx(lines, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (1, 60)))

    rows = _line_positions(h_lines, axis='h',
                           span_min=(x1 - x0) * _ROW_SPAN_RATIO)
    if len(rows) < 3:
        return None, None  # 行线不足：不是线框表格
    cols = _line_positions(v_lines, axis='v',
                           span_min=(y1 - y0) * _COL_SPAN_RATIO)
    if len(cols) >= 2:
        cols_o = [x0 + c for c in cols]  # 图像竖线：相对区域 -> 绝对
    else:
        cols_o = _cols_from_content(words if words else ocr_lines, x0, x1)
        if len(cols_o) < 3:
            return None, None  # 列分隔缺失：交给平文本降级

    rows_o = [y0 + r for r in rows]

    # 内容源：词优先（数字版），否则 OCR 行
    items = []
    if words:
        items = [{'text': w[4], 'bbox': list(w[:4])} for w in words]
    else:
        items = [{'text': ln['text'], 'bbox': ln['bbox']} for ln in ocr_lines]

    md_rows = []
    for i in range(len(rows_o) - 1):
        cells = []
        for j in range(len(cols_o) - 1):
            cell = (cols_o[j] + _CELL_PAD, rows_o[i] + _CELL_PAD,
                    cols_o[j + 1] - _CELL_PAD, rows_o[i + 1] - _CELL_PAD)
            if cell[2] <= cell[0] or cell[3] <= cell[1]:
                cells.append('')
                continue
            texts = [it['text'] for it in items
                     if _center_in(it['bbox'], cell)]
            cells.append(' '.join(texts))
        if any(cells):
            md_rows.append(cells)
    if len(md_rows) < 2:
        return None, None

    md = _to_markdown(md_rows)
    bbox = [cols_o[0], rows_o[0], cols_o[-1], rows_o[-1]]
    return md, bbox


def _cols_from_content(items, x0, x1):
    """按内容 x 中心聚类切列，返回绝对坐标的列边界（含区域左右缘）。

    点状引导线（如分值表的 ".........."）在内容 x 间隙中自然显现。
    """
    centers = sorted((it['bbox'][0] + it['bbox'][2]) / 2 for it in items)
    if len(centers) < 4:
        return []
    gap = (x1 - x0) * _COL_GAP_RATIO
    clusters, cur = [], [centers[0]]
    for c in centers[1:]:
        if c - cur[-1] > gap:
            clusters.append(cur)
            cur = [c]
        else:
            cur.append(c)
    clusters.append(cur)
    bounds = []  # 每个簇的 [lo, hi]
    for cl in clusters:
        if len(cl) < 2:
            continue
        lo = min(it['bbox'][0] for it in items
                 if (it['bbox'][0] + it['bbox'][2]) / 2 in cl)
        hi = max(it['bbox'][2] for it in items
                 if (it['bbox'][0] + it['bbox'][2]) / 2 in cl)
        bounds.append((lo, hi))
    if len(bounds) < 2:
        return []
    boundaries = [x0]  # 区域左缘
    for k in range(len(bounds) - 1):
        boundaries.append((bounds[k][1] + bounds[k + 1][0]) / 2)
    boundaries.append(x1)  # 区域右缘
    return boundaries


def _line_positions(mask, axis, span_min):
    """提取线位置。mask 中线=255。返回相对 mask 顶部/左缘的位置列表。"""
    if axis == 'h':
        proj = mask.sum(axis=1) / 255.0  # 白像素个数
    else:
        proj = mask.sum(axis=0) / 255.0
    on = proj >= span_min
    positions, in_run = [], False
    for idx, v in enumerate(on):
        if v and not in_run:
            start = idx
            in_run = True
        elif not v and in_run:
            positions.append((start + idx - 1) / 2)
            in_run = False
    if in_run:
        positions.append((start + len(on) - 1) / 2)
    return _merge_positions(positions)


def _merge_positions(positions):
    """合并相距过近的线位置（同一粗线被拆成多条时）。"""
    if len(positions) <= 1:
        return positions
    merged, cur = [], [positions[0]]
    for p in positions[1:]:
        if p - cur[-1] <= _LINE_MERGE_GAP:
            cur.append(p)
        else:
            merged.append(sum(cur) / len(cur))
            cur = [p]
    merged.append(sum(cur) / len(cur))
    return merged


def _center_in(bbox, outer):
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def _to_markdown(rows):
    """行列表 -> Markdown 表格字符串。第一行作表头。"""
    def esc(s):
        return s.replace('|', '/').strip()
    header = '| ' + ' | '.join(esc(c) or ' ' for c in rows[0]) + ' |'
    sep = '| ' + ' | '.join('---' for _ in rows[0]) + ' |'
    body = ['| ' + ' | '.join(esc(c) or ' ' for c in r) + ' |' for r in rows[1:]]
    return '\n'.join([header, sep] + body)
