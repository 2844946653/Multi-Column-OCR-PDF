# -*- coding: utf-8 -*-
"""模块二：文档预处理（PDF 渲染）。

仅把探测为 'ocr' 的页面按 300 DPI 渲染为 PNG，供后续版面分析与 OCR 使用。
数字版页面不走本模块（文本层直取，见 text_layer_probe.py）。
"""
import os

import pymupdf

RENDER_DPI = 300


def render_pages(pdf_path, page_numbers, out_dir, dpi=RENDER_DPI):
    """渲染指定页为 PNG。

    参数:
        pdf_path: 源 PDF 路径
        page_numbers: 待渲染的页码列表（0 起始）
        out_dir: 输出目录
        dpi: 渲染分辨率（默认 300；大批量扫描件可用 200 提速减重）
    返回: 生成的 PNG 路径列表，命名 <pdf名>_p<页码3位>.png
    """
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(pdf_path))[0]
    doc = pymupdf.open(pdf_path)
    saved = []
    for p in page_numbers:
        path = os.path.join(out_dir, f'{stem}_p{p:03d}.png')
        doc[p].get_pixmap(dpi=dpi).save(path)
        saved.append(path)
    doc.close()
    return saved
