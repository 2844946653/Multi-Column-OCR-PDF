# -*- coding: utf-8 -*-
"""模块一：文本层探测。

逐页判断 PDF 页面是否有可用的文本层：
- 文档级判定：整本 PDF 的"中位数页字符数"达标，判定为数字版；
- 页级分类：每页输出 'text'（直取文本层）或 'ocr'（需渲染后识别）。

规则（对应设计文档第 3 节模块一）：
1. 字符数为 0 的页（纯图片页）一律 'ocr'——即使整本是数字版；
2. 数字版其余页面一律 'text'（艺术页带少量说明文字也算，直取即可）；
3. 非数字版（扫描件）按单页阈值 PAGE_TEXT_THRESHOLD 兜底：部分扫描件带
   稀疏 OCR 层，字符数过少的页仍走 'ocr' 流程。

已知限制：整本都带密集 OCR 层的扫描件（如部分 WD 杂志）会被误判为数字版，
其 OCR 层文本会被直接提取。report 中保留逐页字符数，便于人工发现此类情况。
"""
import json
import os

import pymupdf

# 单页文本量达标线（字符数），仅用于非数字版文档的兜底判定
PAGE_TEXT_THRESHOLD = 50
# 文档级判定线：中位数页字符数达到该值即视为数字版
DOC_TEXT_THRESHOLD = 50


def probe_pdf(pdf_path):
    """探测 PDF 文本层。

    返回 dict:
        pdf: 文件路径
        page_count: 总页数
        is_digital: 文档级判定结果
        pages: {页码(int): {'class': 'text'|'ocr', 'chars': 字符数, 'text': 原文}}
    """
    doc = pymupdf.open(pdf_path)
    n = doc.page_count
    char_counts = []
    raw_texts = {}
    for i in range(n):
        t = doc[i].get_text('text', sort=True)
        raw_texts[i] = t
        char_counts.append(len(t.strip()))
    doc.close()

    mid = sorted(char_counts)[n // 2]
    is_digital = mid >= DOC_TEXT_THRESHOLD

    pages = {}
    for i, c in enumerate(char_counts):
        if c == 0:
            cls = 'ocr'  # 纯图片页，即使整本是数字版
        elif is_digital:
            cls = 'text'
        else:
            cls = 'text' if c >= PAGE_TEXT_THRESHOLD else 'ocr'
        pages[i] = {'class': cls, 'chars': c, 'text': raw_texts[i]}

    return {'pdf': pdf_path, 'page_count': n,
            'is_digital': is_digital, 'pages': pages}


def save_report(result, report_path):
    """把探测结果（不含文本正文）写入 JSON 报告。"""
    slim = {'pdf': result['pdf'], 'page_count': result['page_count'],
            'is_digital': result['is_digital'],
            'pages': {i: {'class': p['class'], 'chars': p['chars']}
                      for i, p in result['pages'].items()}}
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(slim, f, ensure_ascii=False, indent=2)


def save_text_pages(result, out_dir):
    """把 'text' 类页面的原文写到 out_dir/<页码>.txt，返回保存的文件列表。"""
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(result['pdf']))[0]
    saved = []
    for i, p in result['pages'].items():
        if p['class'] == 'text':
            path = os.path.join(out_dir, f'{stem}_p{i:03d}.txt')
            with open(path, 'w', encoding='utf-8') as f:
                f.write(p['text'])
            saved.append(path)
    return saved
