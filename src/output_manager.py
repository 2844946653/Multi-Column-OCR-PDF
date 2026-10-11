# -*- coding: utf-8 -*-
"""模块五：结果输出与归档。

1. extracted_text.md：数字版直取文本与 OCR 文本按页码和逻辑顺序拼接；
2. report.json：总页数、文本层直取页数、OCR 页数、待人工复核数，
   以及每个待复核区域的页码/坐标/类别/置信度（便于人工快速定位）。
低置信度截图本身由 ocr_engine 写入 needs_human_review/。
"""
import json
import os

REVIEW_PLACEHOLDER = '[WAITING_FOR_HUMAN_INPUT: '

_CORRECTIONS_FILE = 'corrections.json'


def load_corrections(book_dir):
    """读取人工复核校正 {crop 相对路径: 校正文本}。"""
    path = os.path.join(book_dir, _CORRECTIONS_FILE)
    if not os.path.isfile(path):
        return {}
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if v and v.strip()}


def save_corrections(book_dir, corrections):
    """保存人工复核校正（合并已有条目，保留非空值）。"""
    merged = load_corrections(book_dir)
    merged.update(corrections)
    with open(os.path.join(book_dir, _CORRECTIONS_FILE), 'w',
              encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    return merged


def apply_corrections(book_dir):
    """把校正文本回填进 extracted_text.md，生成 corrected_text.md。

    占位符 [WAITING_FOR_HUMAN_INPUT: <crop>] 被替换为对应校正文本；
    无校正的占位符保留原样（继续等人工）。返回 (输出路径, 替换数)。
    """
    src = os.path.join(book_dir, 'extracted_text.md')
    if not os.path.isfile(src):
        return '', 0
    with open(src, encoding='utf-8') as f:
        text = f.read()
    corrections = load_corrections(book_dir)
    replaced = 0
    for crop, value in corrections.items():
        placeholder = f'[WAITING_FOR_HUMAN_INPUT: {crop}]'
        count = text.count(placeholder)
        text = text.replace(placeholder, value)
        replaced += count
    out = os.path.join(book_dir, 'corrected_text.md')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(text)
    return out, replaced


def assemble_text_md(probe, ocr_results, out_path, skipped=None):
    """按页码拼接整本书的文本输出。

    参数:
        probe: text_layer_probe.probe_pdf 的返回
        ocr_results: {页码: ocr_engine.process 的页面结果}（仅 ocr 页）
        out_path: extracted_text.md 输出路径
        skipped: 未处理的 ocr 页码 -> 说明文字（纯图页跳过/范围外/中断），
                 正文放对应说明占位
    """
    skipped = skipped or {}
    parts = []
    for pageno in range(probe['page_count']):
        p = probe['pages'][pageno]
        if p['class'] == 'text':
            content, kind = p['text'], '文本层直取'
        elif pageno in skipped:
            content = skipped[pageno]
            kind = 'OCR'
        else:
            layout = ocr_results.get(pageno)
            content = _page_text(layout) if layout else ''
            kind = 'OCR'
        parts.append(f'## 第 {pageno + 1} 页（{kind}）\n\n{content}')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(parts))
    return out_path


def _page_text(layout):
    """单页 layout -> 文本（阅读顺序，块类别标注，空块跳过）。"""
    out = []
    for b in layout['blocks']:
        if b.get('text'):
            out.append(f'[{b["category"]}] {b["text"]}')
    return '\n\n'.join(out)


def build_report(probe, ocr_results, out_path, extra_items=None):
    """生成 report.json。

    review_items 中每条待人工复核区域记录：
        page(页码,1起始), category(区块类别), bbox(坐标),
        score(置信度, 整块降级时为 None), crop(截图相对路径)
    extra_items: 续跑时从既有 report.json 恢复的复核项（按 page+crop 去重）。
    """
    n_text = sum(1 for p in probe['pages'].values() if p['class'] == 'text')
    n_ocr = probe['page_count'] - n_text
    review_items = []
    for pageno, layout in sorted(ocr_results.items()):
        for b in layout['blocks']:
            if b.get('review_block'):
                review_items.append({
                    'page': pageno + 1, 'category': b['category'],
                    'bbox': b['bbox'], 'score': None,
                    'crop': b['review_block'],
                    'reason': '整块无 OCR 行（花体/漫画气泡等检测失败）'})
            for ln in b.get('lines', []):
                if ln.get('tier') == 'flagged':
                    review_items.append({
                        'page': pageno + 1, 'category': b['category'],
                        'bbox': ln['bbox'], 'score': ln['score'],
                        'crop': ln.get('crop'),
                        'reason': '置信度低于阈值（原文已保留并标 {!}!，可忽略）'})
    seen = {(it['page'], it.get('crop')) for it in review_items}
    for it in extra_items or []:
        key = (it.get('page'), it.get('crop'))
        if key not in seen:
            review_items.append(it)
            seen.add(key)
    review_items.sort(key=lambda it: (it['page'], it.get('crop') or ''))

    report = {
        'pdf': os.path.basename(probe['pdf']),
        'page_count': probe['page_count'],
        'is_digital': probe['is_digital'],
        'text_layer_pages': n_text,
        'ocr_pages': n_ocr,
        'needs_human_review': len(review_items),
        'review_items': review_items,
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report
