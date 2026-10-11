# -*- coding: utf-8 -*-
"""入口：批量处理输入文件夹中的 PDF。

流程（Step 1 阶段）：文本层探测 -> 数字版直取 / 扫描件 300 DPI 渲染。
后续 Step 2/3 会把渲染产物接入版面分析与 OCR。

用法:
    python run.py <输入文件夹> [输出文件夹]

输出结构（默认 output/）:
    output/<pdf名>/probe_report.json   逐页分类报告
    output/<pdf名>/text_layer/pNNN.txt 数字版页面文本
    output/<pdf名>/render/pNNN.png     待 OCR 页面的 300 DPI 渲染
    output/<pdf名>/ocr_text/pNNN.txt   版面+OCR+置信度分级后的文本
    output/<pdf名>/needs_human_review/ 低置信度/花体区域的截图
"""
import glob
import hashlib
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from text_layer_probe import probe_pdf, save_report, save_text_pages
from pdf_loader import render_pages

# 大批量扫描件的默认渲染 DPI（正文可读性与速度/体积的平衡点）
BATCH_RENDER_DPI = 200
# 默认并行线程数（每线程一份模型，质量模式约 1GB 内存/线程）
DEFAULT_WORKERS = 4

_engine_local = threading.local()


def pages_to_process(all_ocr_pages, page_range, existing):
    """计算待 OCR 页面列表。

    all_ocr_pages: 全部 ocr 页码（0 起始）
    page_range: (from, to) 1 起始闭区间，None 表示全部
    existing: 已存在输出（已处理过）的页码集合（断点续跑）
    返回按页码升序的待处理列表。
    """
    pages = all_ocr_pages
    if page_range:
        lo, hi = page_range
        if lo and hi:
            pages = [p for p in pages if lo <= p + 1 <= hi]
        elif lo:
            pages = [p for p in pages if p + 1 >= lo]
        elif hi:
            pages = [p for p in pages if p + 1 <= hi]
    if existing:
        pages = [p for p in pages if p not in existing]
    return sorted(pages)


def page_is_art(png_path, text_row_ratio=0.05, dark_ratio=0.60):
    """像素统计判定纯图页（毫秒级，不跑模型）。

    暗像素占比超过 dark_ratio（满版插图/深色广告页），或含文字的
    行占比低于 text_row_ratio（无正文）即视为纯图页，跳过 OCR。
    """
    import cv2
    import numpy as np
    img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    a = np.asarray(img, dtype=np.float32)
    dark = (a < 128).astype(np.float32)
    if dark.mean() > dark_ratio:
        return True
    rowden = dark.mean(axis=1)
    return float((rowden > 0.02).mean()) < text_row_ratio


def _purge_page_crops(out_dir, stem, pageno):
    """删除某页的旧截图（重跑该页前调用，防同名覆盖与僵尸图堆积）。"""
    import glob
    review_dir = os.path.join(out_dir, 'needs_human_review')
    for old in glob.glob(os.path.join(
            review_dir, f'{stem}_p{pageno:03d}_*.png')):
        try:
            os.remove(old)
        except OSError:
            pass


def _purge_stale_crops(out_dir, report):
    """整本处理完后，删除 report.json 与 corrections.json 都不引用的
    僵尸截图（历史崩溃/旧版残留）。"""
    import glob
    book_dir = out_dir
    review_dir = os.path.join(book_dir, 'needs_human_review')
    if not os.path.isdir(review_dir):
        return 0
    referenced = set()
    for it in report.get('review_items', []):
        crop = it.get('crop') or ''
        referenced.add(os.path.basename(crop))
    # corrections.json 里的校正引用也要保留
    try:
        corr_path = os.path.join(book_dir, 'corrections.json')
        if os.path.isfile(corr_path):
            for key in json.load(open(corr_path, encoding='utf-8')):
                referenced.add(os.path.basename(key))
    except (OSError, ValueError):
        pass
    removed = 0
    for png in glob.glob(os.path.join(review_dir, '*.png')):
        if os.path.basename(png) not in referenced:
            try:
                os.remove(png)
                removed += 1
            except OSError:
                pass
    return removed


def _resolve_out_dir(pdf_path, out_root, stem):
    """输出目录去冲突：同名但来源不同的 PDF 各进各的文件夹。

    同一文件重复处理 -> 原文件夹（断点续跑生效）；
    不同来源的同名文件 -> <书名>_<8位指纹> 文件夹（互不覆盖）。
    指纹 = 来源绝对路径 + 文件大小的 md5 前 8 位。
    """
    out_dir = os.path.join(out_root, stem)
    if os.path.isdir(out_dir):
        report_path = os.path.join(out_dir, 'probe_report.json')
        prev = None
        try:
            with open(report_path, encoding='utf-8') as f:
                prev = json.load(f).get('pdf')
        except (OSError, ValueError):
            prev = None
        cur = os.path.abspath(pdf_path)
        if prev and os.path.abspath(prev) != cur:
            try:
                size = str(os.path.getsize(pdf_path))
            except OSError:
                size = '0'
            h = hashlib.md5((cur + '|' + size).encode('utf-8')).hexdigest()[:8]
            out_dir = os.path.join(out_root, f'{stem}_{h}')
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _get_worker_engine(fast, intra_threads):
    """线程本地模型实例：每个工作线程第一页时懒加载自己的引擎。"""
    engine = getattr(_engine_local, 'engine', None)
    if engine is None:
        from ocr_engine import OcrEngine
        engine = _engine_local.engine = OcrEngine(fast=fast,
                                                  intra_threads=intra_threads)
    return engine


def process_one(pdf_path, out_root, fast=False, progress=None,
                page_range=None, resume=True, workers=DEFAULT_WORKERS,
                dpi=BATCH_RENDER_DPI):
    """处理单个 PDF。progress(stage, done, total) 为可选进度回调，
    stage ∈ {'probe','render','ocr'}。

    page_range: (from, to) 1 起始闭区间，None 处理全部页；
    resume: 跳过已有输出（ocr_text）的页面，中断后可续跑；
    workers: 并行 OCR 线程数（每线程一份模型，质量模式约 1GB 内存/线程）；
    dpi: 渲染分辨率（默认 200，平衡速度与体积）。
    中间渲染 PNG 在每页完成后即删（最终只留文本与复核截图）。
    """
    stem = os.path.splitext(os.path.basename(pdf_path))[0]
    out_dir = _resolve_out_dir(pdf_path, out_root, stem)
    if progress:
        progress('probe', 0, 1)

    result = probe_pdf(pdf_path)
    save_report(result, os.path.join(out_dir, 'probe_report.json'))
    n_text = save_text_pages(result, os.path.join(out_dir, 'text_layer'))
    all_ocr_pages = [i for i, p in result['pages'].items() if p['class'] == 'ocr']
    if progress:
        progress('probe', 1, 1)

    # 断点续跑：找出已完成的 ocr 页（ocr_text 文件已存在）
    ocr_dir = os.path.join(out_dir, 'ocr_text')
    os.makedirs(ocr_dir, exist_ok=True)
    done_ocr = {}
    if resume:
        for p in all_ocr_pages:
            txt_path = os.path.join(ocr_dir, f'{stem}_p{p:03d}.txt')
            if os.path.isfile(txt_path):
                done_ocr[p] = txt_path
    todo_pages = pages_to_process(all_ocr_pages, page_range, set(done_ocr))

    # 渲染待处理页
    render_dir = os.path.join(out_dir, 'render')
    os.makedirs(render_dir, exist_ok=True)
    render_pages(pdf_path, todo_pages, render_dir, dpi=dpi)

    # 纯图页毫秒级跳过（不 OCR，不产出复核项），删除其中间 PNG
    art_pages = set()
    for p in todo_pages:
        png = os.path.join(render_dir, f'{stem}_p{p:03d}.png')
        if page_is_art(png):
            art_pages.add(p)
            os.remove(png)

    # Step 3：版面分析 + OCR + 置信度分级（多线程并行，每页完成即删 PNG）
    ocr_results = {}
    if fast is not None:
        ocr_pages = [p for p in todo_pages if p not in art_pages]
        # 多线程时按 worker 数限每会话核数（实测 4 线程×3 核最优；
        # 单线程会话 93s / 不限核 77s 均更差）
        cores = os.cpu_count() or 4
        intra = max(1, cores // max(1, workers)) if workers > 1 else None
        done_lock = threading.Lock()
        done_count = [0]

        def work(pageno):
            png = os.path.join(render_dir, f'{stem}_p{pageno:03d}.png')
            # 重跑前清掉该页旧截图（续跑/重跑时避免同名覆盖与僵尸图）
            _purge_page_crops(out_dir, stem, pageno)
            try:
                page = _get_worker_engine(fast, intra).process(
                    png, out_dir, page_label=f'p{pageno:03d}')
            finally:
                if os.path.isfile(png):
                    os.remove(png)  # 中间图片用完即删
            with open(os.path.join(ocr_dir, f'{stem}_p{pageno:03d}.txt'),
                      'w', encoding='utf-8') as f:
                f.write(_page_text(page))
            with done_lock:
                done_count[0] += 1
                if progress:
                    progress('ocr', done_count[0], len(ocr_pages))
            return pageno, page

        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            futures = [ex.submit(work, p) for p in ocr_pages]
            for fut in as_completed(futures):
                pageno, page = fut.result()
                ocr_results[pageno] = page

    # 清空已空的 render 目录
    try:
        os.rmdir(render_dir)
    except OSError:
        pass

    # 续跑页的文本与复核项从既有输出恢复
    old_report_items = []
    if resume:
        report_path = os.path.join(out_dir, 'report.json')
        if os.path.isfile(report_path):
            try:
                with open(report_path, encoding='utf-8') as f:
                    old_report_items = json.load(f).get('review_items', [])
            except (OSError, ValueError):
                old_report_items = []
        for p, txt_path in sorted(done_ocr.items()):
            with open(txt_path, encoding='utf-8') as f:
                content = f.read()
            ocr_results[p] = {'blocks': [{'category': 'text', 'text': content}],
                              'needs_human_review': 0}

    # Step 5：整书文本 + 报告（续跑页的复核项并入新报告）
    from output_manager import assemble_text_md, build_report
    skip_reasons = {p: '跳过：纯图页/满版插图' for p in art_pages}
    skip_reasons.update({p: '（未处理：页码范围外或中断前未完成，可重跑续上）'
                         for p in (set(all_ocr_pages) - set(todo_pages)
                                   - set(done_ocr) - art_pages)})
    assemble_text_md(result, ocr_results,
                     os.path.join(out_dir, 'extracted_text.md'),
                     skipped=skip_reasons)
    report = build_report(result, ocr_results,
                          os.path.join(out_dir, 'report.json'),
                          extra_items=old_report_items)
    removed = _purge_stale_crops(out_dir, report)

    print(f"[{stem}] 共{result['page_count']}页 | 数字版={result['is_digital']} | "
          f"直取{len(n_text)}页 | 待OCR{len(all_ocr_pages)}页 | "
          f"本次OCR{len(todo_pages) - len(art_pages)}页"
          f"(跳纯图{len(art_pages)}页, 续跑跳过{len(done_ocr)}页, "
          f"{workers}线程) | 待人工复核{report['needs_human_review']}项 | "
          f"清僵尸截图{removed}张 -> {out_dir}")
    return result


def _page_text(layout):
    out = []
    for b in layout['blocks']:
        if b.get('text'):
            out.append(f'[{b["category"]}] {b["text"]}')
    return '\n\n'.join(out)


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        sys.exit(1)
    in_dir = argv[1]
    # 默认输出目录与 GUI 一致：项目根目录下的 output/
    out_root = argv[2] if len(argv) > 2 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'output')
    force = '--force' in argv
    pdfs = sorted(f for f in os.listdir(in_dir)
                  if f.lower().endswith('.pdf'))
    if not pdfs:
        print(f'输入文件夹中没有 PDF: {in_dir}')
        sys.exit(1)

    for f in pdfs:
        process_one(os.path.join(in_dir, f), out_root, resume=not force)
    print(f'完成，共处理 {len(pdfs)} 个 PDF，输出目录: {out_root}')


if __name__ == '__main__':
    main(sys.argv)
