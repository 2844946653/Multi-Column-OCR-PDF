# -*- coding: utf-8 -*-
"""桌面 GUI 后端 API（pywebview js_api 桥）。

前端 JS 通过 window.pywebview.api.<方法名> 调用。
职责：文件夹选择、后台提取线程与进度轮询、结果读取、
人工复核项读取（截图转 base64）与校正保存/导出。
"""
import base64
import json
import os
import threading

import run as run_mod
from output_manager import (apply_corrections, load_corrections,
                            save_corrections)

_STATE_IDLE = {'running': False, 'stage': 'idle', 'message': '就绪',
               'done': 0, 'total': 1, 'error': None}


def _detect_default_output_dir():
    """默认输出目录 = 桌面上的总文件夹。

    桌面路径探测兼容：普通桌面 / OneDrive 重定向 / 中文系统"桌面"，
    都不存在则退回用户主目录。
    """
    home = os.path.expanduser('~')
    for c in (os.path.join(home, 'Desktop'),
              os.path.join(home, 'OneDrive', 'Desktop'),
              os.path.join(home, '桌面')):
        if os.path.isdir(c):
            return os.path.join(c, 'Multi-Column-OCR输出')
    return os.path.join(home, 'Multi-Column-OCR输出')


def _expand_inputs(input_paths):
    """把输入列表展开为 PDF 文件路径列表：文件直接保留，文件夹取其中全部 PDF。"""
    pdfs = []
    seen = set()
    for p in input_paths or []:
        if os.path.isdir(p):
            for f in sorted(os.listdir(p)):
                if f.lower().endswith('.pdf'):
                    path = os.path.join(p, f)
                    if path not in seen:
                        seen.add(path)
                        pdfs.append(path)
        elif os.path.isfile(p) and p.lower().endswith('.pdf'):
            if p not in seen:
                seen.add(p)
                pdfs.append(p)
    return pdfs


class GuiApi:
    def __init__(self):
        self._window = None
        self.engine = None  # OcrEngine 惰性加载（模型加载约 1-2 分钟）
        self.state = dict(_STATE_IDLE)
        self._thread = None
        self.default_input_dir = os.path.expanduser('~')
        # 默认输出到桌面上的总文件夹（用户好找，且不会每本书撒一个文件夹）
        self.default_output_dir = _detect_default_output_dir()

    def set_window(self, window):
        self._window = window

    def get_default_output_dir(self):
        """前端启动时调用，把真实默认路径显示在输出栏。"""
        return self.default_output_dir

    def _resolve_dir(self, output_dir):
        """前端传空字符串表示"默认输出目录"——空目录会被 isdir 短路，
        导致结果页/复核页拿不到任何数据（曾出过的 bug）。"""
        if output_dir and os.path.isdir(output_dir):
            return output_dir
        return self.default_output_dir

    # ---------- 输入选择 ----------
    def pick_input_files(self):
        """多选 PDF 文件，返回路径列表。"""
        from webview import OPEN_DIALOG
        res = self._window.create_file_dialog(
            OPEN_DIALOG, allow_multiple=True,
            file_types=('PDF 文件 (*.pdf)',))
        files = [p for p in (res or ()) if p.lower().endswith('.pdf')]
        if files:
            self.default_input_files = files
        return getattr(self, 'default_input_files', [])

    def pick_input_dir(self):
        from webview import FOLDER_DIALOG
        res = self._window.create_file_dialog(FOLDER_DIALOG)
        if res and res[0]:
            self.default_input_dir = res[0]
        return self.default_input_dir

    def pick_output_dir(self):
        from webview import FOLDER_DIALOG
        res = self._window.create_file_dialog(FOLDER_DIALOG)
        if res and res[0]:
            self.default_output_dir = res[0]
        return self.default_output_dir

    # ---------- 提取 ----------
    def start_extract(self, input_paths, output_dir, page_from=None,
                      page_to=None, fast=True, workers=4, force=False):
        """input_paths: PDF 文件路径或文件夹路径的列表。逐个顺序处理，
        每本书输出到 output/<PDF名>/ 子目录，互不合并。
        page_from/page_to: 1 起始页码范围（闭区间，空=全部）；
        fast: 快速模式（轻量模型）；workers: 并行 OCR 线程数；
        force: 强制重新处理（忽略已完成页，重跑后旧复核截图会被清扫）；
        断点续跑默认开启；中间渲染图用完即删。
        """
        if self.state['running']:
            return False
        output_dir = self._resolve_dir(output_dir)  # 空=默认输出目录
        pdfs = _expand_inputs(input_paths)
        if not pdfs:
            self.state = dict(_STATE_IDLE)
            self.state.update(message='完成', error='没有找到 PDF 文件（只支持 .pdf）')
            return False
        page_range = None
        if page_from and page_to and page_from <= page_to:
            page_range = (int(page_from), int(page_to))
        elif page_from:
            page_range = (int(page_from), int(page_from))
        self.state = dict(_STATE_IDLE)
        msg = f'正在加载模型（首次运行约 1-2 分钟），共 {len(pdfs)} 个 PDF，' \
              f'{max(1, int(workers))} 线程'
        if page_range:
            msg += f'，页码范围 {page_range[0]}-{page_range[1]}'
        if force:
            msg += '，强制重新处理'
        self.state.update(running=True, stage='loading', message=msg + '…')
        self._thread = threading.Thread(
            target=self._extract_worker,
            args=(pdfs, output_dir, page_range, bool(fast), int(workers),
                  bool(force)),
            daemon=True)
        self._thread.start()
        return True

    def _extract_worker(self, pdfs, output_dir, page_range, fast, workers,
                        force):
        try:
            for idx, path in enumerate(pdfs):
                self.state.update(
                    running=True, stage='book', done=idx, total=len(pdfs),
                    message=f'[{idx+1}/{len(pdfs)}] {os.path.basename(path)}')
                run_mod.process_one(path, output_dir, fast=fast,
                                    progress=self._progress,
                                    page_range=page_range,
                                    workers=max(1, workers),
                                    resume=not force)
            self.state.update(running=False, stage='idle', done=1, total=1,
                              message=f'完成：{len(pdfs)} 个 PDF 已处理，'
                                      f'输出目录：{output_dir}')
        except Exception as e:  # 界面要能看到错误
            self.state.update(running=False, stage='idle',
                              message='出错', error=f'{type(e).__name__}: {e}')

    def _progress(self, stage, done, total):
        if stage == 'ocr':
            self.state.update(stage='ocr', done=done + 1, total=total,
                              message=f'OCR 处理中 {done + 1}/{total} 页…')
        elif stage == 'probe':
            self.state.update(stage='probe', done=done, total=1,
                              message='探测文本层…')

    def get_state(self):
        return dict(self.state)

    # ---------- 结果浏览 ----------
    def list_books(self, output_dir):
        output_dir = self._resolve_dir(output_dir)
        if not os.path.isdir(output_dir):
            return []
        books = []
        for name in sorted(os.listdir(output_dir)):
            report_path = os.path.join(output_dir, name, 'report.json')
            if not os.path.isfile(report_path):
                continue
            try:
                with open(report_path, encoding='utf-8') as f:
                    r = json.load(f)
            except (OSError, ValueError):
                continue
            books.append({
                'stem': name,
                'page_count': r.get('page_count'),
                'text_layer_pages': r.get('text_layer_pages'),
                'ocr_pages': r.get('ocr_pages'),
                'needs_human_review': r.get('needs_human_review'),
                'corrected': os.path.isfile(
                    os.path.join(output_dir, name, 'corrected_text.md')),
            })
        return books

    def get_book_text(self, output_dir, stem):
        output_dir = self._resolve_dir(output_dir)
        path = os.path.join(output_dir, stem, 'extracted_text.md')
        if not os.path.isfile(path):
            return ''
        with open(path, encoding='utf-8') as f:
            return f.read()

    def get_book_corrected(self, output_dir, stem):
        output_dir = self._resolve_dir(output_dir)
        path = os.path.join(output_dir, stem, 'corrected_text.md')
        if not os.path.isfile(path):
            return None
        with open(path, encoding='utf-8') as f:
            return f.read()

    # ---------- 人工复核 ----------
    def get_review_items(self, output_dir, stem):
        output_dir = self._resolve_dir(output_dir)
        book_dir = os.path.join(output_dir, stem)
        report_path = os.path.join(book_dir, 'report.json')
        if not os.path.isfile(report_path):
            return []
        with open(report_path, encoding='utf-8') as f:
            report = json.load(f)
        corrections = load_corrections(book_dir)
        items = []
        for it in report.get('review_items', []):
            crop = it.get('crop') or ''
            crop_path = os.path.join(book_dir, crop.replace('/', os.sep))
            image = None
            if crop and os.path.isfile(crop_path):
                with open(crop_path, 'rb') as f:
                    image = base64.b64encode(f.read()).decode('ascii')
            items.append({
                'page': it.get('page'),
                'category': it.get('category'),
                'bbox': it.get('bbox'),
                'score': it.get('score'),
                'reason': it.get('reason'),
                'crop': crop,
                'image': 'data:image/png;base64,' + image if image else None,
                'correction': corrections.get(crop, ''),
            })
        return items

    def save_corrections(self, output_dir, stem, corrections):
        """corrections: {crop 相对路径: 校正文本}"""
        output_dir = self._resolve_dir(output_dir)
        book_dir = os.path.join(output_dir, stem)
        save_corrections(book_dir, corrections)
        return len(corrections)

    def export_corrected(self, output_dir, stem):
        """应用校正，导出 corrected_text.md。返回 (路径, 替换数)。"""
        output_dir = self._resolve_dir(output_dir)
        book_dir = os.path.join(output_dir, stem)
        path, replaced = apply_corrections(book_dir)
        return {'path': path, 'replaced': replaced}
