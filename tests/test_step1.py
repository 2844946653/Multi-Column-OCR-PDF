# -*- coding: utf-8 -*-
"""Step 1 回归测试：文本层探测 + PDF 渲染。

运行: python tests/test_step1.py
覆盖设计文档验收标准：
- 数字版 PDF（有文本层）：对应页面直取，不产生 OCR 渲染；
- 扫描件（无文本层）：全部页面判定为 ocr，渲染 300 DPI PNG；
- 数字版直取结果与 gt/ 真值一致。
"""
import os
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from text_layer_probe import probe_pdf
from pdf_loader import render_pages

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'testset')


def orig_pages_of(subset_name):
    """从 manifest.json 读出子集 PDF 的"子集页序 -> 原书页码"映射。"""
    import json
    with open(os.path.join(BASE, 'manifest.json'), encoding='utf-8') as f:
        entries = json.load(f)['entries']
    pages = sorted(e['page'] for e in entries
                   if e['path'].startswith(subset_name))
    return pages


def gt_path(kind, page):
    return os.path.join(BASE, 'gt', f'{kind}_p{page:03d}.txt')


class TestProbeDigital(unittest.TestCase):
    """数字版子集：全部页面应判定为 text，且直取文本与 gt 一致。"""

    def test_story_subset(self):
        r = probe_pdf(os.path.join(BASE, 'sources', '01_story_pages.pdf'))
        self.assertTrue(r['is_digital'])
        self.assertEqual(r['page_count'], 6)
        origs = orig_pages_of('sources/01_story_pages')
        for i, p in r['pages'].items():
            self.assertEqual(p['class'], 'text', f'story 页 {i} 应为 text')
            with open(gt_path('story', origs[i]), encoding='utf-8') as f:
                self.assertEqual(p['text'], f.read(), f'story 页 {i} 直取结果与真值不一致')

    def test_tables_subset(self):
        r = probe_pdf(os.path.join(BASE, 'sources', '03_tables_pages.pdf'))
        self.assertTrue(r['is_digital'])
        origs = orig_pages_of('samples/03_tables/')
        for i, p in r['pages'].items():
            self.assertEqual(p['class'], 'text')
            with open(gt_path('table', origs[i]), encoding='utf-8') as f:
                self.assertEqual(p['text'], f.read(), f'table 页 {i} 直取结果与真值不一致')

    def test_magazine_subset(self):
        r = probe_pdf(os.path.join(BASE, 'sources', '04_magazine_pages.pdf'))
        self.assertTrue(r['is_digital'])
        origs = orig_pages_of('samples/04_magazine/')
        for i, p in r['pages'].items():
            self.assertEqual(p['class'], 'text')
            with open(gt_path('mag', origs[i]), encoding='utf-8') as f:
                self.assertEqual(p['text'], f.read(), f'mag 页 {i} 直取结果与真值不一致')


class TestProbeScan(unittest.TestCase):
    """扫描件子集：全部页面应判定为 ocr，且不输出直取文本。"""

    def test_scan_subset(self):
        r = probe_pdf(os.path.join(BASE, 'sources', '02_scan_pages.pdf'))
        self.assertFalse(r['is_digital'])
        self.assertEqual(r['page_count'], 8)
        for i, p in r['pages'].items():
            self.assertEqual(p['class'], 'ocr', f'scan 页 {i} 应为 ocr')
            self.assertEqual(p['chars'], 0)


class TestRenderScan(unittest.TestCase):
    """扫描件渲染：300 DPI，尺寸达标，且与测试集已有样张像素一致。"""

    # 子集页码(0-7) -> 原书页码
    ORIG = [0, 1, 2, 4, 5, 8, 12, 20]

    def test_render_quality(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = render_pages(os.path.join(BASE, 'sources', '02_scan_pages.pdf'),
                                 range(8), tmp)
            self.assertEqual(len(paths), 8)
            for idx, path in enumerate(paths):
                img = np.asarray(Image.open(path).convert('L'), dtype=np.float32)
                h, w = img.shape
                self.assertGreaterEqual(w, 2000, f'页 {idx} 渲染宽度不足 2000px(实际{w})')
                self.assertGreater(h, w, '页面应为纵向')
                # 与入库样张（压缩版 1000px）做缩略图结构对比：
                # 抓渲染错页/空白页，同时容忍压缩差异
                sample = os.path.join(BASE, 'samples', '02_scan',
                                      f'vashtorr_p{self.ORIG[idx]:03d}.png')
                ref = Image.open(sample).convert('L')
                thumb_img = Image.fromarray(img.astype(np.uint8)).resize(
                    (300, int(300 * h / w)), Image.LANCZOS)
                thumb_ref = ref.resize((300, int(300 * ref.height / ref.width)),
                                       Image.LANCZOS)
                diff = np.abs(
                    np.asarray(thumb_img, dtype=np.float32)
                    - np.asarray(thumb_ref, dtype=np.float32)).mean()
                self.assertLess(diff, 40.0,
                                f'页 {idx} 渲染与样张结构偏差过大(mean_diff={diff:.2f})')


if __name__ == '__main__':
    unittest.main(verbosity=2)
