# -*- coding: utf-8 -*-
"""Step 3 回归测试：OCR 引擎与防幻觉不变量（保留式标记语义）。

运行: python tests/test_step3.py
每页约 5-15 秒（快速模式），全量约 1-2 分钟。

防幻觉不变量（设计文档 v3 保留式标记）：
- 正文绝不断流：所有行原文都出现在输出中——ok 直出、
  doubtful 包 {?}、flagged 包 {!}（醒目可全文定位）；
- 不存在"行级占位符"（原文不再被挖走）；
- flagged 行必须有对应截图且文件真实存在（校对凭据）；
- 块级无 OCR 行时：文字样貌块才降级放占位符，图片区域静默跳过；
- 阅读顺序与 layout_analyzer 一致。
"""
import os
import re
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from ocr_engine import OcrEngine, CONF_HIGH, CONF_DOUBT

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'testset')


class TestStep3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = OcrEngine(fast=True)

    def process(self, rel):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        page = self.engine.process(os.path.join(BASE, rel), tmp)
        return page, tmp

    def test_scan_two_col_keeps_all_text(self):
        page, out_dir = self.process('samples/02_scan/vashtorr_p012.png')
        text_blocks = [b for b in page['blocks'] if b.get('lines')]
        self.assertGreaterEqual(len(text_blocks), 3, '扫描双栏页应有多个文本块')
        # 不变量 1：每条行原文都保留在输出中（正文不断流）
        for b in page['blocks']:
            for ln in b['lines']:
                self.assertIn(ln['text'], b['text'],
                              f'行原文丢失: {ln["text"][:30]}...')
        # 不变量 2：标记恰好包裹对应层级
        for b in page['blocks']:
            for ln in b['lines']:
                if ln['tier'] == 'doubtful':
                    self.assertIn('{?' + ln['text'] + '?}', b['text'])
                elif ln['tier'] == 'flagged':
                    self.assertIn('{!' + ln['text'] + '!}', b['text'])
                    self.assertTrue(ln.get('crop'), 'flagged 行应有截图凭据')
                    self.assertTrue(
                        os.path.exists(os.path.join(out_dir, ln['crop'])),
                        f"截图文件缺失: {ln['crop']}")
        # 不变量 3：行级占位符不再出现
        for b in page['blocks']:
            self.assertNotIn('[WAITING_FOR_HUMAN_INPUT', b['text'],
                             '行级内容不得被挖走')
        # 阅读顺序：order 连续
        orders = [b['order'] for b in page['blocks']]
        self.assertEqual(orders, sorted(orders))

    def test_clean_page_high_confidence(self):
        """干净数字渲染页应大部分行 >= 0.90。"""
        page, _ = self.process('samples/05_image_input/plot_15_twocol.png')
        lines = [ln for b in page['blocks'] for ln in b['lines']]
        self.assertGreaterEqual(len(lines), 20, '双栏页应有足够文本行')
        high = sum(1 for ln in lines if ln['score'] >= CONF_HIGH)
        self.assertGreaterEqual(high / len(lines), 0.7,
                                '干净页面大部分行应高置信度')

    def test_art_page_no_text(self):
        page, _ = self.process('samples/02_scan/vashtorr_p001.png')
        texts = [b['text'] for b in page['blocks'] if b.get('text')]
        self.assertLessEqual(len(texts), 1, '暗色插图页不应有大量文本输出')

    def test_block_placeholder_files_exist(self):
        """块级降级占位符（若存在）必须指向真实截图文件。"""
        page, out_dir = self.process('samples/02_scan/vashtorr_p012.png')
        pat = re.compile(r'\[WAITING_FOR_HUMAN_INPUT: ([^\]]+)\]')
        for b in page['blocks']:
            for m in pat.finditer(b['text']):
                self.assertTrue(
                    os.path.exists(os.path.join(out_dir, m.group(1))),
                    f"占位符指向的截图不存在: {m.group(1)}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
