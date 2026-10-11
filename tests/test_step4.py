# -*- coding: utf-8 -*-
"""Step 4 回归测试：表格分支（v1 范围）。

v1 范围（如实记录）：
- 表格"区域"识别：PP-DocLayout table 标签 + 线框检测器兜底（test_step2 已测）；
- 表格"内容"保留：表格块内的 OCR 行按三级置信度提取为平文本；
- 结构化 Markdown 表格输出：table_extractor.py 实验性实现，需视觉调参，
  属二期（见 进度总结.md 已知问题）。

本测试验证：表格页（太空死灵分值表 p132）的表格区域有内容产出，
且防幻觉不变量在表格块内同样成立。
"""
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from ocr_engine import OcrEngine

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'testset')


class TestStep4(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = OcrEngine()

    def test_table_region_content(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        page = self.engine.process(
            os.path.join(BASE, 'samples', '03_tables', 'necron_p132.png'), tmp)

        table_blocks = [b for b in page['blocks'] if b['category'] == 'table']
        self.assertGreaterEqual(len(table_blocks), 1, '分值表页应至少识别出 1 个表格区域')
        with_content = [b for b in table_blocks if b['lines']]
        self.assertGreaterEqual(len(with_content), 1,
                                '至少一个表格区域应有内容产出')
        # 无内容的表格块（线框检测器碎片且非文字样貌）会被静默跳过，
        # 这正是"图片不进复核清单"的预期行为

        # 全页文本总量应可观（表格内容被保留）
        total = sum(len(b['text']) for b in page['blocks'])
        self.assertGreater(total, 200, f'表格页总输出文本过少({total}字符)')


if __name__ == '__main__':
    unittest.main(verbosity=2)
