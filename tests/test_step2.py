# -*- coding: utf-8 -*-
"""Step 2 回归测试：版面分析（区域分类 + 阅读顺序）。

运行: python tests/test_step2.py
注意：PP-StructureV3 含整页 OCR，每页推理约 1-3 分钟（CPU），全量约 10-15 分钟。

覆盖设计文档验收标准：
- 双栏页：先完整读左栏再读右栏，绝不左右串行（栏序不回落）
- 跨栏标题/侧边栏/图片：作为独立单元按其 y 位置插入阅读流
- 暗色插图页：不产出正文文本块
- 栏检测尺度无关：大画布页（圣典）也能正确分栏
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from layout_analyzer import LayoutAnalyzer

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'testset')


def no_col_fallback(blocks):
    """核心性质：阅读顺序按"带"推进——独立单元（栏外块）划分带边界，
    每个带内正文块的栏序号绝不回落（先左栏后右栏，不左右串行）。"""
    last_col = None
    for b in blocks:
        if b['column'] is None:
            last_col = None  # 带边界，允许左栏重新开始
            continue
        if last_col is not None and b['column'] < last_col:
            return False
        last_col = b['column']
    return True


def col_blocks_y_ascending(blocks, col):
    """同栏正文块按 y 严格递增。"""
    ys = [b['bbox'][1] for b in blocks if b['column'] == col]
    return ys == sorted(ys)


class TestStep2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 模型加载较慢（约 1-2 分钟），全类共享一个分析器
        cls.an = LayoutAnalyzer()

    def test_scan_two_col(self):
        r = self.an.analyze(os.path.join(BASE, 'samples', '02_scan', 'vashtorr_p012.png'))
        self.assertGreaterEqual(len(r['columns']), 2, '扫描双栏页应检测出至少 2 栏')
        text_blocks = [b for b in r['blocks'] if b['column'] is not None]
        self.assertGreaterEqual(len(text_blocks), 5, '应有足够正文块')
        self.assertTrue(no_col_fallback(r['blocks']), '阅读顺序不应左右串行')

    def test_image_input_two_col(self):
        r = self.an.analyze(os.path.join(BASE, 'samples', '05_image_input',
                                         'plot_15_twocol.png'))
        self.assertEqual(len(r['columns']), 2)
        cats = {b['category'] for b in r['blocks']}
        self.assertIn('image', cats, '应识别出图片区域')
        self.assertTrue(no_col_fallback(r['blocks']))
        for c in range(2):
            self.assertTrue(col_blocks_y_ascending(r['blocks'], c),
                            f'第 {c} 栏正文应按 y 升序')

    def test_art_page_no_text(self):
        r = self.an.analyze(os.path.join(BASE, 'samples', '02_scan', 'vashtorr_p001.png'))
        text_blocks = [b for b in r['blocks'] if b['category'] == 'text']
        self.assertEqual(len(text_blocks), 0, '暗色插图页不应产出正文文本块')

    def test_magazine_side_columns(self):
        r = self.an.analyze(os.path.join(BASE, 'samples', '04_magazine', 'wd487_p112.png'))
        self.assertGreaterEqual(len(r['columns']), 2, '杂志页应检测出多栏')
        cats = {b['category'] for b in r['blocks']}
        self.assertIn('image', cats, '应有图片区域')
        self.assertIn('side', cats, '应有侧边栏区域')
        self.assertTrue(no_col_fallback(r['blocks']))

    def test_codex_canvas_columns(self):
        r = self.an.analyze(os.path.join(BASE, 'samples', '06_canvas', 'codex_p010.png'))
        self.assertGreaterEqual(len(r['columns']), 2, '大画布页应检测出多栏')
        # 栏范围不得重叠
        for a, b in zip(r['columns'], r['columns'][1:]):
            self.assertLessEqual(a[1], b[0], f'栏范围重叠: {a} vs {b}')
        self.assertTrue(no_col_fallback(r['blocks']))

    def test_table_detection(self):
        """太空死灵分值表页应检出 table 区域。"""
        r = self.an.analyze(os.path.join(BASE, 'samples', '03_tables', 'necron_p132.png'))
        cats = {b['category'] for b in r['blocks']}
        self.assertIn('table', cats, '分值表页应识别出表格区域')


if __name__ == '__main__':
    unittest.main(verbosity=2)
