# -*- coding: utf-8 -*-
"""Step 5 回归测试：输出归档（用合成 fixture，不依赖模型，秒级）。

覆盖设计文档验收标准：
- extracted_text.md 按页码拼接，数字版直取页与 OCR 页内容正确归类；
- report.json 记录总页数/直取页数/OCR 页数/待复核数；
- 每个待复核区域记录页码、坐标、类别、置信度。
"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from output_manager import assemble_text_md, build_report


def fake_probe():
    return {
        'pdf': 'demo.pdf',
        'page_count': 3,
        'is_digital': True,
        'pages': {
            0: {'class': 'text', 'chars': 100, 'text': '第一页数字版文本。'},
            1: {'class': 'text', 'chars': 80, 'text': '第二页数字版文本。'},
            2: {'class': 'ocr', 'chars': 0, 'text': ''},
        },
    }


def fake_ocr_results():
    return {
        2: {
            'blocks': [
                {'order': 0, 'category': 'text', 'column': 0,
                 'bbox': [10, 10, 100, 40],
                 'lines': [
                     {'text': '高置信度行', 'score': 0.96, 'tier': 'ok',
                      'bbox': [10, 10, 100, 30]},
                     {'text': '低置信度原文保留', 'score': 0.55,
                      'tier': 'flagged', 'bbox': [10, 50, 100, 70],
                      'crop': 'needs_human_review/demo_p002_000.png'},
                 ],
                 'text': '高置信度行\n{!低置信度原文保留!}'},
                {'order': 1, 'category': 'side', 'column': None,
                 'bbox': [200, 10, 300, 60], 'lines': [],
                 'review_block': 'needs_human_review/demo_p002_001.png',
                 'text': '[WAITING_FOR_HUMAN_INPUT: '
                         'needs_human_review/demo_p002_001.png]'},
            ],
            'needs_human_review': 2,
        }
    }


class TestStep5(unittest.TestCase):
    def test_assemble_md(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = assemble_text_md(fake_probe(), fake_ocr_results(),
                                    os.path.join(tmp, 'extracted_text.md'))
            with open(path, encoding='utf-8') as f:
                md = f.read()
            self.assertIn('## 第 1 页（文本层直取）', md)
            self.assertIn('第一页数字版文本。', md)
            self.assertIn('## 第 3 页（OCR）', md)
            self.assertIn('高置信度行', md)
            self.assertIn('{!低置信度原文保留!}', md,
                          '低置信度原文保留式输出（带 {!}! 标记）')
            self.assertIn('[WAITING_FOR_HUMAN_INPUT', md,
                          '块级降级占位符仍在')
            # 页码顺序
            self.assertLess(md.index('第 1 页'), md.index('第 2 页'))
            self.assertLess(md.index('第 2 页'), md.index('第 3 页'))

    def test_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = build_report(fake_probe(), fake_ocr_results(),
                                  os.path.join(tmp, 'report.json'))
            self.assertEqual(report['page_count'], 3)
            self.assertEqual(report['text_layer_pages'], 2)
            self.assertEqual(report['ocr_pages'], 1)
            self.assertEqual(report['needs_human_review'], 2)
            self.assertEqual(len(report['review_items']), 2)
            line_item = next(i for i in report['review_items']
                             if i['score'] is not None)
            self.assertEqual(line_item['page'], 3)
            self.assertEqual(line_item['category'], 'text')
            self.assertEqual(line_item['bbox'], [10, 50, 100, 70])
            self.assertAlmostEqual(line_item['score'], 0.55)
            self.assertEqual(line_item['crop'],
                             'needs_human_review/demo_p002_000.png')
            block_item = next(i for i in report['review_items']
                              if i['score'] is None)
            self.assertIn('花体', block_item['reason'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
