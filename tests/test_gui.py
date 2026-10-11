# -*- coding: utf-8 -*-
"""GUI 逻辑回归测试：校正回填闭环（合成 fixture，秒级，不依赖模型/窗口）。

覆盖：
- corrections.json 保存/加载/合并；
- apply_corrections 把占位符替换为校正文本，无校正的占位符保留；
- GuiApi 的复核项读取（含截图 base64）与校正导出链路。
"""
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from output_manager import (apply_corrections, load_corrections,
                            save_corrections)
from gui_api import GuiApi, _expand_inputs
import run as run_mod

REVIEW_A = 'needs_human_review/demo_p002_000.png'
REVIEW_B = 'needs_human_review/demo_p002_001.png'


def make_book_dir(root):
    """构造一本书的输出目录：extracted_text.md + report.json + 截图。"""
    book = os.path.join(root, 'demo')
    os.makedirs(os.path.join(book, 'needs_human_review'), exist_ok=True)
    with open(os.path.join(book, 'extracted_text.md'), 'w', encoding='utf-8') as f:
        f.write('高置信度行。\n'
                f'[WAITING_FOR_HUMAN_INPUT: {REVIEW_A}]\n'
                f'中间文本。\n'
                f'[WAITING_FOR_HUMAN_INPUT: {REVIEW_B}]\n')
    report = {
        'page_count': 1, 'text_layer_pages': 0, 'ocr_pages': 1,
        'needs_human_review': 2,
        'review_items': [
            {'page': 1, 'category': 'text', 'bbox': [10, 50, 100, 70],
             'score': 0.55, 'crop': REVIEW_A, 'reason': '置信度低于阈值'},
            {'page': 1, 'category': 'side', 'bbox': [200, 10, 300, 60],
             'score': None, 'crop': REVIEW_B, 'reason': '整块无 OCR 行'},
        ],
    }
    with open(os.path.join(book, 'report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False)
    # 1x1 像素 PNG
    png = (b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01'
           b'\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00'
           b'\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82')
    for crop in (REVIEW_A, REVIEW_B):
        with open(os.path.join(book, crop.replace('/', os.sep)), 'wb') as f:
            f.write(png)
    return book


class TestCorrections(unittest.TestCase):
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            book = make_book_dir(tmp)
            self.assertEqual(load_corrections(book), {})
            save_corrections(book, {REVIEW_A: '正确文本A'})
            self.assertEqual(load_corrections(book), {REVIEW_A: '正确文本A'})
            save_corrections(book, {REVIEW_B: '正确文本B'})
            self.assertEqual(len(load_corrections(book)), 2)  # 合并保留

    def test_apply(self):
        with tempfile.TemporaryDirectory() as tmp:
            book = make_book_dir(tmp)
            save_corrections(book, {REVIEW_A: '正确文本A'})
            out, replaced = apply_corrections(book)
            self.assertEqual(replaced, 1, '只替换有校正的占位符')
            with open(out, encoding='utf-8') as f:
                text = f.read()
            self.assertIn('正确文本A', text)
            self.assertNotIn(f'[WAITING_FOR_HUMAN_INPUT: {REVIEW_A}]', text)
            self.assertIn(f'[WAITING_FOR_HUMAN_INPUT: {REVIEW_B}]', text,
                          '无校正的占位符必须保留')
            # 重新加载校正后二次应用：两处都被替换
            save_corrections(book, {REVIEW_B: '正确文本B'})
            out2, replaced2 = apply_corrections(book)
            self.assertEqual(replaced2, 2)
            with open(out2, encoding='utf-8') as f:
                self.assertNotIn('[WAITING_FOR_HUMAN_INPUT', f.read())


class TestExpandInputs(unittest.TestCase):
    """输入展开：文件保留、文件夹取全部 PDF、过滤非 PDF、去重。"""

    def test_expand(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = os.path.join(tmp, 'a.pdf')
            b = os.path.join(tmp, 'b.pdf')
            for p in (a, b):
                open(p, 'wb').write(b'%PDF')
            open(os.path.join(tmp, 'c.txt'), 'w').write('not pdf')
            sub = os.path.join(tmp, 'sub')
            os.makedirs(sub)
            open(os.path.join(sub, 'd.pdf'), 'wb').write(b'%PDF')
            # 文件 + 文件夹 + 重复 + 非PDF（文件夹只取顶层 PDF，不递归）
            result = _expand_inputs([a, tmp, a, os.path.join(tmp, 'c.txt')])
            self.assertEqual(sorted(result), sorted([a, b]),
                             '应展开文件夹顶层 PDF、去重、过滤非 PDF')
            self.assertEqual(_expand_inputs([os.path.join(tmp, 'c.txt')]), [],
                             '没有 PDF 时返回空')
            self.assertEqual(_expand_inputs(None), [])


class TestPageIsArt(unittest.TestCase):
    """纯图页像素判定：深色满版图/无文字页跳过 OCR。"""

    def _write(self, path, dark_rows, light_rows):
        import cv2
        import numpy as np
        dark = np.zeros((dark_rows, 400), np.uint8)
        light = np.full((light_rows, 400), 255, np.uint8)
        cv2.imwrite(path, np.vstack([dark, light]) if light_rows else dark)

    def test_detect(self):
        with tempfile.TemporaryDirectory() as tmp:
            dark = os.path.join(tmp, 'dark.png')
            self._write(dark, 100, 0)  # 全黑页
            self.assertTrue(run_mod.page_is_art(dark))
            text = os.path.join(tmp, 'text.png')
            self._write(text, 20, 80)  # 20% 深色行 = 正文页
            self.assertFalse(run_mod.page_is_art(text))


class TestResolveOutDir(unittest.TestCase):
    """同名 PDF 输出目录去冲突：同文件重跑同目录，异源同名进不同目录。"""

    def test_conflict(self):
        with tempfile.TemporaryDirectory() as tmp:
            src_a = os.path.join(tmp, 'srca')
            src_b = os.path.join(tmp, 'srcb')
            os.makedirs(src_a)
            os.makedirs(src_b)
            pdf_a = os.path.join(src_a, 'book.pdf')
            pdf_b = os.path.join(src_b, 'book.pdf')
            open(pdf_a, 'wb').write(b'%PDF-1.4 AAAA')
            open(pdf_b, 'wb').write(b'%PDF-1.4 BBBBBBBBBBBBB')
            out = os.path.join(tmp, 'out')
            d1 = run_mod._resolve_out_dir(pdf_a, out, 'book')
            # 第一次：无冲突，原名文件夹
            self.assertEqual(os.path.basename(d1), 'book')
            with open(os.path.join(d1, 'probe_report.json'), 'w',
                      encoding='utf-8') as f:
                f.write(json.dumps({'pdf': pdf_a}))
            # 同一文件重跑：还是原名文件夹（续跑生效）
            d2 = run_mod._resolve_out_dir(pdf_a, out, 'book')
            self.assertEqual(os.path.basename(d2), 'book')
            # 异源同名：带指纹后缀，且两次一致（幂等）
            d3 = run_mod._resolve_out_dir(pdf_b, out, 'book')
            self.assertNotEqual(os.path.basename(d3), 'book')
            self.assertTrue(os.path.basename(d3).startswith('book_'))
            d4 = run_mod._resolve_out_dir(pdf_b, out, 'book')
            self.assertEqual(d3, d4)


class TestPurgeStaleCrops(unittest.TestCase):
    """僵尸截图清理：只保留 report/corrections 引用的截图。"""

    def test_purge(self):
        with tempfile.TemporaryDirectory() as tmp:
            book = os.path.join(tmp, 'book')
            review = os.path.join(book, 'needs_human_review')
            os.makedirs(review)
            keep1 = os.path.join(review, 'a_p001_000.png')
            keep2 = os.path.join(review, 'a_p001_001.png')
            junk = os.path.join(review, 'a_000.png')  # 旧命名僵尸
            for p in (keep1, keep2, junk):
                open(p, 'wb').write(b'x')
            report = {'review_items': [
                {'crop': 'needs_human_review/a_p001_000.png'},
                {'crop': 'needs_human_review/a_p001_001.png'}]}
            removed = run_mod._purge_stale_crops(book, report)
            self.assertEqual(removed, 1)
            self.assertFalse(os.path.exists(junk))
            self.assertTrue(os.path.exists(keep1))


class TestPagesToProcess(unittest.TestCase):
    """页码范围 + 断点续跑：页码过滤与去重逻辑。"""

    def test_range_and_resume(self):
        all_ocr = list(range(0, 10))  # 0-9
        self.assertEqual(run_mod.pages_to_process(all_ocr, None, set()),
                         all_ocr, '无范围无已完成 = 全部')
        self.assertEqual(run_mod.pages_to_process(all_ocr, (3, 6), set()),
                         [2, 3, 4, 5], '范围 3-6（1 起始）')
        self.assertEqual(run_mod.pages_to_process(all_ocr, (5, None), set()),
                         [4, 5, 6, 7, 8, 9], '只给起始页')
        self.assertEqual(run_mod.pages_to_process(all_ocr, None, {0, 2, 4}),
                         [1, 3, 5, 6, 7, 8, 9], '续跑跳过已完成')
        self.assertEqual(run_mod.pages_to_process(all_ocr, (2, 8), {2, 3}),
                         [1, 4, 5, 6, 7], '范围+续跑组合')


class TestGuiApiFlow(unittest.TestCase):
    def test_review_and_export_flow(self):
        api = GuiApi()  # 无需窗口即可用复核/导出链路
        with tempfile.TemporaryDirectory() as tmp:
            book = make_book_dir(tmp)
            items = api.get_review_items(tmp, 'demo')
            self.assertEqual(len(items), 2)
            self.assertTrue(items[0]['image'].startswith('data:image/png;base64,'),
                            '截图应转成 base64')
            self.assertIsNone(items[1]['score'], '整块降级项 score 应为 None')
            # 保存校正 -> 导出
            api.save_corrections(tmp, 'demo', {REVIEW_A: '手工校对文本'})
            self.assertEqual(load_corrections(book), {REVIEW_A: '手工校对文本'})
            r = api.export_corrected(tmp, 'demo')
            self.assertEqual(r['replaced'], 1)
            with open(os.path.join(book, 'corrected_text.md'),
                      encoding='utf-8') as f:
                self.assertIn('手工校对文本', f.read())

    def test_empty_output_dir_resolves_to_default(self):
        """结果页/复核页传空目录时回落到默认输出目录（曾出过的短路 bug）。"""
        api = GuiApi()
        with tempfile.TemporaryDirectory() as tmp:
            api.default_output_dir = tmp
            make_book_dir(tmp)
            books = api.list_books('')
            self.assertEqual([b['stem'] for b in books], ['demo'])
            self.assertTrue(api.get_book_text('', 'demo'))
            self.assertEqual(len(api.get_review_items('', 'demo')), 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
