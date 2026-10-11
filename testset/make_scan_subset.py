# -*- coding: utf-8 -*-
"""生成扫描件子集 PDF（sources/02_scan_pages.pdf），使扫描件回归测试不依赖 Z 盘源文件。

从 manifest.json 读取 samples/02_scan/ 的采样页码，回到 Z 盘源 PDF 抽取同页，
拼成新的子集 PDF 保存。页面顺序按原书页码排列。
"""
import json
import os
import sys

import pymupdf

BASE = os.path.dirname(os.path.abspath(__file__))
Z = r'Z:\战锤'

manifest = json.load(open(os.path.join(BASE, 'manifest.json'), encoding='utf-8'))
scan_entries = [e for e in manifest['entries'] if e['path'].startswith('samples/02_scan/')]
if not scan_entries:
    print('manifest 中没有 scan 样本'); sys.exit(1)

source = os.path.join(Z, scan_entries[0]['source'])
pages = sorted(e['page'] for e in scan_entries)

doc = pymupdf.open(source)
out = pymupdf.open()
for p in pages:
    out.insert_pdf(doc, from_page=p, to_page=p)
out_path = os.path.join(BASE, 'sources', '02_scan_pages.pdf')
out.save(out_path)
out.close()
doc.close()

size_mb = os.path.getsize(out_path) / 1024 / 1024
print(f'生成 {out_path}：{len(pages)} 页，{size_mb:.1f} MB')
print('页面(原书页码):', pages)
