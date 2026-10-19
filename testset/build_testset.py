# -*- coding: utf-8 -*-
"""Step 0: 组建样张测试集。运行: python build_testset.py"""
import sys, os, json
sys.stdout.reconfigure(encoding='utf-8')
import pymupdf
import numpy as np
from PIL import Image

# 测试集目录：相对本脚本，跨机器可移植
BASE = os.path.dirname(os.path.abspath(__file__))
# 源书库：优先 --source-dir 参数，其次环境变量 WARHAMMER_LIB，
# 最后回退到本机默认路径（仅开发者本机有意义）
Z = (sys.argv[sys.argv.index('--source-dir') + 1]
     if '--source-dir' in sys.argv
     else os.environ.get('WARHAMMER_LIB', r'Z:\战锤'))
os.makedirs(BASE, exist_ok=True)
for sub in ['samples/01_digital_story', 'samples/02_scan', 'samples/03_tables',
            'samples/04_magazine', 'samples/05_image_input', 'sources', 'gt']:
    os.makedirs(os.path.join(BASE, sub), exist_ok=True)

manifest = {'created': '2026-10-08', 'entries': []}
def add_entry(pid, **kw):
    e = {'id': pid, **kw}
    manifest['entries'].append(e)
    return e

# ---------- 通用统计 ----------
def pix_stats(pm):
    img = Image.frombytes('RGB', (pm.width, pm.height), pm.samples).convert('L')
    a = np.asarray(img, dtype=np.float32)
    dark = (a < 128).astype(np.float32)
    rowden = dark.mean(axis=1)
    colden = dark.mean(axis=0)
    w = pm.width
    thirds = [float(colden[j*w//3:(j+1)*w//3].mean()) for j in range(3)]
    return dict(dark=float(dark.mean()), textrows=float((rowden > 0.02).mean()), thirds=thirds)

def classify_page(s):
    if s['textrows'] < 0.05 or s['dark'] > 0.60:
        return 'art'
    t = s['thirds']
    if t[0] > 0.12 and t[2] > 0.12 and t[1] < min(t[0], t[2]) * 0.9:
        return 'two-col'
    if s['dark'] > 0.30:
        return 'text+dense'
    return 'text'

# ---------- 1. 瓦什托尔（扫描件，无文本层） ----------
print('== 瓦什托尔 扫描分析 ==')
doc = pymupdf.open(os.path.join(Z, '厄兆方舟：瓦什托尔.pdf'))
vstats = []
for i in range(doc.page_count):
    pm = doc[i].get_pixmap(dpi=72)
    s = pix_stats(pm)
    s['page'] = i
    s['cat'] = classify_page(s)
    vstats.append(s)
doc.close()
from collections import Counter
print('页面类别分布:', Counter(s['cat'] for s in vstats))
# 每类按页码间隔 >=3 抽样
vpick, last = [], {}
for s in vstats:
    c = s['cat']
    if last.get(c, -99) + 3 <= s['page']:
        vpick.append(s)
        last[c] = s['page']
quota = {'art': 2, 'text': 3, 'two-col': 2, 'text+dense': 1}
vfinal = []
seen = Counter()
for s in vpick:
    if seen[s['cat']] < quota.get(s['cat'], 0):
        vfinal.append(s)
        seen[s['cat']] += 1
for s in vfinal:
    print(f"  pick p{s['page']:>3} [{s['cat']:<10}] dark={s['dark']:.2f} textrows={s['textrows']:.2f}")

# 300 DPI 渲染保存
doc = pymupdf.open(os.path.join(Z, '厄兆方舟：瓦什托尔.pdf'))
for s in vfinal:
    png = os.path.join(BASE, f"samples/02_scan/vashtorr_p{s['page']:03d}.png")
    doc[s['page']].get_pixmap(dpi=300).save(png)
    add_entry(f'scan-{s["page"]:03d}', source='厄兆方舟：瓦什托尔.pdf', page=s['page'],
              category=s['cat'], path=f'samples/02_scan/vashtorr_p{s["page"]:03d}.png',
              tests=['300dpi渲染', '版面分析', 'OCR', '置信度分级'],
              note='纯扫描件，无文本层')
doc.close()

# ---------- 2. 阿巴顿剧情部分（数字版，有文本层） ----------
print('== 阿巴顿剧情部分 数字版 ==')
doc = pymupdf.open(os.path.join(Z, '厄兆方舟：阿巴顿（剧情部分）.pdf'))
info = []
for i in range(doc.page_count):
    page = doc[i]
    txt = page.get_text().strip()
    blocks = [b for b in page.get_text('blocks') if b[6] == 0]
    w = page.rect.width
    left = [b for b in blocks if (b[0] + b[2]) / 2 < w / 2]
    right = [b for b in blocks if (b[0] + b[2]) / 2 >= w / 2]
    two = len(left) >= 5 and len(right) >= 5
    info.append(dict(page=i, chars=len(txt), blocks=len(blocks), two_col=two))
apick = []
apick.append(next(x for x in info if x['chars'] < 60))              # 艺术页
twos = [x for x in info if x['two_col'] and x['page'] not in [a['page'] for a in apick]]
if twos:
    apick.append(twos[len(twos)//2])                                  # 双栏页(中间位置)
texts = [x for x in info if x['chars'] > 800 and not x['two_col'] and x['page'] not in [a['page'] for a in apick]]
step = max(1, len(texts)//4)
for k in range(4):
    apick.append(texts[min(k*step, len(texts)-1)])
apick = sorted(apick, key=lambda x: x['page'])
out = pymupdf.open()
for x in apick:
    out.insert_pdf(doc, from_page=x['page'], to_page=x['page'])
    gt = doc[x['page']].get_text('text', sort=True)
    open(os.path.join(BASE, f"gt/story_p{x['page']:03d}.txt"), 'w', encoding='utf-8').write(gt)
    add_entry(f'story-{x["page"]:03d}', source='厄兆方舟：阿巴顿（剧情部分）.pdf', page=x['page'],
              category='two-col' if x['two_col'] else ('art' if x['chars'] < 60 else 'text'),
              path=f'sources/01_story_pages.pdf (内页{apick.index(x)})',
              gt=f'gt/story_p{x["page"]:03d}.txt', tests=['文本层探测', '直取', '阅读顺序'],
              note=f'数字版有文本层, chars={x["chars"]}, blocks={x["blocks"]}')
    print(f"  pick p{x['page']:>3} chars={x['chars']:>5} blocks={x['blocks']:>3} two_col={x['two_col']}")
out.save(os.path.join(BASE, 'sources/01_story_pages.pdf'))
out.close()
doc.close()

# ---------- 3. 太空死灵10版圣典（数字版，表格页） ----------
print('== 太空死灵10版 表格页 ==')
doc = pymupdf.open(os.path.join(Z, '太空死灵10版圣典.pdf'))
ruled = []
for i in range(doc.page_count):
    h = v = 0
    for d in doc[i].get_drawings():
        for it in d['items']:
            if it[0] == 'l':
                p1, p2 = it[1], it[2]
                if abs(p1.y - p2.y) < 1.5: h += 1
                elif abs(p1.x - p2.x) < 1.5: v += 1
            elif it[0] == 're':
                h += 1; v += 1
    if h >= 15 and v >= 10:
        ruled.append(dict(page=i, score=h + v, words=len(doc[i].get_text().split())))
ruled.sort(key=lambda x: -x['score'])
tpick = ruled[:2]  # 线框密度最高的两页 = 真数据表页
for extra in (131, 133):  # 补两页邻近非表格页作对照
    tpick.append(dict(page=extra, score=0,
                      words=len(doc[extra].get_text().split())))
tpick.sort(key=lambda x: x['page'])  # 子集按原书页码升序，与测试的 gt 映射一致
out = pymupdf.open()
for t in tpick:
    out.insert_pdf(doc, from_page=t['page'], to_page=t['page'])
    gt = doc[t['page']].get_text('text', sort=True)
    open(os.path.join(BASE, f"gt/table_p{t['page']:03d}.txt"), 'w', encoding='utf-8').write(gt)
    png = os.path.join(BASE, f"samples/03_tables/necron_p{t['page']:03d}.png")
    doc[t['page']].get_pixmap(dpi=300).save(png)
    cat = 'table' if t['score'] > 0 else 'text'
    add_entry(f'table-{t["page"]:03d}', source='太空死灵10版圣典.pdf', page=t['page'],
              category=cat, path=f'samples/03_tables/necron_p{t["page"]:03d}.png',
              gt=f'gt/table_p{t["page"]:03d}.txt', tests=['表格结构', '文本层表格'],
              note=f'数字版, 线框得分={t["score"]}, words={t["words"]}')
    print(f"  pick p{t['page']:>3} 线框得分={t['score']:>3} words={t['words']:>4}")
out.save(os.path.join(BASE, 'sources/03_tables_pages.pdf'))
out.close()
doc.close()

# ---------- 3b. 圣血天使9版 大画布页（尺度无关分栏的对照样本） ----------
print('== 圣血天使9版 大画布页 ==')
doc = pymupdf.open(os.path.join(Z, '圣血天使9版圣典.pdf'))
png = os.path.join(BASE, 'samples/06_canvas/codex_p010.png')
os.makedirs(os.path.dirname(png), exist_ok=True)
doc[10].get_pixmap(dpi=300).save(png)
add_entry('canvas-010', source='圣血天使9版圣典.pdf', page=10,
          category='canvas', path='samples/06_canvas/codex_p010.png',
          tests=['大画布分栏', '尺度无关栏检测'],
          note='6000x7921 大画布页（两页内容拼一张），测栏检测尺度无关性')
print('  pick p10 (canvas)')
doc.close()

# ---------- 4. White Dwarf 487（杂志复杂排版） ----------
print('== WD487 杂志页 ==')
doc = pymupdf.open(os.path.join(Z, 'White Dwarf Magazine 487 - Apr 2023 ENG (UK).pdf'))
bscore = []
for i in range(doc.page_count):
    blocks = [b for b in doc[i].get_text('blocks') if b[6] == 0]
    bscore.append(dict(page=i, blocks=len(blocks)))
bscore.sort(key=lambda x: -x['blocks'])
mpick, used = [], []
for b in bscore:
    if len(mpick) >= 4:
        break
    if all(abs(b['page'] - u) >= 10 for u in used):
        mpick.append(b); used.append(b['page'])
mpick.sort(key=lambda x: x['page'])  # 子集按原书页码升序，与测试的 gt 映射一致
out = pymupdf.open()
for b in mpick:
    out.insert_pdf(doc, from_page=b['page'], to_page=b['page'])
    gt = doc[b['page']].get_text('text', sort=True)
    open(os.path.join(BASE, f"gt/mag_p{b['page']:03d}.txt"), 'w', encoding='utf-8').write(gt)
    png = os.path.join(BASE, f"samples/04_magazine/wd487_p{b['page']:03d}.png")
    doc[b['page']].get_pixmap(dpi=300).save(png)
    add_entry(f'mag-{b["page"]:03d}', source='White Dwarf Magazine 487.pdf', page=b['page'],
              category='magazine', path=f'samples/04_magazine/wd487_p{b["page"]:03d}.png',
              gt=f'gt/mag_p{b["page"]:03d}.txt', tests=['侧边栏', '多栏', '图片注释'],
              note=f'数字版, 文本块数={b["blocks"]}(全书靠前)')
    print(f"  pick p{b['page']:>3} blocks={b['blocks']:>3}")
out.save(os.path.join(BASE, 'sources/04_magazine_pages.pdf'))
out.close()
doc.close()

# ---------- 5. 图片输入（用户拆的 WebP） ----------
print('== 图片输入 ==')
img_src = os.path.join(Z, '厄兆方舟：阿巴顿（剧情部分）')
imgs = [('1.jpg', 'plot_01_title.png', 'art', '标题艺术页'),
        ('5.jpg', 'plot_05_darkart.png', 'art', '满幅暗色插图'),
        ('15.jpg', 'plot_15_twocol.png', 'two-col', '双栏正文')]
for src, dst, cat, note in imgs:
    png = os.path.join(BASE, f'samples/05_image_input/{dst}')
    Image.open(os.path.join(img_src, src)).convert('RGB').save(png, 'PNG')
    add_entry(f'img-{src.split(".")[0]}', source='厄兆方舟：阿巴顿（剧情部分）拆分图', page=src,
              category=cat, path=f'samples/05_image_input/{dst}', tests=['图片路径输入', '格式探测'],
              note=f'{note} (原文件为WebP伪装.jpg, 已转真PNG)')
    print(f"  pick {src} -> {dst} [{cat}]")

json.dump(manifest, open(os.path.join(BASE, 'manifest.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=2)
print('== done: manifest.json 已生成, 共', len(manifest['entries']), '条 ==')
