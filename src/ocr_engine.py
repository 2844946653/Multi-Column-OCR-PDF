# -*- coding: utf-8 -*-
"""模块四：文本提取与置信度分级（防幻觉核心，保留式标记策略）。

复用 PP-StructureV3 的整页 OCR 输出（行级文本/置信度/坐标），不重复跑 OCR：
1. 按版面块分配文本行（行中心落在块内），块内按自上而下排序；
2. 竖排文字（检测框高窄，如侧栏竖排/书脊字）：旋转 ±90° 后用轻量 OCR
   重新识别，取高分结果——扫描件常见的竖排字直接转正解决；
3. 三级置信度处理（保留式，正文绝不断流）：
   - score >= CONF_HIGH：直接输出
   - CONF_DOUBT <= score < CONF_HIGH：输出原文，就地标记 {?原文?}
   - score < CONF_DOUBT：原文保留并标 {!原文!}，同时截图进校对清单——
     不打断内容，但绝不悄悄输出（标记醒目，可全文查找 {! 定位）
4. 文本块内没有任何 OCR 行（花体/漫画气泡等检测失败区域）：
   做"文字样貌"检测——图片/装饰区域静默跳过（不产复核项），
   其余整块截图降级并放占位符；
5. 表格块行级提取（v1 平文本）；图片块不提取文本。
"""
import os
import threading

from PIL import Image

# 置信度三级阈值（保留式标记；< CONF_DOUBT 的原文仍保留，但标 {!}!）
CONF_HIGH = 0.90
CONF_DOUBT = 0.60
# 截图外扩像素（行级/块级）
CROP_PAD = 6
# 竖排判定：检测框高宽比超过该值视为竖排/旋转文字
_VERTICAL_RATIO = 1.8

# 参与文本提取的版面类别
_TEXT_CATEGORIES = {'text', 'title', 'span_title', 'side', 'caption_of'}

# 全进程共享的抢救管线（共享一份 + 锁串行，避免多线程各建一套把内存
# 挤爆——实测 4 线程各建一套 ≈ 8GB 直接卡死）。
# 三档管线：
# - v6 tiny 纯识别（行级抢救主力）：0.16s/张，快；
# - v6 tiny 检测+识别（块级抢救）：需重新分行；
# - server 纯识别（边界分接力）：v6 tiny 给出 0.45-0.75 边界分时
#   用大模型确认，只跑少量行，速度代价小、准确率接近 server 全程。
_shared_rec = None
_shared_recdet = None
_shared_server_rec = None
_rec_lock = threading.Lock()
_recdet_lock = threading.Lock()
_server_rec_lock = threading.Lock()


def _build_pipeline(det, server=False):
    from paddleocr import PaddleOCR
    if server:
        rec = 'PP-OCRv5_server_rec'
        det_name = None
    else:
        rec = 'PP-OCRv6_tiny_rec'
        det_name = 'PP-OCRv6_tiny_det' if det else None
    return PaddleOCR(
        text_detection_model_name=det_name,
        text_recognition_model_name=rec,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_det_limit_side_len=960,
        text_det_limit_type='max',
        text_recognition_batch_size=16,
        engine='onnxruntime',
    )


def _retry_ocr_batch(paths, det=False, chunk=16):
    """在共享锁内批量识别多张图，返回与 paths 对齐的结果 dict 列表。

    det=False：纯识别（行级抢救用，快）；det=True：检测+识别（块级用）。
    """
    global _shared_rec, _shared_recdet
    lock = _recdet_lock if det else _rec_lock
    out = []
    with lock:
        if det:
            if _shared_recdet is None:
                _shared_recdet = _build_pipeline(det=True)
            pipe = _shared_recdet
        else:
            if _shared_rec is None:
                _shared_rec = _build_pipeline(det=False)
            pipe = _shared_rec
        for i in range(0, len(paths), chunk):
            batch = paths[i:i + chunk]
            res = pipe.predict(batch)
            out.extend(r.json['res'] for r in res)
    return out


def _server_rec_batch(paths, chunk=16):
    """server 大模型纯识别（边界分接力用），共享一份。"""
    global _shared_server_rec
    out = []
    with _server_rec_lock:
        if _shared_server_rec is None:
            _shared_server_rec = _build_pipeline(det=False, server=True)
        for i in range(0, len(paths), chunk):
            batch = paths[i:i + chunk]
            res = _shared_server_rec.predict(batch)
            out.extend(r.json['res'] for r in res)
    return out


class OcrEngine:
    """OCR 引擎：版面 + OCR 一次推理，行级置信度分级与降级处理。

    fast=True 使用轻量模型（约 2-3 倍速，准确率略降）。
    """

    def __init__(self, fast=False, intra_threads=None):
        from layout_analyzer import LayoutAnalyzer
        self.fast = fast
        self.analyzer = LayoutAnalyzer(fast=fast, intra_threads=intra_threads)

    @staticmethod
    def _enhance(crop):
        """暗扫描件图像增强：自动对比度拉伸 + 轻度放大（小图放大有助识别）。"""
        from PIL import ImageEnhance, ImageOps
        if min(crop.size) < 48:
            crop = crop.resize((crop.width * 2, crop.height * 2),
                               Image.LANCZOS)
        crop = ImageOps.autocontrast(crop, cutoff=2)
        return ImageEnhance.Contrast(crop).enhance(1.3)

    def _crop(self, img, bbox):
        """bbox 外扩裁剪，返回 PIL Image 或 None。"""
        pad = CROP_PAD * 2
        x0, y0, x1, y1 = [int(v) for v in bbox]
        x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
        x1, y1 = min(img.width, x1 + pad), min(img.height, y1 + pad)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None
        return img.crop((x0, y0, x1, y1))

    def _batch_retry_lines(self, img, candidates, out_dir, stem, counter,
                           page_label):
        """行级批量抢救：候选行（竖排或低分）的裁剪变体一次性打包，
        只拿一次共享锁批量识别，比逐张排队快数倍。

        candidates: [(ln, vertical), ...]，就地更新 ln 的 text/score。
        """
        import threading
        tid = threading.get_ident()
        temp_counter = [0]  # 临时文件独立计数，不占复核截图号
        tasks = []
        for ln, vertical in candidates:
            ln['retried'] = True
            crop = self._crop(img, ln['bbox'])
            if crop is None:
                continue
            for deg in ((90, 270) if vertical else (0,)):
                tmp = (f'{stem}_{page_label}_{temp_counter[0]:03d}'
                       f'_t{tid}_r{deg}.png')
                temp_counter[0] += 1
                path = os.path.join(out_dir, tmp)
                rotated = crop.rotate(deg, expand=True) if deg else crop
                self._enhance(rotated).save(path)
                tasks.append((ln, deg, path))
        if not tasks:
            return
        try:
            results = _retry_ocr_batch([t[2] for t in tasks])  # 行级：纯识别
            for (ln, deg, path), res in zip(tasks, results):
                texts = res.get('rec_texts') or []
                scores = [float(s) for s in (res.get('rec_scores') or [])]
                if texts:
                    i = max(range(len(texts)), key=lambda k: scores[k])
                    if scores[i] > ln['score']:
                        ln['text'], ln['score'] = texts[i], scores[i]
                        ln['rotated'] = (deg != 0)
            # 边界分接力：tiny 给 0.45-0.75 的行，用 server 大模型确认
            # （只跑少量边界行，速度代价小，准确率接近 server 全程）
            esc_tasks = []
            for ln, _vertical in candidates:
                if 0.45 <= ln['score'] < 0.75:
                    crop = self._crop(img, ln['bbox'])
                    if crop is None:
                        continue
                    tmp = (f'{stem}_{page_label}_{temp_counter[0]:03d}'
                           f'_t{tid}_esc.png')
                    temp_counter[0] += 1
                    path = os.path.join(out_dir, tmp)
                    self._enhance(crop).save(path)
                    esc_tasks.append((ln, path))
            if esc_tasks:
                esc_res = _server_rec_batch([t[1] for t in esc_tasks])
                for (ln, path), res in zip(esc_tasks, esc_res):
                    texts = res.get('rec_texts') or []
                    scores = [float(s) for s in (res.get('rec_scores') or [])]
                    if texts and scores[0] > ln['score']:
                        ln['text'], ln['score'] = texts[0], scores[0]
                        ln['escalated'] = True
            for _, path in esc_tasks:
                if os.path.isfile(path):
                    os.remove(path)
        finally:
            for _, _, path in tasks:
                if os.path.isfile(path):
                    os.remove(path)

    def _batch_rescue_blocks(self, img, blocks, out_dir, stem, counter,
                             page_label):
        """块级批量抢救：无 OCR 行的文字样貌块（多为竖排文字，检测模型
        找不到框），旋转 ±90° 整块转正批量识别。成功（平均分 >=
        CONF_DOUBT）就把行注入块内；失败标记 rescue_failed。
        旋转 270° 的块文字自下而上，行序需倒置。
        """
        import threading
        tid = threading.get_ident()
        temp_counter = [0]  # 临时文件独立计数，不占复核截图号
        tasks = []
        for b in blocks:
            crop = self._crop(img, b['bbox'])
            if crop is None:
                b['rescue_failed'] = True
                continue
            for deg in (90, 270):
                tmp = (f'{stem}_{page_label}_{temp_counter[0]:03d}'
                       f'_t{tid}_blk{deg}.png')
                temp_counter[0] += 1
                path = os.path.join(out_dir, tmp)
                self._enhance(crop.rotate(deg, expand=True)).save(path)
                tasks.append((b, deg, path))
        if not tasks:
            return
        try:
            results = _retry_ocr_batch([t[2] for t in tasks], det=True)  # 块级：需分行
            best = {}
            for (b, deg, path), res in zip(tasks, results):
                texts = res.get('rec_texts') or []
                scores = [float(s) for s in (res.get('rec_scores') or [])]
                if not texts:
                    continue
                avg = sum(scores) / len(scores)
                if id(b) not in best or avg > best[id(b)][3]:
                    best[id(b)] = (texts, scores, deg, avg)
            for b in blocks:
                got = best.get(id(b))
                if got is None:
                    b['rescue_failed'] = True
                    continue
                texts, scores, deg, avg = got
                if avg < CONF_DOUBT:
                    b['rescue_failed'] = True
                    continue
                if deg == 270:
                    texts = texts[::-1]
                    scores = scores[::-1]
                for text, score in zip(texts, scores):
                    if not text.strip():
                        continue
                    b['lines'].append({
                        'text': text, 'score': score,
                        'tier': 'ok' if score >= CONF_HIGH else 'doubtful',
                        'bbox': list(b['bbox']),
                        'rescued_block': True})
        finally:
            for _, _, path in tasks:
                if os.path.isfile(path):
                    os.remove(path)

    def process(self, img_path, out_dir, page_label=None):
        """处理单页图像。

        返回: layout dict，其中每个块的 blocks[i] 增加
            lines: [{'text','score','tier','bbox',(crop)}] 行级结果
            text:  该块按置信度规则拼好的文本（含 {?}/{!} 标记与占位符）
        副作用: 低置信度行的截图写入 out_dir/needs_human_review/
        page_label: 截图文件名的页码部分（缺省从文件名 _pNNN 解析），
            保证不同页的截图不互相覆盖。
        """
        layout, lines = self.analyzer.analyze_with_ocr(img_path)
        stem = os.path.splitext(os.path.basename(img_path))[0]
        if page_label is None:
            import re
            m = re.search(r'_p(\d{3,4})(?:\.|$)', os.path.basename(img_path))
            page_label = f'p{m.group(1)}' if m else ''
        review_dir = os.path.join(out_dir, 'needs_human_review')
        os.makedirs(review_dir, exist_ok=True)
        img = Image.open(img_path)
        counter = [0]  # 本页截图序号

        def save_crop(bbox):
            fname = (f'{stem}_{page_label}_{counter[0]:03d}.png'
                     if page_label else f'{stem}_{counter[0]:03d}.png')
            path = os.path.join(review_dir, fname)
            counter[0] += 1
            if _crop_save(img, bbox, path):
                return f'needs_human_review/{fname}'
            return None

        assigned = set()  # 每行只归一个块，防重叠区块重复输出同一行
        # ---------- 第一遍：行分配（不做重识别） ----------
        line_candidates = []
        for b in layout['blocks']:
            b['lines'] = []
            if b['category'] == 'table' or b['category'] in _TEXT_CATEGORIES:
                block_lines = sorted(
                    (ln for ln in lines
                     if id(ln) not in assigned
                     and _center_in(ln['bbox'], b['bbox'])),
                    key=lambda ln: ln['bbox'][1])
                for ln in block_lines:
                    # 检测框的半截拼接/碎片（空文本）直接丢弃
                    if not (ln.get('text') or '').strip():
                        continue
                    h = ln['bbox'][3] - ln['bbox'][1]
                    w = ln['bbox'][2] - ln['bbox'][0]
                    vertical = h > w * _VERTICAL_RATIO
                    # 重识别仅针对竖排与即将 flagged 的低分行
                    if vertical or ln['score'] < CONF_DOUBT:
                        line_candidates.append((ln, vertical))
                    assigned.add(id(ln))
                    b['lines'].append(ln)

        # ---------- 第二遍：批量抢救（行级 + 块级，各只拿一次共享锁） ----------
        if line_candidates:
            self._batch_retry_lines(img, line_candidates, review_dir,
                                    stem, counter, page_label)
        rescue_blocks = []
        for b in layout['blocks']:
            if b['lines'] or b['category'] not in _TEXT_CATEGORIES:
                continue  # 有行块已处理；表格/图片块不产复核项
            if _overlaps_block_with_lines(b, layout['blocks']):
                continue  # 与有行块重叠：内容已在他处输出
            if _crop_looks_textlike(img, b['bbox']):
                rescue_blocks.append(b)
        if rescue_blocks:
            self._batch_rescue_blocks(img, rescue_blocks, review_dir,
                                      stem, counter, page_label)

        # ---------- 第三遍：分级 + 截图 + 文本 ----------
        for b in layout['blocks']:
            for ln in b['lines']:
                s = ln['score']
                if s >= CONF_HIGH:
                    ln['tier'] = 'ok'
                elif s >= CONF_DOUBT:
                    ln['tier'] = 'doubtful'
                else:
                    ln['tier'] = 'flagged'
                    rel = save_crop(ln['bbox'])
                    ln['crop'] = rel  # 截图失败时为 None，文本仍保留
            if b.get('rescue_failed'):
                rel = save_crop(b['bbox'])
                b['review_block'] = rel
            b['text'] = _assemble_text(b)

        layout['needs_human_review'] = counter[0]
        img.close()
        return layout


def _assemble_text(block):
    """按保留式标记规则拼块内文本——正文绝不断流。"""
    parts = []
    for ln in block.get('lines', []):
        if ln['tier'] == 'ok':
            parts.append(ln['text'])
        elif ln['tier'] == 'doubtful':
            parts.append('{?' + ln['text'] + '?}')
        else:  # flagged：原文保留 + 醒目标记（全文搜 {! 可定位）
            parts.append('{!' + ln['text'] + '!}')
    if not parts and block.get('review_block'):
        parts.append(f"[WAITING_FOR_HUMAN_INPUT: {block['review_block']}]")
    return '\n'.join(parts)


def _overlaps_block_with_lines(block, blocks, iou_thresh=0.3):
    """该块是否与任何'有 OCR 行'的块高度重叠（行去重后内容已在他处）。"""
    for other in blocks:
        if other is block or not other.get('lines'):
            continue
        if _iou(block['bbox'], other['bbox']) > iou_thresh:
            return True
    return False


def _iou(b1, b2):
    x0 = max(b1[0], b2[0]); y0 = max(b1[1], b2[1])
    x1 = min(b1[2], b2[2]); y1 = min(b1[3], b2[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    union = ((b1[2] - b1[0]) * (b1[3] - b1[1])
             + (b2[2] - b2[0]) * (b2[3] - b2[1]) - inter)
    return inter / union if union > 0 else 0.0


def _crop_looks_textlike(img, bbox):
    """块截图的"文字样貌"粗判：有若干横排的暗色行 = 像文字；
    深色满幅/几乎空白 = 图片/装饰，返回 False。"""
    x0, y0, x1, y1 = [int(v) for v in bbox]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(img.width, x1), min(img.height, y1)
    if x1 - x0 < 12 or y1 - y0 < 12:
        return False
    import numpy as np
    a = np.asarray(img.crop((x0, y0, x1, y1)).convert('L'), dtype=np.float32)
    dark = (a < 128).astype(np.float32)
    if dark.mean() > 0.75 or dark.mean() < 0.01:
        return False  # 满幅深色图 / 近空白
    rowden = dark.mean(axis=1)
    rows = int((rowden > 0.05).sum())
    # 至少 2 个横排暗行（文字行），且行数占比合理
    return rows >= 2


def _center_in(bbox, outer):
    """bbox 的中心点是否落在 outer 内。"""
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def _crop_save(img, bbox, path, pad=CROP_PAD):
    """截图 bbox（外扩 pad 像素）保存为 PNG。越界裁剪，空区域返回 False。"""
    x0, y0, x1, y1 = [int(v) for v in bbox]
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(img.width, x1 + pad)
    y1 = min(img.height, y1 + pad)
    if x1 <= x0 or y1 <= y0:
        return False
    img.crop((x0, y0, x1, y1)).save(path)
    return True
