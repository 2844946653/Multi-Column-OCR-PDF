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


class OcrEngine:
    """OCR 引擎：版面 + OCR 一次推理，行级置信度分级与降级处理。

    fast=True 使用轻量模型（约 2-3 倍速，准确率略降）。
    """

    def __init__(self, fast=False, intra_threads=None):
        from layout_analyzer import LayoutAnalyzer
        self.fast = fast
        self.analyzer = LayoutAnalyzer(fast=fast, intra_threads=intra_threads)
        self._line_ocr = None  # 竖排重识别的轻量 OCR（惰性创建）

    def _get_line_ocr(self):
        """独立的质量级检测+识别管线，用于单行/竖排重识别。

        用 server 模型而非 mobile：重识别对象正是"mobile 判了低分的行"，
        其中大量是暗扫描件的正常字——server 模型 + 图像增强能大幅挽回。
        惰性创建（约 1GB，仅在有候选行时加载）。
        """
        if self._line_ocr is None:
            from paddleocr import PaddleOCR
            self._line_ocr = PaddleOCR(
                text_detection_model_name='PP-OCRv5_server_det',
                text_recognition_model_name='PP-OCRv5_server_rec',
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                # 关键：默认"最小边=736"会把 44px 高的小截图放大 ~17 倍
                # （1100x44 -> 1100x18385），单数组 300MB，多线程必爆内存。
                # 改为"最大边=960"，小图保持原尺寸。
                text_det_limit_side_len=960,
                text_det_limit_type='max',
                text_recognition_batch_size=8,
                engine='onnxruntime',
            )
        return self._line_ocr

    @staticmethod
    def _enhance(crop):
        """暗扫描件图像增强：自动对比度拉伸 + 轻度放大（小图放大有助识别）。"""
        from PIL import ImageEnhance, ImageOps
        if min(crop.size) < 48:
            crop = crop.resize((crop.width * 2, crop.height * 2),
                               Image.LANCZOS)
        crop = ImageOps.autocontrast(crop, cutoff=2)
        return ImageEnhance.Contrast(crop).enhance(1.3)

    def _re_recognize(self, img, bbox, out_dir, stem, counter,
                      page_label='', force_rotations=None):
        """对低分/竖排行做旋转重识别，返回 (text, score) 或 None。

        force_rotations: 指定尝试的旋转角度（块级抢救传 (90, 270)）。
        临时文件名带页码 + 线程 id，防止多线程并行时互相覆盖。
        注意：_get_line_ocr 返回的是 PaddleOCR 经典管线，
        rec_texts/rec_scores 直接位于 result[0].json['res'] 下
        （曾误用 ['overall_ocr_res'] 键导致整条抢救路径静默失效）。
        """
        import threading
        try:
            pad = CROP_PAD * 2
            x0, y0, x1, y1 = [int(v) for v in bbox]
            x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
            x1, y1 = min(img.width, x1 + pad), min(img.height, y1 + pad)
            if x1 - x0 < 8 or y1 - y0 < 8:
                return None
            crop = img.crop((x0, y0, x1, y1))
            best = None
            w, h = crop.size
            rotations = (force_rotations
                         if force_rotations is not None
                         else ((90, 270) if h > w * _VERTICAL_RATIO else (0,)))
            tid = threading.get_ident()
            for deg in rotations:
                tmp = (f'{stem}_{page_label}_{counter[0]:03d}'
                       f'_t{tid}_rot{deg}.png')
                counter[0] += 1
                path = os.path.join(out_dir, tmp)
                rotated = crop.rotate(deg, expand=True) if deg else crop
                self._enhance(rotated).save(path)
                try:
                    result = self._get_line_ocr().predict(path)
                    o = result[0].json['res']
                    texts, scores = o.get('rec_texts') or [], o.get('rec_scores')
                    if texts and scores is not None:
                        i = max(range(len(texts)),
                                key=lambda k: scores[k])
                        if best is None or scores[i] > best[1]:
                            best = (texts[i], float(scores[i]))
                finally:
                    if os.path.isfile(path):
                        os.remove(path)
            return best
        except Exception:
            return None

    def _rescue_block(self, img, bbox, out_dir, stem, counter, page_label):
        """块级旋转抢救：无 OCR 行的文字样貌块（多为竖排文字，检测模型
        找不到框），旋转 ±90° 后整块转正识别。返回按阅读顺序的
        [(text, score), ...] 或 None。旋转 270° 的块文字自下而上，
        需倒序还原行序。
        """
        import threading
        try:
            pad = CROP_PAD * 2
            x0, y0, x1, y1 = [int(v) for v in bbox]
            x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
            x1, y1 = min(img.width, x1 + pad), min(img.height, y1 + pad)
            if x1 - x0 < 8 or y1 - y0 < 8:
                return None
            crop = img.crop((x0, y0, x1, y1))
            tid = threading.get_ident()
            best = None
            for deg in (90, 270):
                tmp = (f'{stem}_{page_label}_{counter[0]:03d}'
                       f'_t{tid}_blk{deg}.png')
                counter[0] += 1
                path = os.path.join(out_dir, tmp)
                self._enhance(crop.rotate(deg, expand=True)).save(path)
                try:
                    result = self._get_line_ocr().predict(path)
                    o = result[0].json['res']
                    texts = o.get('rec_texts') or []
                    scores = [float(s) for s in (o.get('rec_scores') or [])]
                    if texts:
                        avg = sum(scores) / len(scores)
                        if best is None or avg > best[3]:
                            best = (texts, scores, deg, avg)
                finally:
                    if os.path.isfile(path):
                        os.remove(path)
            if best is None:
                return None
            texts, scores, deg, avg = best
            if deg == 270:
                texts = texts[::-1]
                scores = scores[::-1]
            return list(zip(texts, scores))
        except Exception:
            return None

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
        for b in layout['blocks']:
            b['lines'] = []
            if b['category'] == 'table' or b['category'] in _TEXT_CATEGORIES:
                block_lines = sorted(
                    (ln for ln in lines
                     if id(ln) not in assigned
                     and _center_in(ln['bbox'], b['bbox'])),
                    key=lambda ln: ln['bbox'][1])
                for ln in block_lines:
                    s = ln['score']
                    h = ln['bbox'][3] - ln['bbox'][1]
                    w = ln['bbox'][2] - ln['bbox'][0]
                    vertical = h > w * _VERTICAL_RATIO
                    # 重识别仅针对竖排与即将 flagged 的低分行，
                    # 不给每条 0.6-0.9 的行都跑额外 OCR（性能）
                    if vertical or s < CONF_DOUBT:
                        # 竖排/低分：质量模型 + 图像增强重识别，取高分结果
                        ln['retried'] = True
                        retry = self._re_recognize(
                            img, ln['bbox'], review_dir, stem, counter,
                            page_label=page_label)
                        if retry and retry[1] > s:
                            ln['text'], ln['score'] = retry
                            ln['rotated'] = vertical
                            s = ln['score']
                    # 重识别后仍为空 = 检测框的半截拼接/碎片（把相邻两行的
                    # 下半截+上半截拼成一行），纯噪声，直接丢弃
                    if not (ln.get('text') or '').strip():
                        continue
                    assigned.add(id(ln))
                    if s >= CONF_HIGH:
                        ln['tier'] = 'ok'
                    elif s >= CONF_DOUBT:
                        ln['tier'] = 'doubtful'
                    else:
                        ln['tier'] = 'flagged'
                        rel = save_crop(ln['bbox'])
                        ln['crop'] = rel  # 截图失败时为 None，文本仍保留
                    b['lines'].append(ln)
            b['text'] = _assemble_text(b)

        # 第二遍：处理"整块无 OCR 行"的块（需等所有块的行分配完成，
        # 才能正确判断块间重叠——行被去重归了别的块的内容不算丢失）
        for b in layout['blocks']:
            if b['lines'] or b['category'] not in _TEXT_CATEGORIES:
                continue  # 有行块已处理；表格/图片块不产复核项
            if _overlaps_block_with_lines(b, layout['blocks']):
                continue  # 与有行块重叠：内容已在他处输出
            if not _crop_looks_textlike(img, b['bbox']):
                continue  # 图片/装饰区域静默跳过
            # 块级旋转抢救：竖排文字检测模型常找不到框，整块转正识别
            rescued = self._rescue_block(img, b['bbox'], review_dir,
                                         stem, counter, page_label)
            if rescued and sum(s for _, s in rescued) / len(rescued) \
                    >= CONF_DOUBT:
                for text, score in rescued:
                    if not text.strip():
                        continue
                    b['lines'].append({
                        'text': text, 'score': score,
                        'tier': 'ok' if score >= CONF_HIGH else 'doubtful',
                        'bbox': list(b['bbox']),
                        'rescued_block': True})
            if not b['lines']:
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
