# -*- coding: utf-8 -*-
"""模块三：版面分析（区域分类 + 阅读顺序）。

调用 PP-StructureV3（PP-DocLayout_plus-L 版面检测）分析渲染后的页面图像，
输出五类区域（栏内正文 / 跨栏标题 / 侧边栏 / 表格 / 图片）及阅读顺序。

流程：
1. 版面检测 -> 20 类原始标签（见 PP-DocLayout_plus-L 的 id2label）
2. 类别映射：页眉页脚/页码/公式等家具丢弃；aside_text 直接入"侧边栏"
3. 分栏检测：按正文块 x 中心聚簇（大间隙切栏）
4. 标题分类：横跨多栏或超宽的标题 -> 跨栏标题（span_title）
5. 图注合并：figure_title 就近挂到上方最近的图片/表格
6. 阅读顺序：按纵向"带"推进——每个带内先完整读完左栏再读右栏，
   跨栏标题/侧边栏/表格/图片等独立单元在其 y 位置插入阅读流，
   绝不逐行左右串行。

注意：PaddlePaddle 3.3.1 CPU 的原生推理路径在本机有 bug（静态图 oneDNN
PIR 报错 / 动态图段错误），本模块固定使用 onnxruntime 引擎绕过。
"""
import os

import numpy as np

# PP-DocLayout_plus-L 的 20 类标签 -> 本工具类别
# 'furniture' 为丢弃项（页眉页脚页码公式等，规则书无实质内容）
_LABEL_MAP = {
    'text': 'text', 'content': 'text', 'abstract': 'text',
    'footnote': 'text', 'reference': 'text', 'reference_content': 'text',
    'paragraph_title': 'title', 'doc_title': 'title',
    'aside_text': 'side',
    'table': 'table',
    'image': 'image', 'chart': 'image', 'seal': 'image',
    'figure_title': 'caption',
    'number': 'furniture', 'header': 'furniture', 'footer': 'furniture',
    'formula': 'furniture', 'formula_number': 'furniture', 'algorithm': 'furniture',
}

# 参与分栏聚簇的块类别
_COLUMN_CATEGORIES = {'text', 'title', 'caption'}

# 栏间隙阈值：以正文块"中位宽度"为尺度基准（尺度无关，兼容大画布页）
_GUTTER_MED_RATIO = 0.8
# 分栏聚簇时忽略超宽块（跨栏段落会把栏中心连成串）
_WIDE_BLOCK_RATIO = 1.8
# 标题跨栏判定：块宽超过中位正文块宽该比例视为跨栏标题
_SPAN_TITLE_MED_RATIO = 1.5
# 图注合并距离：与上方最近图片/表格的纵向距离不超过图注高度的该倍数
_CAPTION_MERGE_RATIO = 2.5


class LayoutAnalyzer:
    """版面分析器。初始化加载模型（首次运行需联网下载权重，之后离线）。

    fast=True 使用轻量 OCR 模型（PP-OCRv5 mobile 检测/识别），
    识别速度约为质量模式的数倍，准确率略降；版面模型无轻量 ONNX 包，
    两种模式共用 PP-DocLayout_plus-L。扫描件质量差时建议质量模式。
    """

    def __init__(self, fast=False, intra_threads=None):
        # 延迟导入，避免未安装依赖时 import 模块即报错
        from paddleocr import PPStructureV3
        self.fast = fast
        common = dict(
            layout_detection_model_name='PP-DocLayout_plus-L',
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            use_seal_recognition=False,
            use_table_recognition=False,
            use_formula_recognition=False,
            use_chart_recognition=False,
            use_region_detection=False,
            text_recognition_batch_size=16,
            # 本机 paddle 3.3.1 原生推理有 bug，固定 onnxruntime 引擎
            engine='onnxruntime',
        )
        if intra_threads:
            # 多线程并行时限制每个会话的核数，避免各会话抢满全部 CPU
            common['engine_config'] = {
                'onnxruntime': {'intra_op_num_threads': int(intra_threads)}}
        try:
            self.pipe = PPStructureV3(
                text_detection_model_name=('PP-OCRv5_mobile_det' if fast else None),
                text_recognition_model_name=('PP-OCRv5_mobile_rec' if fast else None),
                **common)
        except Exception:
            if fast:
                # 轻量模型 ONNX 包官方源不可用（404），回退质量模型
                self.fast = False
                self.pipe = PPStructureV3(**common)
            else:
                raise

    def analyze(self, img_path):
        """分析单页版面（不含 OCR 行数据）。见 _parse_layout。"""
        res = self.pipe.predict(img_path)[0].json['res']
        return self._parse_layout(img_path, res)

    def analyze_with_ocr(self, img_path):
        """分析单页版面 + 返回整页 OCR 行数据（行文本/置信度/坐标）。

        返回 (layout, lines)，lines 为
        [{'text': str, 'score': float, 'bbox': [x0,y0,x1,y1]}, ...]
        与 analyze() 共用同一次推理，不额外耗时。
        """
        res = self.pipe.predict(img_path)[0].json['res']
        layout = self._parse_layout(img_path, res)
        o = res.get('overall_ocr_res') or {}
        texts = o.get('rec_texts') or []
        scores = o.get('rec_scores')
        boxes = o.get('rec_boxes')
        lines = []
        if scores is not None and boxes is not None and len(texts):
            lines = [{'text': t, 'score': float(s),
                      'bbox': [float(v) for v in b]}
                     for t, s, b in zip(texts, scores, boxes)]
        return layout, lines

    def _parse_layout(self, img_path, res):
        """版面解析：分类、分栏、阅读顺序。

        返回 dict:
            page: 图像路径
            width/height: 页面尺寸
            columns: 检测到的栏 x 范围列表 [[x0, x1], ...]（无分栏则为空）
            blocks: 按阅读顺序排列的区域列表，每项
                {order, category, column, bbox[x0,y0,x1,y1], score}
        """
        page_w, page_h = res['width'], res['height']

        blocks = []
        for box in res['layout_det_res']['boxes']:
            label = box['label']
            cat = _LABEL_MAP.get(label, 'text')  # 未知标签按正文处理
            if cat == 'furniture':
                continue
            blocks.append({
                'category': cat,
                'bbox': [float(v) for v in box['coordinate']],
                'score': float(box['score']),
                'label': label,
            })
        if not blocks:
            return {'page': img_path, 'width': page_w, 'height': page_h,
                    'columns': [], 'blocks': []}

        _dedup_overlaps(blocks)

        # 线框表格补充检测：PP-DocLayout 对细线表格常漏检。
        # 整页级大块（装饰边框）过滤；与模型 table 块重叠的跳过。
        from ruled_table_detector import detect_tables
        for tb in detect_tables(img_path):
            if (tb[2] - tb[0]) * (tb[3] - tb[1]) > page_w * page_h * 0.85:
                continue
            if any(_iou(tb, b['bbox']) > 0.3
                   for b in blocks if b['category'] == 'table'):
                continue
            blocks.append({'category': 'table', 'bbox': tb, 'score': 1.0,
                           'label': 'ruled_table', 'column': None})

        # 跨栏标题先于分栏判定：宽度显著超过正文块中位宽度的标题不参与分栏
        med_w = _median_block_width(blocks)
        for b in blocks:
            if b['category'] == 'title' and med_w:
                if b['bbox'][2] - b['bbox'][0] > med_w * _SPAN_TITLE_MED_RATIO:
                    b['category'] = 'span_title'

        columns = _detect_columns(blocks, page_w)

        # 标题跨栏补充判定：横跨多栏的栏宽标题
        for b in blocks:
            if b['category'] == 'title' and columns:
                overlapped = sum(1 for c in columns
                                 if b['bbox'][0] < c[1] and b['bbox'][2] > c[0])
                if overlapped >= 2:
                    b['category'] = 'span_title'

        # 栏分配：正文类块就近入栏；够不到的 -> 宽块保持独立，窄块入 side
        for b in blocks:
            b['column'] = None
            if b['category'] in _COLUMN_CATEGORIES and columns:
                cx = (b['bbox'][0] + b['bbox'][2]) / 2
                col_i = min(range(len(columns)),
                            key=lambda i: abs(cx - (columns[i][0] + columns[i][1]) / 2))
                half = (columns[col_i][1] - columns[col_i][0]) / 2
                if abs(cx - (columns[col_i][0] + columns[col_i][1]) / 2) <= half * 1.2:
                    b['column'] = col_i
            if b['category'] not in ('table', 'image', 'side') and b['column'] is None:
                w = b['bbox'][2] - b['bbox'][0]
                if not (med_w and w > med_w * _SPAN_TITLE_MED_RATIO):
                    b['category'] = 'side'  # 窄栏外文本块（侧边栏/图片说明框）

        _merge_captions(blocks)

        ordered = _reading_order(blocks, columns, page_h)
        return {'page': img_path, 'width': page_w, 'height': page_h,
                'columns': columns,
                'blocks': [{**b, 'order': i} for i, b in enumerate(ordered)]}


def _detect_columns(blocks, page_w):
    """按正文类块 x 中心聚簇切栏。返回 [[x0,x1],...]，单栏或无栏返回 []。

    尺度无关：间隙阈值基于中位块宽。超宽块（跨栏段落等）不参与聚簇，
    避免把左右栏的中心连成一串。
    """
    candidates = [b for b in blocks if b['category'] in _COLUMN_CATEGORIES]
    if len(candidates) < 3:
        return []
    widths = sorted(b['bbox'][2] - b['bbox'][0] for b in candidates)
    med_w = widths[len(widths) // 2]
    narrow = [b for b in candidates
              if b['bbox'][2] - b['bbox'][0] <= med_w * _WIDE_BLOCK_RATIO]
    centers = sorted((b['bbox'][0] + b['bbox'][2]) / 2 for b in narrow)
    if len(centers) < 3:
        return []
    # 两遍中位数：阈值基于"窄块"宽度的中位数，避免宽段落/标题抬高阈值
    narrow_ws = sorted(b['bbox'][2] - b['bbox'][0] for b in narrow)
    narrow_med = narrow_ws[len(narrow_ws) // 2]
    gap = max(narrow_med * _GUTTER_MED_RATIO, page_w * 0.05)
    clusters, cur = [], [centers[0]]
    for c in centers[1:]:
        if c - cur[-1] > gap:
            clusters.append(cur)
            cur = [c]
        else:
            cur.append(c)
    clusters.append(cur)
    columns = []
    for cl in clusters:
        if len(cl) >= 3:  # 至少 3 个块才算一栏，过滤零散噪声
            members = [b for b in narrow
                       if (b['bbox'][0] + b['bbox'][2]) / 2 in cl]
            columns.append([min(b['bbox'][0] for b in members),
                            max(b['bbox'][2] for b in members)])
    return columns if len(columns) >= 2 else []


def _dedup_overlaps(blocks):
    """重叠去重：文本类块（text/title/caption）IoU 过高时保留得分高的，
    防止同区域被重复识别造成输出重复文本。"""
    textish = [b for b in blocks if b['category'] in ('text', 'title', 'caption')]
    removed_ids = set()
    for i, a in enumerate(blocks):
        if id(a) in removed_ids or a not in textish:
            continue
        for b in blocks[i + 1:]:
            if id(b) in removed_ids or b not in textish:
                continue
            inter = _iou(a['bbox'], b['bbox'])
            if inter > 0.6:
                loser = b if a['score'] >= b['score'] else a
                removed_ids.add(id(loser))
    blocks[:] = [b for b in blocks if id(b) not in removed_ids]


def _iou(b1, b2):
    x0 = max(b1[0], b2[0]); y0 = max(b1[1], b2[1])
    x1 = min(b1[2], b2[2]); y1 = min(b1[3], b2[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    union = ((b1[2] - b1[0]) * (b1[3] - b1[1])
             + (b2[2] - b2[0]) * (b2[3] - b2[1]) - inter)
    return inter / union if union > 0 else 0.0


def _median_block_width(blocks):
    widths = sorted(b['bbox'][2] - b['bbox'][0]
                    for b in blocks if b['category'] in _COLUMN_CATEGORIES)
    return widths[len(widths) // 2] if widths else None


def _merge_captions(blocks):
    """figure_title 就近挂到上方最近的图片/表格上。"""
    anchors = [b for b in blocks if b['category'] in ('image', 'table')]
    for b in blocks:
        if b['category'] != 'caption':
            continue
        y0 = b['bbox'][1]
        above = [a for a in anchors if a['bbox'][3] <= y0]
        if not above:
            continue
        best = min(above, key=lambda a: (
            abs((a['bbox'][0] + a['bbox'][2]) / 2 - (b['bbox'][0] + b['bbox'][2]) / 2)
            + (y0 - a['bbox'][3]) * 3))
        if y0 - best['bbox'][3] <= (b['bbox'][3] - y0) * _CAPTION_MERGE_RATIO:
            b['category'] = 'caption_of'
            b['anchor_order_key'] = (best['bbox'][1], best['bbox'][0])


def _reading_order(blocks, columns, page_h):
    """按纵向带推进：带内先左栏后右栏；独立单元（跨栏标题/侧栏/表格/图片）
    按其 y 位置插入阅读流。"""
    col_blocks = [[] for _ in columns]
    standalone = []
    for b in blocks:
        if b.get('category') == 'caption_of':
            standalone.append(b)  # 已挂靠图表的图注随锚点走
        elif b['category'] in ('table', 'image', 'side', 'span_title'):
            standalone.append(b)
        elif b['column'] is not None:
            col_blocks[b['column']].append(b)
        else:
            standalone.append(b)

    for col in col_blocks:
        col.sort(key=lambda b: (b['bbox'][1], b['bbox'][0]))
    standalone.sort(key=lambda b: (b['bbox'][1], b['bbox'][0]))

    ordered, ptr = [], [0] * len(col_blocks)
    for unit in standalone:
        y = unit['bbox'][1]
        for ci, col in enumerate(col_blocks):  # 带内：先左栏后右栏
            while ptr[ci] < len(col) and col[ptr[ci]]['bbox'][1] < y:
                ordered.append(col[ptr[ci]])
                ptr[ci] += 1
        ordered.append(unit)
    for ci, col in enumerate(col_blocks):      # 尾部：剩余栏内容
        ordered.extend(col[ptr[ci]:])
    return ordered
