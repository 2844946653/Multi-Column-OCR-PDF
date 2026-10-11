# 复杂排版 PDF 提取工具

本地离线提取《战锤40K》规则书、杂志扫描件等**复杂排版** PDF/图像的文本。
核心原则：**高保真提取，绝对禁止 AI 幻觉**——低置信度内容绝不猜测输出，
而是截图留待人工复核。

设计文档见 `设计文档.md`，进度与已知问题见 `进度总结.md`。

## 能力

| 场景 | 处理方式 |
|---|---|
| 数字版 PDF（有文本层） | 直接提取文本层，100% 准确，不跑 OCR |
| 扫描件/纯图片 PDF | 300 DPI 渲染 → 版面分析 → OCR → 置信度分级 |
| 多栏排版 | 版面分析按栏排序阅读顺序，绝不左右串行 |
| 表格 | v1：区域识别 + 内容平文本保留（结构化 Markdown 输出属二期） |
| 花体/哥特体/漫画气泡 | 检测失败区域整块截图进人工复核，不猜 |
| 低置信度文本 | <0.70 的行禁出原文，截图 + 占位符 `[WAITING_FOR_HUMAN_INPUT: 路径]`；0.70-0.90 标 `{?原文?}` |

## 安装

要求 Python 3.10+，Windows/Linux 均可。**无需梯子**，全部走国内直连源：

```bash
pip install pymupdf paddlepaddle paddleocr "paddlex[ocr]" onnxruntime
```

首次运行时自动下载模型权重（版面分析 PP-DocLayout_plus-L + PP-OCRv5 检测/识别，
共约 1GB，存于用户目录 `~/.paddlex/`），之后**完全离线**运行。

> 注意：PaddlePaddle 3.3.1 在部分 Windows 机器上的原生推理有 bug
> （oneDNN PIR 报错/动态图段错误），本项目已固定使用 `engine='onnxruntime'`
> 绕行，无需额外配置。

## 使用

**桌面界面（推荐）**：

```bash
pip install pywebview   # 若安装慢可加 -i https://pypi.tuna.tsinghua.edu.cn/simple
python src/app.py       # 弹出桌面窗口（Windows 用系统自带 WebView2）
```

界面三个页面：
- **提取**：多选 PDF 文件（或直接拖 PDF 进窗口）→ 一键提取 → 实时进度
  （模型加载/第几本书/逐页 OCR）；多个 PDF 逐个处理，每本书输出到独立的
  子文件夹，互不合并。支持**页码范围**（1 起始闭区间，留空=全部）、
  **快速/质量模式**（快速=轻量模型，单页约 4-5s）、
  **线程数 1-8**（多线程并行 OCR，默认 4，每线程一份模型）、
  **断点续跑**（中断后重跑自动跳过已完成页）、
  **纯图页自动跳过**（像素统计毫秒级判定，不跑模型）
- 渲染默认 200 DPI（平衡速度/体积），中间 PNG **用完即删**，
  输出只留文本、复核截图与报告
- **结果**：浏览每本书的 extracted_text.md 与处理统计
- **人工复核**：待复核截图一张张显示，输入正确文本 → 保存校正 → 导出
  corrected_text.md（占位符自动替换为校正文本；未校正的占位符保留）

**命令行**：

```bash
python src/run.py <输入文件夹> [输出文件夹]
```

输入文件夹放 PDF（可多个），输出结构：

```
output/<PDF名>/
├── probe_report.json      逐页分类报告（文本层/OCR）
├── text_layer/pNNN.txt    数字版页面文本（直取）
├── render/pNNN.png        扫描页 300 DPI 渲染
├── ocr_text/pNNN.txt      版面+OCR+置信度分级后的文本
├── needs_human_review/    低置信度/花体区域的截图
├── extracted_text.md      整本书文本（按页码拼接）
└── report.json            处理统计 + 每个待复核区域的页码/坐标/类别/置信度
```

## 测试

```bash
python tests/test_step1.py   # 文本层探测+渲染（秒级）
python tests/test_step2.py   # 版面分析（每页1-3分钟，CPU整页OCR）
python tests/test_step3.py   # OCR+防幻觉不变量
python tests/test_step4.py   # 表格分支 v1
python tests/test_step5.py   # 输出归档（秒级，合成fixture）
```

样张测试集在 `testset/`（26 条样本，自包含，不依赖源盘），
由 `testset/build_testset.py` 生成，`testset/make_scan_subset.py` 生成扫描件子集。

## 已知限制（如实记录）

1. **表格结构化输出属二期**：PP-Structure 表格模型（SLANet）的 ONNX 包在官方源
   不可用（404），本机 paddle 原生推理又有 bug；网格方案 `src/table_extractor.py`
   已实现但需视觉调参。当前表格内容以平文本完整输出，不丢不猜。
2. **线框表格检测器在装饰边框密集的扫描页有碎片级误报**（整页级已过滤）。
3. **置信度阈值（0.90/0.70）为初始值**，需按真实样张校准（见进度总结）。
4. 单页推理约 1-3 分钟（CPU + 整页 OCR + RT-DETR 版面检测），批量处理请耐心。
5. 扫描件本身分辨率过低（<150 DPI）时 OCR 质量受限——问题在源，不在工具。

## 二期规划

- 结构化表格（数字版用文本层词坐标重建网格；扫描版调通图像网格方案）
- 置信度阈值校准
- 轻量化：一键安装脚本；剥离 paddle 运行时直连 ONNX（依赖仅剩 onnxruntime）
- 接入翻译管线（mars_trilogy）：提取文本 → chunker → 本地 Ollama 翻译
