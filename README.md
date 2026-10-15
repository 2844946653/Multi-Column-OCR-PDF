# Multi-Column OCR

面向复杂排版 PDF 与扫描件的本地 OCR 文本提取工具，基于 [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) 构建。

Multi-Column OCR extracts text from PDFs and page images whose layout defeats
generic OCR tools: multi-column articles, vertical text, sidebars, tables and
full-width headers. It runs entirely offline.

## 项目定位

该工具用于解决的具体问题是**复杂排版文档的文本提取**。常见的 OCR 工具处理
规则书、杂志这类版面时，多栏内容会左右串行、竖排文字无法识别、低置信度
文本要么被猜出来要么被丢弃。

本项目对此类问题处理如下：

- **多栏阅读顺序**：版面分析按栏排序，先读完整栏再读下一栏
- **竖排文字**：旋转重识别，侧栏竖排、书脊字可以正常提取
- **置信度分级**：极易错误识别的低置信度文本（通常为漫画字、手写花体这类浮夸文本）保留在正文中并就地标记，①不会静默丢弃，
  ②不会猜测输出；低置信度文本截图进入人工复核区交由用户处理
- **本地离线**：全部推理在本机完成，运行时无网络请求

项目不包含模型训练能力，也不是通用 OCR 引擎。表格内容以文本形式输出，
表格结构化（Markdown 表格）在计划中。

## 能力

| 输入 | 处理方式                                                           |
|---|----------------------------------------------------------------|
| 数字版 PDF（含文本层） | 直接提取文本层，不运行 OCR                                                |
| 扫描件 / 纯图片 PDF | 渲染后经版面分析与 OCR 提取                                               |
| 多栏排版 | 按栏排序阅读顺序，不左右串行                                                 |
| 竖排 / 旋转文字 | 自动旋转重识别                                                        |
| 低置信度文本 | 保留并标记：≥0.90 直接输出；0.60-0.90 标 `{?原文?}`；<0.60 标 `{!原文!}` 并截图进校对清单 |
| 花体 / 漫画字等 | 检测失败区域整块截图进人工复核，工具不会对其进行猜测输出                                   |

## 安装

需要 Python 3.10+。依赖：

```bash
pip install pymupdf paddlepaddle paddleocr "paddlex[ocr]" onnxruntime
```

首次运行自动下载模型权重（存放于用户目录 `~/.paddlex/`），之后完全离线。

注意：PaddlePaddle 3.3.1 在部分 Windows 机器上原生推理有已知问题
（oneDNN PIR 报错或段错误），本项目固定使用 `engine='onnxruntime'` 绕开。

## 使用

**桌面界面**：

```bash
pip install pywebview
python src/app.py
```

界面提供三个页面：提取（多选/拖拽 PDF、页码范围、断点续跑）、结果浏览、
人工复核（逐张查看低置信度截图并回填校正，导出校正版全文）。

**命令行**：

```bash
python src/run.py <输入文件夹> [输出文件夹]
```

输出结构：

```
output/<PDF名>/
├── probe_report.json      逐页分类报告（文本层/OCR）
├── text_layer/pNNN.txt    数字版页面文本
├── ocr_text/pNNN.txt      扫描页提取文本
├── needs_human_review/    低置信度/花体区域截图
├── extracted_text.md      整本书文本
└── report.json            处理统计与复核清单
```

## 性能

以下数据在本机实测（i7-12700H，CPU 推理，Windows 11）：

| 指标 | 数据 |
|---|---|
| 单页耗时 | 357s → 13s（抢救管线重构后，快速模式） |
| 模型级联 | tiny 重识别 0.16s/张，server 2.8s/张；边界分才用大模型 |
| 内存占用 | 多线程并行优化后约 3.5GB（4 线程） |
| 复核误报 | 351 项 → 51 项（剩余项经最强模型验证，50/51 为真实难例） |

## 工程细节

- **输入兼容性**：支持 RGB、灰度、RGBA（透明通道）PNG、JPEG、WebP 及
  扩展名与实际格式不符的图片；透明通道与异常格式在预处理阶段统一归一化
- **多线程推理**：每个工作线程持有独立模型实例，低分行抢救管线全进程共享
  一份（4 线程运行时内存约 3.5GB；若各线程各自加载大模型，约 8GB）
- **断点续跑**：中断后重新运行自动跳过已完成页面；同名但来源不同的 PDF
  按文件指纹分目录输出，互不覆盖
- **置信度抢救**：低分文本行由三档模型级联重识别（快速模型初筛 → tiny
  小模型重读 → 边界分由大模型仲裁），减少"可识别却被判定为不可识别"的情况

## 测试

```bash
python tests/test_step1.py   # 文本层探测与渲染
python tests/test_step2.py   # 版面分析与阅读顺序
python tests/test_step3.py   # OCR 与置信度分级不变量
python tests/test_step4.py   # 表格区域内容提取
python tests/test_step5.py   # 输出归档
python tests/test_gui.py     # 校正回填闭环
```

测试集说明见 `testset/README.md`。

## 已知限制

- 识别链路本身虽然可兼容常见图片格式（PNG/JPEG 等）直接作为输入，
  但不作为程序主要处理方向，测试不多，运行结果请注意甄别
- 表格结构化输出（Markdown 表格）尚未完成，当前以文本形式输出表格内容
- 置信度阈值（0.90/0.60）为初始值，未在更大样本上校准
- 线框表格检测器在装饰边框密集的扫描页上可能有碎片级误报
- 源文件分辨率过低（低于约 150 DPI）时识别质量受限

## 许可证

MIT License，见 [LICENSE](LICENSE)。

本项目使用 PaddleOCR（Apache-2.0）等开源组件，各组件遵循其自身许可证。
