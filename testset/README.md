# 样张测试集（Step 0）

供「复杂排版 PDF 提取工具」各里程碑回归测试使用。设计文档见上级目录 `设计文档.md`。

## 组成（26 条样本，由 build_testset.py 生成）

| 目录 | 来源 | 内容 | 测什么 |
|---|---|---|---|
| `sources/01_story_pages.pdf` | 厄兆方舟：阿巴顿（剧情部分）.pdf | 6 页数字版子集 | 文本层探测、直取、阅读顺序 |
| `sources/02_scan_pages.pdf` + `samples/02_scan/` | 厄兆方舟：瓦什托尔.pdf（纯扫描件） | 8 页子集 PDF + 8 页 300 DPI PNG | 渲染→版面分析→OCR→置信度分级 全流程 |
| `sources/03_tables_pages.pdf` + `samples/03_tables/` | 圣血天使9版圣典.pdf | 5 页（表格候选） | 表格结构识别（文本层 + 光栅化两路） |
| `sources/04_magazine_pages.pdf` + `samples/04_magazine/` | White Dwarf 487 | 4 页（全书文本块最多） | 侧边栏、多栏、图片注释 |
| `samples/05_image_input/` | 用户手工拆分的图片 | 3 张真 PNG | 图片路径输入、格式探测 |

## 目录约定

- `samples/`：OCR 路径的输入（300 DPI PNG，自包含，不依赖 Z:\ 源文件）
- `sources/`：文本层路径的输入（页面子集 PDF）
- `gt/`：文本层真值（`get_text(sort=True)` 结果），数字版页面的回归比对基准
- `manifest.json`：每条样本的元数据（来源、页码、类别、预期难点）

## 回归方式

每个里程碑完成后，在 `testset/` 下运行该里程碑对应的测试脚本（后续随各 Step 创建），
比对输出与 `gt/` 真值（数字版页面）或人工检查 `needs_human_review` 行为（扫描页）。

## 注意

- 扫描件（02_scan）无真值文本，回归靠人工抽查 + report.json 统计。
- 03_tables 的 5 页是按"行/列词簇密度"启发式选出的候选，其中可能混有非表格页，以 PP-Structure 实际输出为准。
- 05 图片输入的原始文件是 WebP 伪装 .jpg（RIFF/WEBP 头），已统一转为真 PNG。
