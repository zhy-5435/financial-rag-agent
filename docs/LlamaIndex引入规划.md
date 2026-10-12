# LlamaIndex 引入规划：文档处理与自动化更新

> 日期：2026-10-06 ｜ 状态：规划评审稿
> 命题：评估在当前 L1 金融知识库引入 LlamaIndex，解决 PDF/表格/PPT 文档处理、结构化感知、切块策略、劣质文档处理、自动化文档更新的可行性与落地路径
>
> **阅读前提（后续变更）**：本文保留立项时的「现状 / 缺口」对照作为决策依据，其中关于
> `data/excel/L1_pro.xlsx` 与「Excel 人工通道保留、双通道共存」的描述**已不成立**：该规划落地后，
> 人工 Excel 导入通道已于代码层整体删除（`load_excel_to_db` / `_clean_*` / `EXCEL_PATH` / `openpyxl`），
> 切片唯一来源为 `src/data/ingestion/` 自动摄取，`python -m src.data.pipeline` 改为整库全量重建入口。
> 最新事实以 README 与 `docs/切块对齐评测报告.md` 为准。

## 一、背景与现状问题

当前 L1 数据流为 **`data/excel/L1_pro.xlsx`（人工预切片的权威源）→ SQLite 双表（doc_version / knowledge_chunk，含 FTS5 关键词索引）→ Milvus Lite 向量同步 → 双路混合检索（向量 0.8 + 关键词 0.2，网格搜索校准）**。

逐项对照本次命题，现状缺口如下：

| 能力项 | 现状 | 缺口 |
|---|---|---|
| 文档处理 | 仅支持 Excel 两个 sheet 导入；raw 目录只存人工转换好的 `.txt` | **无 PDF / PPT / Word 解析器**，一份新监管文件入库需人工"读原文→剪贴→切块→填 Excel"全流程 |
| 结构化感知 | `clause_position`、`heading_path` 全靠人工填写 | 无自动章节/条款号识别，人工填写易漏易错、无法扩展到大批量文档 |
| 切块策略 | 无切块逻辑（Excel 里已是成品切片） | 完全没有可编程的切块策略，切块质量不可复现、不可调参 |
| 劣质文档处理 | `verify_doc_files()` 仅做路径模糊对账；Excel 脏数据靠 `_clean_*` 逐条打补丁 | 无解析质量评估（乱码/扫描件/表格碎裂检测），坏数据只能进库后靠 CHECK 约束兜底 |
| 自动化文档更新 | `python -m src.data.pipeline` 全删重插式全量重建 | 文件变了不会自动感知；无版本递增、旧版置 `superseded` 的机制，每次更新都是人工全量操作 |

**一句话概括：知识库的"库存管理"（双库一致性、检索、过滤）已经做得很扎实，但"进货流水线"完全还是人肩挑。** LlamaIndex 强的正是进货流水线这一段。

## 二、可行性结论：引入，但只取其所短，限域使用

### 2.1 为什么不能"整体替换"

LlamaIndex 的默认形态是 `Document → Node → VectorStoreIndex → QueryEngine` 一条龙。若整体接管：

1. **绕过 SQLite 权威主库**。本项目核心不变量是"Excel/SQLite 为唯一权威源、双库强一致、全删重插幂等构建"（见《表结构设计.txt》与 `sqlite_client.py` 注释）。LlamaIndex 自带向量存储与检索，会架空这一套治理结构。
2. **丢失已调优的混合检索**。FTS5 中文单字分词 + BM25 + min-max 归一化加权（0.8/0.2）是经 14 条评测集网格搜索校准的稳健最优平台（见《检索验证报告》）；LlamaIndex 默认ANN 检索无法等价复现，且金融规则类文本的关键词精准召回（条款号、费率数字）不可妥协。
3. **检索前置过滤依赖 JOIN**。现有检索要过滤"切片/文档双有效 + 生效期内"（`keyword_search` 中 JOIN doc_version），这一逻辑建立在关系型主库上，换成 LlamaIndex 的 metadata filter 需要重建且不如 SQL 可控。

### 2.2 引入方式（核心决策）

**LlamaIndex 仅作为 L1 数据层内部的"文档摄取与切块引擎"，产出物仍是 `doc_version` / `knowledge_chunk` 两张表的标准行。** 类比：请一位会看 PDF/PPT 的"备菜厨师"，但菜单（表结构）、收银台（检索链路）、仓库（双库）一概不动，`src/data/hybrid_retriever.py`、`milvus_client.py`、L2/L3 各层 **零改动**。

具体选用模块（`llama-index-core` 即可覆盖大半，按需加 reader 包）：

| 需求 | LlamaIndex 组件 | 用途 |
|---|---|---|
| PDF 解析 | `PDFReader`（pypdf）→ 升级为 `MarkItDownReader` / pymupdf4llm | 文本型 PDF 抽取；金融监管 PDF 常见双栏/页眉页脚噪声，需对比选型 |
| PPT 解析 | `PPTReader`（python-pptx） | 投产品宣讲 PPT 的文本+表格抽取 |
| 表格感知 | `MarkdownNodeParser` + `Table.extract()` | 表格转 Markdown 保留行列结构，费率表整表成一个 `chunk_type='table'` 切片，不切碎 |
| Word/扫描件 | `ADFReader`/`MarkItDownReader`；OCR 走 PaddleOCR 兜底（LlamaIndex 生态外补充） | 见 2.3 劣质文档 |
| 结构化感知 | Markdown 标题树（`#`/`##` 层级）→ `heading_path`；章节号正则 → `clause_position` | 自动生成现在靠人工填的两个字段 |
| 切块策略 | `MarkdownNodeParser`（主）+ `SemanticSplitterNodeParser`（兜底）+ 自写 transformation | 层级切块为主，语义切块处理无标题长文 |
| 自动化更新 | `IngestionPipeline` + `docstore` + 转换链 | 文件 hash 变更检测、增量重跑切块链 |

注意：**不使用** `VectorStoreIndex` / `StorageContext` / `QueryEngine` 等检索侧组件，embedding 继续走本项目 `EmbeddingService`（BGE 前缀指令、本地模型目录、768 维约束均已固定，且已有"模型升级导致维度不匹配"的踩坑记录，不应再引入第二套向量通路）。

### 2.3 劣质文档处理策略（金融场景必须显式设计）

金融文档"劣质"主要表现为：扫描件图片 PDF、表格跨页碎裂、乱码/编码混杂、条款编号断档。LlamaIndex 不提供现成答案，需自建**入库前质量门控**，全程以"宁缺毋滥"为原则（与项目防幻觉基调一致）：

1. **扫描件检测**：PDF 平均每页字符数 < 阈值 → 判为疑似扫描件 → 走 OCR 通道或转人工，绝不把乱码写进 `content`；
2. **解析完整度对账**：自动切块数/字符量与原文规模比对，落差过大（如目录识别失败导致整章丢失）→ 该文档版本整体拒绝入库；
3. **低质切片隔离带**：单切片级质量不合格（正文乱码率、表格行列错位、长度异常）→ 插入 `knowledge_chunk` 但 `status='need_review'`（表结构已有此枚举值，且向量/关键词两路均只取 `valid`，天然隔离，**无需改任何检索代码**）；
4. **人工兜底闭环**：`need_review` 切片经人工修正后置 `valid`，即"自动流水线 + 人工质检岗"双轨，自动产出不直接进检索。

## 三、目标架构

```
                     ┌─────────────────────────────────────────────┐
 data/incoming/      │  LlamaIndex 摄取层（新增，仅 L1 数据层内部）  │
 ├─ *.pdf  ────────►│  TikaDirReader/自动路由                       │
 ├─ *.pptx ────────►│   ├─ 结构感知：标题树 → heading_path          │
 ├─ *.docx ────────►│   │            章节号正则 → clause_position   │
 └─ *.xlsx(数据表)  │   ├─ 切块链：Markdown → 表格保护 → 长度兜底    │
                     │   ├─ 质量门控：扫描件/完整度/低质三检          │
                     │   └─ 版本管理：hash 变更 → 新版本行+superseded│
                     └──────────────┬──────────────────────────────┘
                                    ▼ 写入（新增"自动摄取"通道）
                     SQLite doc_version / knowledge_chunk（权威主库，表结构不变）
                                    │  Excel 人工通道保留，双通道共存
                                    ▼
                     Milvus 向量同步 → 双路混合检索 → L2/L3（全部零改动）
```

关键设计：

- **双通道写入**：Excel 人工通道（现有 28 条精标切片）继续有效，自动通道新产切片 `knowledge_id` 使用独立前缀段（如 `AUTO-xxxx`），`keywords` 由 LLM/规则自动抽取生成；两通道数据同表同约束，检索侧无感知。
- **自动化更新语义**（复用现有表结构字段，不改 DDL）：
  1. 扫描 `data/incoming/` 与上次快照（文件路径+mtime+hash 存 SQLite 新增轻量 `ingest_state` 表）比对，找出新增/变更文件；
  2. 变更文件 → 按 `doc_id` 生成新 `doc_version_id`（版本递增），旧版本 `status` 置 `superseded`；旧切片自动置 `expired`（沿用现有 `get_valid_knowledge_with_meta` 的双有效过滤，旧版向量随下次同步自然退出 Milvus）；
  3. 触发增量向量同步（只插新 `valid` 切片、删失效切片，替代当前全删重插；全量重建命令保留作为对账兜底）；
  4. 幂等：同一 hash 重复执行无副作用，中途失败可重跑。

## 四、实施阶段计划

### Phase 0：依赖评估与骨架（1 天）
- `pip install llama-index-core`（进 `.venv`，入 `requirements.txt`），验证与 `pymilvus`/`sentence-transformers` 无依赖冲突；
- 新建 `src/data/ingestion/` 子包：`readers.py`（格式路由）、`structuring.py`（heading_path/clause_position 提取）、`chunking.py`（切块链）、`quality.py`（门控）、`versioning.py`（hash 检测与版本递增）、`loaders.py`（写 SQLite）；
- 交付物：空跑通 `python -m src.data.ingestion --check`（只打印解析结果不落库）。

### Phase 1：PDF/文本型文档自动切块，对齐现有切片（2~3 天）
- 用 raw 目录 4 份监管规则 `.txt` 的"原文 + Excel 人工切片"做**切块质量基准**：自动切块结果与人工金标注切片对比，统计 `heading_path`/`clause_position` 命中率、切片边界一致率；
- 调通 Markdown 层级切块 + 条款号正则（`第X条`/`X.Y.Z`）两条结构化提取规则；
- **验收门槛：与人工切片边界一致率 ≥ 80% 且结构字段命中 ≥ 90%，不达标不进入下一步**（切不准宁可继续人工通道）。

### Phase 2：表格与 PPT（2 天）
- 引入 MarkItDown/pymupdf4llm 将 PDF 表格转 Markdown，`MarkdownNodeParser` 保表整块，落 `chunk_type='table'`；
- `PPTReader` 覆盖产品宣讲材料，每 slide 一节、speaker notes 并入；
- 用 2 份含真实费率表的文档验证：表格行列不被切碎、表头与数据行同块。

### Phase 3：劣质文档门控 + 扫描件通道（2~3 天）
- 实现三检（扫描件/完整度/低质切片），`need_review` 隔离带打通；
- OCR 兜底通道（PaddleOCR 本地推理）处理扫描件，产出带"OCR 置信度"标注的切片；
- 劣质样例集验证：故意投喂乱码 txt、扫描版 PDF、碎裂表格，确认全部被拦截或隔离，无一条脏数据直接进 `valid`。

### Phase 4：自动化更新流水线（2 天）
- `ingest_state` 快照表 + 增量检测 + 版本递增 + 旧版 `superseded` 全流程；
- 触发方式：手动命令 `python -m src.data.ingestion` 起步；后续可选 watchdog 文件监听；
- 端到端演练：投放 1 份新规则 PDF → 自动入库 → 混合检索命中新切片 → 同 doc_id 旧版自动退出检索。

### Phase 5：回归评测与文档（1 天）
- 评测集从 14 条扩至 30+ 条（含自动摄取切片的多问法查询），复测 Hit@1/Hit@3/MRR 与双路权重，确认自动通道未稀释既有检索质量；
- 更新 README 目录树、启动命令与数据流图（遵循 README 真实性规范，示例数据来自实测）。

**合计约 10~11 个工作日**；各阶段独立可交付、可回退（自动通道数据可按 `AUTO-` 前缀整体清除，不影响 Excel 人工通道）。

## 五、风险与开放问题

| 风险 | 说明 | 缓解 |
|---|---|---|
| 中文解析质量不可控 | LlamaIndex reader 对中文监管 PDF 的双栏排版、条款编号效果未实测 | Phase 1 用金标对比设硬门槛，不达标即止损 |
| 切块粒度与人工精标差距 | 现有 28 条切片质量高（含逐字 `original_text`），自动切片检索表现可能拉低整体 | 双通道隔离评测；自动切片先进 `need_review` 抽检再放量 |
| 依赖引入副作用 | `llama-index-core` 依赖面较宽，可能与 `.venv` 现有包冲突 | 仅装 core + 最小 reader 包，先隔离环境验证再合并 requirements.txt |
| MilvusLite 独占锁 | 自动更新若在服务运行中写库会 `DataDirLockedError` | 摄取只写 SQLite；向量同步仍走"服务停止窗口"或后续升级为独立 Milvus Server |
| `original_text` 逐字性 | 人工规范要求原文逐字摘录，自动切块必须保证 content/original_text 均未经改写 | 切块链只做边界切割，禁用任何 LLM 改写步骤（与防幻觉护栏一致） |
| OCR 成本 | 扫描件通道依赖 PaddleOCR 本地模型（体积/首次加载慢） | 作为独立可选项延后到 Phase 3 末评估，不阻塞主线 |

## 六、结论

1. **可以引入，且引入边界清晰**：LlamaIndex 限定在"文档摄取与切块"职能上（readers / node parsers / transformations / IngestionPipeline 四组能力），检索与存储架构维持 SQLite 权威 + Milvus 向量 + 双路混合检索不变，下游 L2/L3 零改动。
2. **五个命题均有对应解法**：PDF/PPT/表格 → reader + Markdown 表格保护切块；结构化感知 → 标题树与条款号规则自动生成 `heading_path`/`clause_position`；劣质文档 → 三检门控 + `need_review` 隔离带（复用现有 status 枚举，检索天然过滤）；自动化更新 → hash 快照增量 + 版本递增 + `superseded` 机制（复用现有表结构）。
3. **最大不确定性在中文 PDF 的切块质量**，故计划以"与人工金标一致率 ≥ 80%"作为 Phase 1 硬门槛，将风险前置暴露；全阶段约 10~11 个工作日，双通道设计保证任一阶段可无损回退。
