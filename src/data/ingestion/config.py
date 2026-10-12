"""L1 摄取层专有配置：投放目录、切块与质量门控阈值、关键词自动抽取

Phase 0 仅使用解析/切块/质检相关阈值；版本快照与写库配置随 Phase 4 补充。
"""
import json
import os

from dotenv import load_dotenv

from src.config import DATA_DIR, PROJECT_ROOT

# 独立运行时自行加载 .env（与 agent 层同一份配置，环境变量名即契约，不构成跨层依赖；重复加载幂等）
load_dotenv(PROJECT_ROOT / ".env")

# 自动摄取投放目录：新文档（txt/md/pdf/pptx/docx）放入此处待入库
INCOMING_DIR = DATA_DIR / "incoming"

# 支持的文件后缀 → 读取器路由键（readers.py 按此分发，未列出的后缀在 --check 中标记跳过）
SUPPORTED_SUFFIXES = {".txt", ".md", ".markdown", ".pdf", ".pptx", ".docx"}

# ---------- 切块参数 ----------
# 长度双约束：SECTION_MAX_CHARS=300 为目标/基本上限（贴合 bge-base 512 token 编码窗口，约 300~350 汉字），
# CHUNK_HARD_MAX=500 为任何切片都不得突破的硬上限。基本单位=「节」：同一 章>节 下的条款合并为一个
# Chunk（heading_path 记到节级）；节内条款总字数超过 300 时按整条装箱拆为多片（每片目标 ≤300）。
SECTION_MAX_CHARS = 300
# 硬上限 500：条款原则上不可分，单条 300~500 落在兜底带仍整条成块；超过 500 的条款（含误粘连/异常
# 超长体——正式版实测产品说明书释义条粘连 2.1~6.6 万字符、及少数长法规条款）必须回退段落软切，
# 软切无标点处以 SECTION_MAX_CHARS 硬断兜底，确保无块 >500；quality.check_chunk 再复核硬上限，
# 任何漏切（含超大表格整节）赶进 need_review 隔离带，valid 检索集绝不超 500
CLAUSE_SPLIT_MAX_CHARS = 500
# 单切片字符数下限：低于此值视为碎片（如孤立标题行），并入相邻块或标记低质
MIN_CHUNK_CHARS = 40
# 超长章节二次切分时保留的上一块尾部字符数（保语义衔接）
CHUNK_OVERLAP_CHARS = 60

# ---------- 要素切分通道（产品说明书类，structuring.py 要素路由用） ----------
# 路由信号：正文 `**要素名**：` 标记数达下限且无行首「第X条」（制度文书不会误入）；
# 目录提示词仅作降档信号（须至少 1 个标记方可直接入通道——pypdf 抽取的产品类 PDF 无标记，
# 单靠目录命中会崩溃/误切，回退制度文书通道）
PRODUCT_DIR_HINTS = ("产品说明书",)
ELEMENT_MIN_MARKS = 4
# 要素名→chunk_type 直接映射（按序子串命中，全不中默认 product_element），精度高于关键词粗分
ELEMENT_TYPE_RULES: tuple[tuple[str, str], ...] = (
    ("风险等级", "product_element"),   # 需先于「风险」：风险等级是要素字段而非风险揭示段
    ("费率", "fee"),
    ("费用", "fee"),
    ("风险", "risk"),
    ("业绩", "product_element"),
    ("投资", "product_element"),
    ("规模", "product_element"),
    ("代码", "product_element"),
    ("流程", "process"),
    ("定义", "definition"),
    ("例外", "exception"),
)

# ---------- 质量门控阈值（quality.py 三检用） ----------
# 乱码判定：切片中「非 中文/字母/数字/常见标点」字符占比超过该值 → 低质
GARBLED_RATIO = 0.15
# 扫描件判定：PDF 平均每页可抽取字符数低于该值 → 疑似扫描图片版，转 OCR/人工
SCANNED_MIN_CHARS_PER_PAGE = 50
# 扫描件判定②（半扫描防漏）：总页数达下限且空白页占比超阈值同样隔离；
# 扫描件通道不做自动 OCR（重依赖与防幻觉护栏不允许改写原文），隔离后转人工/OCR 处理
SCANNED_MIN_PAGES = 3
SCANNED_EMPTY_PAGE_RATIO = 0.5
# 单页抽取文本低于该字符数计为「空白页」（页眉页脚级残留）
SCANNED_PAGE_MIN_CHARS = 10
# 逐字断行抽取损坏判定（正式版实测：pypdf 抽取竖排版式 PDF 产出一字一行的碎片文本，
# 字符集全部合法致乱码门控放行）：非空行数达下限且 1~2 字短行占比超阈 → 整篇隔离，
# 属抽取层选型问题非切块问题，待抽取器评估（pypdf vs pymupdf4llm）/转人工处理
VERTICAL_MIN_LINES = 200
VERTICAL_LINE_RATIO = 0.6
# 解析完整度判定：抽取总字符数 / (文件大小 × 系数) 低于该值 → 疑似整章丢失
EXTRACTION_COMPLETENESS = 0.3

# chunk_type 关键词启发式（chunking.py 粗分类，Phase 1 对齐人工金标后再调）
CHUNK_TYPE_HINTS = {
    "fee": ("费率", "费用", "管理费", "托管费", "申购", "赎回"),
    "risk": ("风险", "警示", "不利", "损失"),
    "definition": ("定义", "是指", "含义"),
    "exception": ("除外", "不适用", "免于"),
    "process": ("流程", "步骤", "办理", "程序"),
    "table": ("|---",),  # Markdown 表格分隔行
}

# ---------- 关键词自动抽取（keywords.py，混合召回增强） ----------
# 模式：llm=主路（可补正文未出现的同义词/简称，需 .env LLM 接入，单批失败自动退 rule）；
# rule=纯本地兜底（实体词+章节词+高频 n-gram，确定性、离线可复现，--check/演练用）；
# off=不抽取（维持 keywords NULL 的旧行为）
KEYWORDS_MODE = os.getenv("INGEST_KEYWORDS_MODE", "llm").strip().lower()
# 单切片关键词上限（防长尾噪声词稀释 FTS body 的 BM25 词频）
KEYWORDS_TOP_N = int(os.getenv("KEYWORDS_TOP_N", "8"))
# LLM 路单批切片数：约 300 字/片，10 片≈3000 字/次，平衡调用次数与单次响应截断风险
KEYWORDS_LLM_BATCH_ITEMS = 10
# 规则路 n-gram 仅保留片内出现次数达该值的词（出现一次视为偶然搭配非术语）
KEYWORDS_MIN_GRAM_FREQ = 2

# LLM 接入（与 agent/config.py 同名环境变量，单源 .env）
KW_LLM_MODEL = os.getenv("LLM_MODEL", "GLM-5.1")
KW_LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://tokenhub.tencentmaas.com/v1")
KW_LLM_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
KW_LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
# 端点扩展参数透传（如关闭推理模型思考模式），非法 JSON 按空处理
try:
    KW_LLM_EXTRA_BODY = json.loads(os.getenv("LLM_EXTRA_BODY") or "null")
except json.JSONDecodeError:
    KW_LLM_EXTRA_BODY = None
