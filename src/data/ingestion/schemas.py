"""摄取层数据结构：解析原文 → 结构章节 → 候选切片 → 质检结论，字段与 knowledge_chunk 表对齐"""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RawDocument:
    """单个文件经 reader 抽取后的纯文本中间结果（readers.py 产出）"""
    source_path: Path
    title: str                 # 缺省取文件名（去后缀），作为 heading_path 首级
    text: str                  # 抽取全文（逐字保留，不做任何改写）
    page_count: int | None = None  # PDF/PPT 页数，用于扫描件判定；文本类为 None
    empty_pages: int | None = None  # 无有效文本的空白页数（仅 PDF 回填，半扫描件的空页率判定用）


@dataclass
class Section:
    """结构化感知结果：一段带章节定位的原文（structuring.py 产出）"""
    heading_path: str          # 如 第二章 交易时间 > 2.4 交易限制；要素通道为 文档标题 > 要素名
    clause_position: str | None  # 如 2.4.2 或 第三条；无法识别/要素通道为 None
    text: str
    chunk_type_hint: str | None = None  # 要素通道直接给出的 chunk_type（高于关键词粗分精度）；制度文书通道为 None
    soft_clause: bool = False  # 软条款（cn_mode 「一、/案例一：」序号项）：短项豁免碎片质检，超长仍可按段落软切，不受「第X条不可分」约束


@dataclass
class ChunkDraft:
    """候选知识切片：字段语义与 knowledge_chunk 表一致，loaders.py 负责最终映射落库"""
    source_path: Path
    heading_path: str
    clause_position: str | None
    chunk_type: str            # 沿用表 CHECK 枚举：clause/table/exception/...
    content: str               # 检索用文本（含章节路径前缀，解决跨切片主体割裂）
    original_text: str         # 原文逐字摘录
    issues: list[str] = field(default_factory=list)  # quality.py 检出问题，空=合格
    is_element: bool = False   # 要素通道产出标记：完整语义要素允许短于 MIN_CHUNK_CHARS，不判碎片
    keywords: list[str] = field(default_factory=list)  # keywords.py 自动抽取结果，空则落库 NULL

    @property
    def suggested_status(self) -> str:
        """质检结论映射到表 status：有问题进 need_review 隔离带，合格才 valid"""
        return "need_review" if self.issues else "valid"


@dataclass
class FileSnapshot:
    """投放目录文件的当前状态快照（versioning.py 用于变更检测）"""
    path: Path
    mtime: float
    sha256: str
