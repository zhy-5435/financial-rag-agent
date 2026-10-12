"""质量门控（入库前三检 + 扫描件隔离通道）：文档级「扫描件/乱码/章节丢失/完整度」+ 切片级「低质」

文档级问题（整篇不可信）通过 annotate(doc_issues) 强制全篇切片进 need_review 隔离带，
防止一份损坏文档 valid/碎片混写；切片级问题仅隔离单片。检索两路只取 valid，
隔离带内容天然不进召回；扫描件通道不做自动 OCR（重依赖+护栏不允许改写原文），隔离后转人工。
"""
import re

from src.data.ingestion.config import (
    CLAUSE_SPLIT_MAX_CHARS,
    EXTRACTION_COMPLETENESS,
    GARBLED_RATIO,
    MIN_CHUNK_CHARS,
    SCANNED_EMPTY_PAGE_RATIO,
    SCANNED_MIN_CHARS_PER_PAGE,
    SCANNED_MIN_PAGES,
    VERTICAL_LINE_RATIO,
    VERTICAL_MIN_LINES,
)
from src.data.ingestion.schemas import ChunkDraft, RawDocument, Section
from src.data.ingestion.structuring import chapter_token, chapters_in_toc

# 允许字符集：中文、字母、数字、常用中英文标点/括号/百分号/小数点等（其余视为噪声字符）
_ALLOWED = re.compile(
    r"[\u4e00-\u9fffA-Za-z0-9"
    r"\s，。、；：！？,.;:!?()（）《》\u3000“”‘’\"'`~\-—–_/\\%‰+#*&=<>|【】〔〕·\.]"
)


def check_document(doc: RawDocument) -> list[str]:
    """文档级检查：疑似扫描件（均值/空页率双判）/ 整篇乱码 / 逐字断行抽取损坏 / 抽取完整度过低，
    返回问题列表（空=通过）"""
    issues: list[str] = []
    if not doc.text:
        issues.append("抽取结果为空文本")
        return issues
    # 扫描件检测①：仅对有页数的格式（PDF/PPT）生效，平均每页字符过少判为图片版
    if doc.page_count and len(doc.text) / doc.page_count < SCANNED_MIN_CHARS_PER_PAGE:
        issues.append(
            f"疑似扫描件：平均每页仅 {len(doc.text) // doc.page_count} 字符，应转 OCR/人工通道"
        )
    # 扫描件检测②（半扫描防漏）：少量文字页会拉高均值遮蔽图片页，改看空白页占比
    if (doc.page_count and doc.empty_pages is not None
            and doc.page_count >= SCANNED_MIN_PAGES
            and doc.empty_pages / doc.page_count >= SCANNED_EMPTY_PAGE_RATIO):
        issues.append(
            f"疑似扫描件：{doc.empty_pages}/{doc.page_count} 页无可抽取文本"
            f"（空页率 {doc.empty_pages / doc.page_count:.0%}），应转 OCR/人工通道"
        )
    # 文档级乱码：抽取文本整体乱码率超阈判编码损坏（如二进制误当文本读入），
    # 整篇隔离而非逐片判——防部分片乱码率恰好压线漏网混进主库
    ratio = _garbled_ratio(doc.text)
    if ratio > GARBLED_RATIO:
        issues.append(f"文档级乱码率 {ratio:.0%} 超阈值 {GARBLED_RATIO:.0%}，疑似编码/抽取损坏")
    # 逐字断行检测：竖排/文本框错位抽取产出一字一行碎片（字符全合法、乱码门控拦不住），
    # 整篇隔离防 300 字/片的竖排垃圾进检索库；修复属抽取层选型（待 pymupdf4llm 评估）
    nonblank = [ln for ln in doc.text.splitlines() if ln.strip()]
    if len(nonblank) >= VERTICAL_MIN_LINES:
        tiny_ratio = sum(1 for ln in nonblank if len(ln.strip()) <= 2) / len(nonblank)
        if tiny_ratio >= VERTICAL_LINE_RATIO:
            issues.append(
                f"疑似逐字断行抽取损坏：1~2 字短行占比 {tiny_ratio:.0%}"
                f"（≥{VERTICAL_LINE_RATIO:.0%}），应重选抽取器或转人工"
            )
    # 解析完整度对账：仅限纯文本类（txt/md 字节量≈字符量）；PDF/PPT 为压缩二进制无可比性，
    # 整章丢失由 check_chapters 目录比对覆盖
    suffix = doc.source_path.suffix.lower()
    file_chars = doc.source_path.stat().st_size if doc.source_path.exists() else 0
    if suffix in (".txt", ".md", ".markdown") and file_chars and len(doc.text) < file_chars / 3 * EXTRACTION_COMPLETENESS:
        issues.append(
            f"抽取完整度存疑：文本 {len(doc.text)} 字符 vs 文件 {file_chars} 字节"
        )
    return issues


def check_chapters(doc: RawDocument, sections: list[Section]) -> list[str]:
    """章节对账（Phase 3 完整度方案）：含目录的文档，目录声明的每个「第X章」都应在解析结果
    中出现；整章缺席→疑似抽取丢章（目录缺失的文档不判，零误报优先）"""
    declared = chapters_in_toc(doc.text)
    if not declared:
        return []
    found = {t for s in sections for p in s.heading_path.split(" > ") if (t := chapter_token(p))}
    return [f"目录声明「{c}」但正文未解析到，疑似整章丢失" for c in sorted(declared - found)]


def check_chunk(draft: ChunkDraft) -> list[str]:
    """切片级检查：乱码率 / 过短碎片 / 超硬上限未切分，问题切片应进 need_review 隔离带"""
    issues: list[str] = []
    text = draft.original_text
    # 短切片豁免两类：①带条款号（如「1.3　证券交易遵循公开、公平、公正的原则。」22 字，
    # 完整法定条款非碎片，人工金标存在大量同类短条款）；②要素通道片（如「业绩比较基准」
    # 一行即一个完整语义要素，短是常态）
    if len(text) < MIN_CHUNK_CHARS and not draft.clause_position and not draft.is_element:
        issues.append(f"切片过短（{len(text)} < {MIN_CHUNK_CHARS}），疑为边界碎片")
    # 硬上限复检：任何切片不得超 CLAUSE_SPLIT_MAX_CHARS(500)，漏切（如超大表格整节、
    # 切块链异常直出）在此接住进隔离带，保证 valid 检索集长度不超 500
    if len(text) > CLAUSE_SPLIT_MAX_CHARS:
        issues.append(f"切片超硬上限（{len(text)} > {CLAUSE_SPLIT_MAX_CHARS}），切块链漏切，应人工核对切分")
    # 乱码率：非允许字符占比（允许集外的控制符/异体乱码等）
    ratio = _garbled_ratio(text)
    if ratio > GARBLED_RATIO:
        issues.append(f"乱码率 {ratio:.0%} 超阈值 {GARBLED_RATIO:.0%}")
    return issues


def _garbled_ratio(text: str) -> float:
    if not text:
        return 0.0
    return len(_ALLOWED.sub("", text)) / len(text)


def annotate(drafts: list[ChunkDraft], doc_issues: list[str] | None = None) -> list[ChunkDraft]:
    """合并文档级+切片级检查写回 draft.issues：文档级问题（扫描件/乱码/丢章）强制
    全篇切片带「[整篇问题]」前缀进 need_review 隔离带，整篇不可信不允许碎片混写 valid"""
    prefixed = [f"[整篇问题] {i}" for i in (doc_issues or [])]
    for d in drafts:
        d.issues = prefixed + check_chunk(d)
    return drafts
