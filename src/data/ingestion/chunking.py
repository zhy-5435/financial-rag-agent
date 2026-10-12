"""切块链：条级 Section → 节级 ChunkDraft（基本单位=节，超 300 字按整条装箱拆分）

策略（粒度决策：节为基本 Chunk、条款尽量完整，只做边界合并/切割，禁止任何改写，保 original_text 逐字性；长度双约束 300 基本/500 硬顶）：
1. 节级合并：同一「章 > 节」父路径下的连续条款合并为一个 Chunk，heading_path 记到节级；
2. 超长装箱：节内条款总字数 > SECTION_MAX_CHARS(300) 时，按顺序把整条装入 ≤300 的多片
   （heading_path 仍记节级）；条款尽量不拆，单条 300~500（CLAUSE_SPLIT_MAX_CHARS 硬顶）整条成块，
   超过 500 的异常粘连/长条款回退段落软切（保证 ≤500）；
3. 表格保护：含 Markdown 表格的节整节成一块，长度让位于表格行列完整性（chunk_type='table'）；
   若整表超硬顶，由 quality.check_chunk 判超长进 need_review 隔离带（valid 集不超 500）；
4. 段落兜底：仅对无条款结构的文体（md/ppt 小节、章节导语）、软条款序号项与超 CLAUSE_SPLIT_MAX_CHARS
   的异常单条超长时按段落窗口切分并保留重叠字符；窗口内无标点时硬断兜底，确保不产出超 500 块；
5. content 前缀「文档标题 + 章节路径」——检索侧已确认的跨切片主体割裂治理手段
   （孪生规则/费率切片靠路径前缀携带文档与章节信息）。
"""
from src.data.ingestion.config import (
    CHUNK_TYPE_HINTS,
    CHUNK_OVERLAP_CHARS,
    CLAUSE_SPLIT_MAX_CHARS,
    MIN_CHUNK_CHARS,
    SECTION_MAX_CHARS,
)
import re
from src.data.ingestion.schemas import ChunkDraft, Section
from src.data.ingestion.structuring import split_sections, _RE_DOTTED_NUM

# heading_path 层级分隔符（与 structuring 一致）
_PATH_SEP = " > "
# Markdown 表格行特征
_TABLE_LINE = "| "
_TABLE_DASH = "|---"


def chunk_document(doc, sections=None) -> list[ChunkDraft]:
    """全文切块入口：结构化感知（条级原子单元）→ 节级合并 → 整条装箱 → 表格/段落保护；
    sections 可由调用方传入（避免与章节对账重复解析），缺省自行 split"""
    sections = sections if sections is not None else split_sections(doc)
    drafts: list[ChunkDraft] = []
    for group in _group_by_parent(sections):
        drafts.extend(_chunk_group(doc, group))
    return drafts


def _parent_path(section: Section) -> str:
    """去掉 heading_path 末级条款号的父路径（章 > 节）；无条款号时自身即父路径"""
    if section.clause_position and _PATH_SEP in section.heading_path:
        return section.heading_path.rsplit(_PATH_SEP, 1)[0]
    # 无章级标题可寻的多级点号条款（标准体/细则体）：按编号父段作装箱分组
    # （4.1/4.2 → 4），使短条款能节级装箱成 ≤300 块而非各自成片（正式版标准文书实测碎片严重）
    if section.clause_position and _RE_DOTTED_NUM.fullmatch(section.clause_position):
        return section.clause_position.rsplit(".", 1)[0]
    return section.heading_path


def _group_by_parent(sections: list[Section]) -> list[list[Section]]:
    """连续同父路径的条款归为一节；父路径变化即开新组（节是合并单位边界）"""
    groups: list[list[Section]] = []
    for sec in sections:
        parent = _parent_path(sec)
        if groups and _parent_path(groups[-1][-1]) == parent:
            groups[-1].append(sec)
        else:
            groups.append([sec])
    return groups


def _chunk_group(doc, group: list[Section]) -> list[ChunkDraft]:
    """单节切块：≤300 合为一片；含表格整节一块；超限按整条装箱拆分"""
    parent = _parent_path(group[0])
    merged_text = "\n".join(s.text.strip() for s in group if s.text.strip())

    # 表格完整性优先于长度：含表格的节恒定整块（Phase 2 长表实测依赖此分支）
    if any(_has_table(s.text) for s in group):
        if len(group) == 1:
            return [_make_draft(doc, group[0])]
        return [_make_draft(doc, Section(heading_path=parent, clause_position=None, text=merged_text))]

    # 节级整体成块（单条组保留条款号与三级路径，多条组记节级路径）
    if len(merged_text) <= SECTION_MAX_CHARS:
        if len(group) == 1:
            return [_make_draft(doc, group[0])]
        return [_make_draft(doc, Section(heading_path=parent, clause_position=None, text=merged_text))]

    # 超限拆分：以条款为尽量不拆单元顺序装箱；单成员的非条款文体（md/ppt 小节、导语）与
    # 软条款（cn_mode 序号项如招募说明书「释义」节）走段落兜底；法定「第X条」≤CLAUSE_SPLIT_MAX_CHARS(500)
    # 整条成块（300~500 落兜底带），更长的（误粘连体或少数长条款）一律兜底软切，硬保证单块不超 500
    if len(group) == 1:
        sec = group[0]
        if sec.clause_position and not sec.soft_clause and len(sec.text) <= CLAUSE_SPLIT_MAX_CHARS:
            return [_make_draft(doc, sec)]
        return [
            _make_draft(doc, sec, part)
            for part in _split_by_paragraphs(sec.text, SECTION_MAX_CHARS, CHUNK_OVERLAP_CHARS)
        ]
    return _pack_clauses(doc, group, parent)


def _pack_clauses(doc, group: list[Section], parent: str) -> list[ChunkDraft]:
    """整条装箱：顺序累积条款，装不下即封块开新箱；每片 ≤SECTION_MAX_CHARS（单条超限除外）"""
    drafts: list[ChunkDraft] = []
    buf: list[Section] = []
    size = 0
    for sec in group:
        text = sec.text.strip()
        if not text:
            continue
        if buf and size + len(text) + 1 > SECTION_MAX_CHARS:
            drafts.extend(_emit_bin(doc, buf, parent))
            buf, size = [], 0
        buf.append(sec)
        size += len(text) + 1
    if buf:
        drafts.extend(_emit_bin(doc, buf, parent))
    return drafts


def _emit_bin(doc, bin_: list[Section], parent: str) -> list[ChunkDraft]:
    """单箱出片：单条箱保留条款号与三级路径（异常超长/软条款同样兜底软切）；多条箱合并为节级片"""
    if len(bin_) == 1:
        sec = bin_[0]
        # 与 _chunk_group 单成员分支同判据：仅「法定第X条」且 ≤CLAUSE_SPLIT_MAX_CHARS 才整条成块；
        # clause=None 的巨型非条款块（如招募说明书「第X部分」正文单节）与软条款/异常粘连体一律兜底软切，
        # 否则多成员装箱里混入的单个巨型节会整块直出（正式版产品说明书实测漏出 6380 字块）
        keep_whole = (sec.clause_position and not sec.soft_clause
                      and len(sec.text) <= CLAUSE_SPLIT_MAX_CHARS)
        if not keep_whole:
            return [
                _make_draft(doc, sec, part)
                for part in _split_by_paragraphs(sec.text, SECTION_MAX_CHARS, CHUNK_OVERLAP_CHARS)
            ]
        return [_make_draft(doc, sec)]
    merged = "\n".join(s.text.strip() for s in bin_)
    return [_make_draft(doc, Section(heading_path=parent, clause_position=None, text=merged))]


def _make_draft(doc, section: Section, text: str | None = None) -> ChunkDraft:
    # content = 文档主体 + 章节路径前缀（heading_path 自身不含文档名），original_text 逐字原文；
    # 无结构退化块 heading_path 即文档标题，不重复拼接
    text = section.text if text is None else text
    if not section.heading_path:
        prefix = doc.title
    elif section.heading_path.startswith(doc.title):
        prefix = section.heading_path
    else:
        prefix = f"{doc.title} {section.heading_path}"
    content = f"{prefix}：{text}"
    return ChunkDraft(
        source_path=doc.source_path,
        heading_path=section.heading_path,
        clause_position=section.clause_position,
        # 要素通道直接给型（精度高于关键词粗分）；is_element 同步作为短碎片豁免标记
        chunk_type=section.chunk_type_hint or _infer_chunk_type(text),
        content=content,
        original_text=text,
        is_element=section.chunk_type_hint is not None,
    )


def _has_table(text: str) -> bool:
    return any(ln.startswith(_TABLE_LINE) for ln in text.splitlines()) or _TABLE_DASH in text


def _infer_chunk_type(text: str) -> str:
    """chunk_type 粗分类：表格 → table；其余按关键词提示命中类型，默认 clause"""
    if _has_table(text):
        return "table"
    for ctype, hints in CHUNK_TYPE_HINTS.items():
        if ctype == "table":
            continue
        if any(h in text for h in hints):
            return ctype
    return "clause"


def _split_by_paragraphs(text: str, max_chars: int, overlap: int) -> list[str]:
    """散文窗口切分（仅用于无条款文体）：累积单元至上限即封块，下一块携带上一块尾部 overlap 字符保衔接"""
    parts: list[str] = []
    buf: list[str] = []
    size = 0
    for unit in _iter_units(text, max_chars):
        if buf and size + len(unit) + 1 > max_chars:
            block = "\n".join(buf)
            parts.append(block)
            tail = block[-overlap:] if overlap else ""
            # 尾巴+新单元仍超容量时放弃衔接（长度硬约束优先，避免尾60+头300撑爆块）
            if len(tail) + len(unit) + 1 > max_chars:
                tail = ""
            buf = [tail] if tail else []
            size = len(tail)
        buf.append(unit)
        size += len(unit) + 1
    if buf:
        block = "\n".join(buf)
        if block.strip():
            parts.append(block)
    return [p for p in parts if len(p.strip()) >= MIN_CHUNK_CHARS] or [text]


_RE_SENT_END = re.compile(r"[。；！？]")
# 软标点回退：无句末标点可封时按逗号/顿号等封口（仅散文文体；条款不走此路径）
_RE_SOFT_END = re.compile(r"[，、；：,;:]")


def _iter_units(text: str, max_chars: int):
    """段落→行→句逐级细化切分单元：尽量产出不超 max_chars 的可读块（单句无句末标点仍超长则整段退化）"""
    for para in (p.strip() for p in text.split("\n\n") if p.strip()):
        if len(para) <= max_chars:
            yield para
            continue
        for line in (l.rstrip() for l in para.splitlines() if l.strip()):
            if len(line) <= max_chars:
                yield line
                continue
            start = 0
            while start < len(line):
                # 取 [start, start+max_chars] 范围内最后一个句末标点作封口点；
                # 无句末标点（如整句修订记录）退而按逗号等软标点；仍无则整段剩余直出（不强断半句）
                end = start + max_chars + 1
                cut = max((m.end() for m in _RE_SENT_END.finditer(line, start, end)), default=None)
                if cut is None:
                    soft = [m.end() for m in _RE_SOFT_END.finditer(line, start, end)]
                    # 软切至少保留半块长度，避免产出接连的碎小片
                    cut = max((e for e in soft if e - start >= max_chars // 2), default=None)
                if cut is None:
                    # 窗口内无任何句读/软标点（表格/数字流水行为主）：为守住硬上限不再整段直出，
                    # 退到 max_chars 处硬断兜底（宁可断半句，也不产出超上限长块致尾部编码不可达）
                    cut = min(start + max_chars, len(line))
                yield line[start:cut]
                start = cut
