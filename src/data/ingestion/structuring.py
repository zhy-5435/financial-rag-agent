"""结构化感知：从抽取文本自动识别章节层级与条款号，产出 Section（heading_path / clause_position）

三条规则通道（监管规则/内部制度对齐评测见 tests/chunk_alignment_eval.py，要素通道见 tests/element_chunk_eval.py）：
- 要素通道（产品说明书类键值体）：按 `**要素名**：` 字段块切分，heading_path 记「文档标题 > 要素名」，
  chunk_type 由要素名直接映射；路由命中条件见 config.PRODUCT_DIR_HINTS / ELEMENT_MIN_MARKS；
- Markdown 通道：# 标题层级直接构成 heading_path；
- 中文制度文书通道：第X章/节标题栈 + 行首条款（1.1　/ 第三条　）+ 一行多条的行内拆分；
  并按全篇编号形态做两种文档级自适应（正式版文档预检暴露的标准体/序号体不适配）：
  「一、/案例一：」纯序号体整篇无条级条款时提升为软条款单元；多级点号条款密集且无章标题的
  标准文书（GB/行标/地标）识别「4 组织职责」「附 录 A」「A.1」标题层级。
均不命中时整篇退化为单 Section（heading_path=文档标题），交由切块链按长度兜底。
"""
import re

from src.data.ingestion.config import (
    ELEMENT_MIN_MARKS,
    ELEMENT_TYPE_RULES,
    PRODUCT_DIR_HINTS,
)
from src.data.ingestion.schemas import RawDocument, Section

# Markdown 标题行：#{1,6} + 空格
_RE_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

# 页码噪声行：PDF/Word 转出的「- 5 -」「— 12 —」（金标原文不含，切块前剥离）
_RE_PAGE_NOISE = re.compile(r"^\s*[-—–]\s*\d+\s*[-—–]\s*$")
# 裸页码行：pypdf 抽取后单独成行的「12」（正文不会出现纯 1~3 位数字行，目录页码换行也是此形态）
_RE_BARE_PAGE = re.compile(r"^\d{1,3}$")
# 章/编/篇级标题行（depth 1）：第一章 总则（标题短、无句读结尾）
_RE_CHAPTER = re.compile(r"^(第[一二三四五六七八九十百千零〇\d]+[章编篇部])\s*([^。；\n]{0,30})$")
# 节级标题行（depth 2）：第二节 交易参与人与交易权限
_RE_SECTION = re.compile(r"^(第[一二三四五六七八九十百千零〇\d]+节)\s*([^。；\n]{0,30})$")
# 行首条款：第X条 / 数字编号（多级点号如 2.4.1 后接任意空白/顿号；纯整数必须后接全角/窄空格
# 且不能是日期字（如「30　日」），以排除修订记录里的「2001 年11月」「30 日《…」等行误判）
_RE_TIAO_HEAD = re.compile(r"^(第[一二三四五六七八九十百千零〇\d]+条)([\u2002\u3000\s、.]|$)")
_RE_NUM_HEAD = re.compile(r"^(\d+(?:\.\d+)+)(?=[\u2002\u3000\s、.])|^(\d+)[\u2002\u3000]+(?![日月年时])")
# 行尾粘连页码：「第一章 总则     - 5 -」（章标题行末尾也带，需单独剪掉）
_RE_TRAIL_PAGE = re.compile(r"\s*[-—–]\s*\d+\s*[-—–]\s*$")
# 行内后续条款（一条挤满多款的内部制度常见）：句末标点后紧跟第X条/多级编号
_RE_INLINE_CLAUSE = re.compile(
    r"(?<=[。；])(?=\s*(?:第[一二三四五六七八九十百千零〇\d]+条[\u2002\u3000\s]|\d+(?:\.\d+)+[\u2002\u3000\s]))"
)
# 目录行：点线 + 可选行尾页码（如「第三节 挂牌、摘牌…........24」；PDF 排版常把页码硬换行
# 甩到下一行，故页码可缺失，行尾≥4 个连续点/省略号即判目录行；正文无此形态）。
# 目录是导航信息不是知识内容，连同「目 录」标题行整行丢弃（与页码剥离同理，不触碰正文逐字性）
_RE_TOC_LINE = re.compile(r"(?:[.…·]{4,}|\s{6,})\s*\d{0,3}\s*$")
_RE_TOC_HEAD = re.compile(r"^目\s*录$")

# 多级点号条款号（1.1 / 4.2.3）：判定标准体信号与切块装箱分组共用
_RE_DOTTED_NUM = re.compile(r"\d+(?:\.\d+)+")
# 行首中文大项（一、/十二、）与案例标题（案例一：）：纯序号体文书（员工手册、招募说明书、
# 案例汇编）的主结构单元，仅在 cn_mode（全篇无任何条级条款行）下启用，不影响法规文书「第X条不可分」
_RE_CN_ITEM = re.compile(r"^([一二三四五六七八九十百]{1,3}、)(?=\S)")
_RE_CASE_HEAD = re.compile(r"^(案例[一二三四五六七八九十]+[：:])(?=\S)")
# GB 标准体单层编号标题行：「4 组织职责」（1~2 位整数+空格+短标题行，标题不含句读、排除日期字开头；
# 半角空格正文句（如「2 小时以上，按旷工处理。」）由句读护栏排除）——仅 gb_mode（无章标题且点号条款密集）启用
_RE_GB_HEADING = re.compile(r"^(\d{1,2})[\u2002\u3000\s]+([^\s\d、.月日年时分秒].{0,23})$")
# 标准附录：附录标题行「附 录 A（资料性）…」与附录条款「A.1 技术信息」（正文无此形态）
_RE_APPENDIX_HEAD = re.compile(r"^(附\s*录\s*[A-Z])([^\n]{0,28})$")
_RE_APPENDIX_CLAUSE = re.compile(r"^([A-Z]\.\d+(?:\.\d+)*)(?=[\u2002\u3000\s、.])")
# 标题行句读护栏：含句读即非标题而是正文句
_RE_TITLE_PUNCT = re.compile(r"[。；，、！？,;]")

# heading_path 各级分隔符，与人工金标切片保持一致
_PATH_SEP = " > "

# 要素标记：行首 `**要素名**：`（粗体键 + 全/半角冒号），产品说明书键值体专用
_RE_ELEMENT_MARK = re.compile(r"^\*\*([^*\n]{2,20})\*\*\s*[:：]", re.M)


def split_sections(doc: RawDocument) -> list[Section]:
    """将全文按结构切为章节列表；通道产出为空时逐级回退，无结构可识别时整篇退化为单章节"""
    lines = doc.text.splitlines()
    sections: list[Section] = []
    if _is_product_element(doc, lines):
        sections = _split_product_elements(doc)
    if not sections:
        if _looks_like_markdown(lines):
            sections = _split_markdown(doc, lines)
        else:
            sections = _split_cn_regulation(doc, lines)
    return sections if sections else [
        Section(heading_path=doc.title, clause_position=None, text=doc.text)
    ]


# 目录行章 token 只取行首（完整 _RE_CHAPTER 要求全行≤30 字，目录点线超长不适配）
_RE_TOC_CHAPTER = re.compile(r"^(第[一二三四五六七八九十百千零〇\d]+章)")


def chapters_in_toc(text: str) -> set[str]:
    """目录行中声明的章 token（第一章…）：供 Phase 3 章节比对，检出整章丢失；无目录返回空集"""
    out: set[str] = set()
    for ln in text.splitlines():
        ln = ln.strip()
        if _RE_TOC_LINE.search(ln) and (m := _RE_TOC_CHAPTER.match(ln)):
            out.add(m.group(1))
    return out


def chapter_token(part: str) -> str | None:
    """heading_path 某级的章 token（如「第三章 证券买卖」→第三章），非章级返回 None"""
    m = _RE_CHAPTER.match(part.strip())
    return m.group(1) if m else None


def _is_product_element(doc: RawDocument, lines: list[str]) -> bool:
    """要素通道路由：须有真实键值标记方可进入——目录提示词命中但正文零标记（产品类 PDF 经
    pypdf 抽取后为纯文本体，无 ** 粗体标记，正式版 80 份实测全中此坑）不再单独成立，
    回退制度文书通道；无提示词时仍按「标记数达下限且无行首条款」判定（制度文书不会误入）"""
    marks = len(_RE_ELEMENT_MARK.findall(doc.text))
    if marks == 0:
        return False
    if any(hint in part for part in doc.source_path.parts for hint in PRODUCT_DIR_HINTS):
        return True
    has_clause = any(_RE_TIAO_HEAD.match(ln.strip()) for ln in lines)
    return marks >= ELEMENT_MIN_MARKS and not has_clause


def _element_chunk_type(element_name: str) -> str:
    """要素名→chunk_type 直接映射（优于关键词粗分），全不中默认 product_element"""
    for kw, ctype in ELEMENT_TYPE_RULES:
        if kw in element_name:
            return ctype
    return "product_element"


def _split_product_elements(doc: RawDocument) -> list[Section]:
    """要素切分：每个 `**要素名**：` 到下一标记前的整块为一个 Section（含其后子列表，逐字保留）"""
    cuts = list(_RE_ELEMENT_MARK.finditer(doc.text))
    sections: list[Section] = []
    if not cuts:  # 防御：无任何标记不进要素通道（路由已拦，双保险防越界）
        return []
    # 首标记前的前言（去掉纯标题行）若仍有实质内容，以「产品概览」单独成块
    head = "\n".join(
        ln for ln in doc.text[: cuts[0].start()].splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ).strip()
    if head:
        sections.append(Section(
            heading_path=f"{doc.title}{_PATH_SEP}产品概览", clause_position=None,
            text=head, chunk_type_hint="product_element",
        ))
    for i, m in enumerate(cuts):
        end = cuts[i + 1].start() if i + 1 < len(cuts) else len(doc.text)
        body = doc.text[m.start(): end].strip()
        if not body:
            continue
        name = m.group(1).strip()
        sections.append(Section(
            heading_path=f"{doc.title}{_PATH_SEP}{name}", clause_position=None,
            text=body, chunk_type_hint=_element_chunk_type(name),
        ))
    return sections


def _looks_like_markdown(lines: list[str]) -> bool:
    return sum(1 for ln in lines[:200] if _RE_MD_HEADING.match(ln)) >= 2


def _split_markdown(doc: RawDocument, lines: list[str]) -> list[Section]:
    """Markdown：按 # 层级维护标题栈，正文归属最近标题"""
    stack: list[tuple[int, str]] = []   # (层级, 标题文本)
    sections: list[Section] = []
    buf: list[str] = []

    def flush(heading: list[tuple[int, str]]):
        body = "\n".join(buf).strip()
        if body and heading:
            path = _PATH_SEP.join([t for _, t in heading])
            sections.append(Section(heading_path=path, clause_position=None, text=body))

    for ln in lines:
        m = _RE_MD_HEADING.match(ln)
        if m:
            flush(stack)
            buf = []
            level, title = len(m.group(1)), m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
        else:
            buf.append(ln)
    flush(stack)
    return sections


def _clause_head_id(line: str) -> tuple[str, str] | None:
    """行首条款识别：返回 (clause_position, 去掉条款号后的行剩余)；非条款起始返回 None"""
    m = _RE_TIAO_HEAD.match(line)
    if m:
        return m.group(1), line[m.end(1):]
    m = _RE_NUM_HEAD.match(line)
    if m:
        num = m.group(1) or m.group(2)
        return num, line[m.end():]
    return None


def _split_cn_regulation(doc: RawDocument, lines: list[str]) -> list[Section]:
    """中文制度文书：章/节构成标题栈，条款行开新切片；一行多条时行内再拆。

    预扫描全篇编号形态后启用两种文档级自适应（互斥于常规法规文书，保证金标行为不变）：
    - cn_mode：全篇无条级条款行且「一、/案例一：」行≥3（纯序号体：员工手册、招募说明书、
      案例汇编）——序号行提升为软条款（soft_clause：短项豁免碎片质检，超长项如「释义」
      节仍允许段落软切，不受「第X条不可分」约束）；
    - gb_mode：多级点号条款≥5 且全篇无第X章/节标题（标准文书：GB/团标/地标）——识别
      「4 组织职责」单层编号章级标题与「附 录 A」「A.1」附录层级（此前整篇粘连成数千字巨型块）。"""
    stripped = [ln.strip() for ln in lines]
    heads = [_clause_head_id(ln) for ln in stripped if ln]
    n_dot = sum(1 for h in heads if h and _RE_DOTTED_NUM.fullmatch(h[0]))
    n_item = sum(1 for ln in stripped if ln and (_RE_CN_ITEM.match(ln) or _RE_CASE_HEAD.match(ln)))
    has_chapter = any(_RE_CHAPTER.match(ln) or _RE_SECTION.match(ln) for ln in stripped)
    cn_mode = not any(heads) and n_item >= 3
    gb_mode = n_dot >= 5 and not has_chapter

    chapters: list[str] = []              # 当前 章/节 标题链（depth 1~2）
    sections: list[Section] = []
    cur_clause: list[str | None] = [None]  # 当前条款号（heading_path 末级 + clause_position）
    cur_soft: list[bool] = [False]         # 当前条款是否软条款（cn_mode 序号项，超长可软切）
    buf: list[str] = []

    def emit(text: str, clause: str | None, soft: bool = False):
        text = text.strip()
        if not text:
            return
        parts = chapters + ([clause] if clause else [])
        sections.append(Section(
            heading_path=_PATH_SEP.join(parts) if parts else doc.title,
            clause_position=clause,
            text=text,
            soft_clause=soft,
        ))

    def flush():
        body = "\n".join(buf).strip()
        buf.clear()
        if not body:
            return
        # 行内多条款拆分（首段归属当前条款号，后续段各归自己的条款号）
        cuts = list(_RE_INLINE_CLAUSE.finditer(body))
        if not cuts:
            emit(body, cur_clause[0], cur_soft[0])
            return
        segs, prev = [], 0
        for c in cuts:
            head = _clause_head_id(body[c.start():].strip())
            if not head:
                continue
            segs.append((body[prev:c.start()].strip(), cur_clause[0], cur_soft[0]))
            cur_clause[0] = head[0]
            cur_soft[0] = False
            prev = c.start()
        segs.append((body[prev:].strip(), cur_clause[0], cur_soft[0]))
        for text, clause, soft in segs:
            emit(text, clause, soft)

    for raw in lines:
        ln = _RE_TRAIL_PAGE.sub("", raw.strip()).rstrip()
        if not ln or _RE_PAGE_NOISE.match(ln) or _RE_BARE_PAGE.match(ln):
            continue
        if _RE_TOC_LINE.search(ln) or _RE_TOC_HEAD.match(ln):
            continue  # 目录导航行（含换行甩落的页码裸行）整行丢弃，不入正文也不参与章节识别
        mc, depth = None, None
        if m := _RE_CHAPTER.match(ln):
            mc, depth = m, 1
        elif m := _RE_SECTION.match(ln):
            mc, depth = m, 2
        if mc:
            flush()
            cur_clause[0] = None
            cur_soft[0] = False
            title = (mc.group(1) + (" " + mc.group(2).strip() if mc.group(2) else "")).strip()
            chapters = (chapters[: depth - 1] + [title])[:depth]
            continue
        if gb_mode and (m := _RE_APPENDIX_HEAD.match(ln)) and not _RE_TITLE_PUNCT.search(m.group(2)):
            flush()  # 附录标题行开新章级（「附 录 A（资料性）…」括号非句读，不入护栏）
            cur_clause[0] = None
            cur_soft[0] = False
            tail = m.group(2).strip()
            chapters = [m.group(1).replace(" ", "") + (f" {tail}" if tail else "")]
            continue
        if head := _clause_head_id(ln):
            flush()
            cur_clause[0] = head[0]
            cur_soft[0] = False
            buf.append(ln)
            continue
        if gb_mode and (m := _RE_GB_HEADING.match(ln)) and not _RE_TITLE_PUNCT.search(m.group(2)):
            flush()  # 「4 组织职责」式单层编号标题：章级栈，其下 x.y 条款归组装箱
            cur_clause[0] = None
            cur_soft[0] = False
            chapters = [f"{m.group(1)} {m.group(2).strip()}"]
            continue
        if gb_mode and (m := _RE_APPENDIX_CLAUSE.match(ln)):
            flush()
            cur_clause[0] = m.group(1)
            cur_soft[0] = False
            buf.append(ln)
            continue
        if cn_mode and ((m := _RE_CN_ITEM.match(ln)) or (m := _RE_CASE_HEAD.match(ln))):
            flush()  # 纯序号体：行首「一、/案例一：」开新软条款
            cur_clause[0] = m.group(1)
            cur_soft[0] = True
            buf.append(ln)
            continue
        buf.append(ln)
    flush()
    return sections
