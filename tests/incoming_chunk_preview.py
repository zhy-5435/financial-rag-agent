"""正式版文档切块预检（只读，不写任何库）：逐文档报告路由通道、Section/Chunk 统计与质量信号

用于判断当前切片策略对 data/incoming/正式版文档 三类文档（尤其内部制度类）的适配性：
- 通道：要素 / Markdown / 制度文书 / 退化单节（无结构兜底）
- 质量信号：条款识别数、最短片、>300 片占比、退化整篇直出的文档、疑似漏识别的行首编号形态

启动方式（项目根目录）：
    python -X utf8 tests/incoming_chunk_preview.py [目录名过滤词] [输出文件]
"""
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.ingestion import quality, structuring  # noqa: E402
from src.data.ingestion.chunking import chunk_document  # noqa: E402
from src.data.ingestion.config import CLAUSE_SPLIT_MAX_CHARS, SECTION_MAX_CHARS  # noqa: E402
from src.data.ingestion.readers import is_supported, read_document  # noqa: E402

INCOMING = Path("data/incoming/正式版文档")

# 行首中文序号（一、/（一）/1、）等制度文书常见但当前通道不识别的编号形态，统计其出现量
_RE_CN_SEQ = re.compile(r"^[（(]?[一二三四五六七八九十]{1,3}[）)、．.]\s*\S")
_RE_DIGIT_SEQ = re.compile(r"^\d+[、)]\s*\S")


def channel_of(doc, lines):
    if structuring._is_product_element(doc, lines):
        return "要素"
    if structuring._looks_like_markdown(lines):
        return "Markdown"
    secs = structuring._split_cn_regulation(doc, lines)
    return "制度文书" if len(secs) > 1 else "退化单节"


def preview(path: Path):
    doc = read_document(path)
    lines = doc.text.splitlines()
    chan = channel_of(doc, lines)
    sections = structuring.split_sections(doc)
    drafts = chunk_document(doc, sections)
    issues = quality.check_document(doc)
    lengths = [len(d.original_text) for d in drafts]
    n_clause = sum(1 for s in sections if s.clause_position)
    cn_seq = sum(1 for ln in lines if _RE_CN_SEQ.match(ln.strip()))
    dg_seq = sum(1 for ln in lines if _RE_DIGIT_SEQ.match(ln.strip()))
    stats = Counter(d.chunk_type for d in drafts)

    print(f"\n### {path.parent.name} / {path.name}")
    n_over300 = sum(1 for x in lengths if x > SECTION_MAX_CHARS)
    n_over_hard = sum(1 for x in lengths if x > CLAUSE_SPLIT_MAX_CHARS)
    print(f"  字符数={len(doc.text)}  通道={chan}  Section={len(sections)}(条款{n_clause})  "
          f"Chunk={len(drafts)}  片长 min={min(lengths) if lengths else 0}/"
          f"med={sorted(lengths)[len(lengths)//2] if lengths else 0}/max={max(lengths) if lengths else 0}"
          f"  超{SECTION_MAX_CHARS}片={n_over300}"
          f"  超{CLAUSE_SPLIT_MAX_CHARS}硬顶片={n_over_hard}"
          f"  (<40短片={sum(1 for x in lengths if x < 40)})")
    print(f"  chunk_type: {dict(stats)}")
    if issues:
        print(f"  ⚠ 文档级问题: {issues}")
    if chan == "退化单节":
        print(f"  ⚠ 无结构退化：整篇按段落软切，行首中文序号行={cn_seq} 数字顿号行={dg_seq}")
    elif cn_seq >= 15:
        print(f"  ⚠ 大量行首中文序号({cn_seq}行)，可能有序号标题未入 heading_path")
    # 抽样首尾两片
    for d in drafts[:1] + drafts[-1:]:
        t = d.original_text.replace("\n", "⏎")[:80]
        print(f"  样例[{d.chunk_type}] path={d.heading_path[:60]} | {t}")


class _Tee:
    """报告同时写入文件（Windows 控制台 GBK 解码乱码，以 UTF-8 报告文件为准）"""

    def __init__(self, stream, buf: list[str]):
        self._s, self._buf = stream, buf

    def write(self, s: str):
        self._buf.append(s)
        try:
            self._s.write(s)
        except UnicodeEncodeError:
            pass

    def flush(self):
        self._s.flush()


def main():
    filt = sys.argv[1] if len(sys.argv) > 1 else ""
    report = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("tests/_chunk_preview_report.txt")
    buf: list[str] = []
    sys.stdout = _Tee(sys.__stdout__, buf)
    files = sorted(p for d in INCOMING.iterdir() if d.is_dir() for p in d.iterdir()
                   if is_supported(p) and filt in str(p))
    print(f"共 {len(files)} 个文件")
    for p in files:
        try:
            preview(p)
        except Exception as e:
            print(f"\n### {p.name}\n  ✗ 解析失败: {e!r}")
    report.write_text("".join(buf), encoding="utf-8")
    sys.stdout = sys.__stdout__
    print(f"报告已写入 {report}")


if __name__ == "__main__":
    main()
