"""要素切分通道验证：产品说明书类 10 份走 `**要素名**：` 字段块切分（只读，不写任何库）

断言五项：
1. 路由命中：每片 heading_path 均为「文档标题 > 要素名」两级（要素通道，非无结构退化）；
2. 要素完整：≤300 字的要素块整块落在同一个 chunk 内（切点与 `**要素名**：` 标记对齐，
   不出现"- "列表残段开头的片）；
3. 类型映射：费率要素→fee、风险揭示→risk、投资范围/业绩基准/风险等级→product_element；
4. 短要素豁免：一行式短要素（如业绩比较基准）不进 need_review 碎片带；
5. 长度约束：每片 ≤ SECTION_MAX_CHARS+1（装箱/段落兜底均守 300 字编码窗口）。

启动方式（项目根目录）：
    python -X utf8 tests/element_chunk_eval.py
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RAW_DATA_DIR  # noqa: E402
from src.data.ingestion import quality  # noqa: E402
from src.data.ingestion.chunking import chunk_document  # noqa: E402
from src.data.ingestion.config import SECTION_MAX_CHARS  # noqa: E402
from src.data.ingestion.readers import read_document  # noqa: E402
from src.data.ingestion.structuring import _RE_ELEMENT_MARK, split_sections  # noqa: E402

PRODUCT_DIR = RAW_DATA_DIR / "正式版文档" / "产品说明书类"

_TYPE_PROBE = re.compile(r"\*\*(费率|风险(?!等级)|投资范围|业绩比较基准|风险等级)[^*]*\*\*[:：]")


def eval_doc(path: Path) -> tuple[list[str], str]:
    """单文档断言，返回（失败列表，摘要行）"""
    failures: list[str] = []
    doc = read_document(path)
    sections = split_sections(doc)
    drafts = quality.annotate(chunk_document(doc))

    # 1. 路由命中：全部片带两级路径
    flat = [d for d in drafts if " > " not in d.heading_path]
    if flat:
        failures.append(f"{path.stem}：{len(flat)} 片 heading_path 未达「文档 > 要素」层级，疑未走要素通道")

    # 2. 要素完整：每个 ≤300 的要素 Section 整块落入某个 chunk
    for sec in sections:
        if sec.chunk_type_hint is None or len(sec.text) > SECTION_MAX_CHARS:
            continue
        if not any(sec.text in d.original_text for d in drafts):
            failures.append(f"{path.stem}：要素「{sec.heading_path.split(' > ')[-1]}」被切断，未整块保留")
    frag = [d for d in drafts if d.original_text.lstrip().startswith(("- ", "，", "；"))]
    if frag:
        failures.append(f"{path.stem}：存在以列表残段/标点开头的片（{len(frag)} 片），切点未对齐要素标记")

    # 3. 类型映射抽查：按正文首个要素标记判定期望型，与 chunk_type 比对
    for d in drafts:
        m = _TYPE_PROBE.search(d.original_text[:60])
        if m:
            expect = "fee" if m.group(1) == "费率" else ("risk" if m.group(1) == "风险" else "product_element")
            if d.chunk_type != expect:
                failures.append(f"{path.stem}：「{m.group(1)}」片 chunk_type={d.chunk_type}，期望 {expect}")

    # 4. 短要素豁免：要素通道 need_review 应为 0（乱码/过短都不该出现）
    review = [d for d in drafts if d.issues]
    if review:
        failures.append(f"{path.stem}：{len(review)} 片进 need_review（样例：{review[0].issues[0]}）")

    # 5. 长度约束
    over = [d for d in drafts if len(d.original_text) > SECTION_MAX_CHARS + 1]
    if over:
        failures.append(f"{path.stem}：{len(over)} 片超 {SECTION_MAX_CHARS}+1 字编码窗口")

    types = sorted({d.chunk_type for d in drafts})
    marks = len(_RE_ELEMENT_MARK.findall(doc.text))
    return failures, f"  {path.stem[:22]:<24} 要素 {marks:2d} 个 → chunk {len(drafts):2d} 片，类型={types}"


def main():
    print("=" * 60)
    print("要素切分通道验证（产品说明书类）")
    print("=" * 60)
    files = sorted(f for f in PRODUCT_DIR.iterdir() if f.is_file())
    if not files:
        print(f"[FAIL] 目录不存在或为空：{PRODUCT_DIR}")
        return 1

    all_failures: list[str] = []
    for f in files:
        failures, summary = eval_doc(f)
        print(summary)
        all_failures += failures

    print("\n" + "=" * 60)
    if all_failures:
        print(f"结果 FAIL（{len(all_failures)} 项）：")
        for x in all_failures:
            print(f"  - {x}")
        return 1
    print(f"结果 PASS：{len(files)} 份产品说明书均按要素整块切分、类型映射与长度约束达标")
    return 0


if __name__ == "__main__":
    sys.exit(main())
