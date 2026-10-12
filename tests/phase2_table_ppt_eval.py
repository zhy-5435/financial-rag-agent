"""Phase 2 验证：表格整块保护 + PPT 解析通路（真实业务数据，临时产物脚本内自动清理）

覆盖两条链路：
1. 表格：以 10 只真实产品说明书费率原文构造 Markdown 对比表（短表 + 超 800 字长表），
   断言整表出现在同一个 chunk、行数无损、chunk_type 识别为 table；
2. PPT：用 python-pptx 生成 3 页真实产品介绍（封面/费率表页/风险提示页），走
   _read_pptx → 切块链，断言页数、正文逐字覆盖、表格页整块、风险提示页类型识别。

启动方式（项目根目录）：
    python -X utf8 tests/phase2_table_ppt_eval.py
"""
import re
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.ingestion import quality  # noqa: E402
from src.data.ingestion.chunking import chunk_document  # noqa: E402
from src.data.ingestion.config import SECTION_MAX_CHARS  # noqa: E402
from src.data.ingestion.readers import read_document  # noqa: E402

# 真实费率数据（逐字摘自 l1_core.db 各 PROD 文档 fee/product_element 切片原文，"—"= 原文未提及）
FEES_SHORT = [
    ("管理费", "0.15%"),
    ("托管费", "0.05%"),
    ("销售服务费", "0.01%"),
    ("申购 / 赎回费", "0"),
]  # 易方达易理财货币市场基金 B
FEES_LONG_HEADER = ("产品", "管理费", "托管费", "销售服务费", "申购 / 赎回费")
FEES_LONG_ROWS = [
    ("信银理财日盈象天天利 199 号", "0.30%", "0.03%", "0.30%", "0"),
    ("大成现金增利货币市场基金 A", "0.33%", "0.05%", "0.25%", "0"),
    ("易方达易理财货币市场基金 B", "0.15%", "0.05%", "0.01%", "0"),
    ("广发景荣纯债 A", "0.60%", "0.20%", "C 类计提 0.40%/年", "赎回费随持有期限递减，持有满 2 年免"),
    ("易方达纯债 A", "0.30%", "0.20%", "C 类免申购费计提销售服务费", "A 类认购 / 申购费前端收取"),
    ("平安惠铭纯债", "0.30%/年", "0.10%/年", "—", "持有满 7 天免赎回费"),
    ("中银理财-稳富（90 天持有期）19", "按日计提", "—", "—", "持有满 90 天免赎回费"),
    ("中邮理财邮银财富·邮鸿宝 30 号", "按产品说明书约定", "按产品说明书约定", "—", "0；无浮动管理费"),
    ("工银理财·天天鑫添益 1 号", "按日计提", "按日计提", "—", "申购费 0；持有满一定期限赎回费为 0"),
    ("招银理财招赢日日金 60 号", "—", "—", "B/C 类免申购费，计提销售服务费", "不同份额费率差异化设计"),
]
FEES_LONG_NOTE = "上表费率摘自各产品说明书与基金合同公示原文，按年化口径列示，未提及项以—标注。"
# 真实费率原文（逐字来自 l1_core.db 各 PROD 文档 fee 切片 original_text），用于把长表所在节撑过 SECTION_MAX_CHARS
FEE_PROSE = [
    "信银理财日盈象天天利 199 号：费率结构（年化）： 固定管理费：0.30% 销售服务费：0.30% 托管费：0.03% 申购 / 赎回费：0",
    "中邮理财邮银财富·邮鸿宝 30 号：费率结构： 申购 / 赎回费：0 固定管理费、托管费按产品说明书约定计提 无浮动管理费",
    "中银理财-稳富（90 天持有期）19：费率结构：持有满 90 天免赎回费，管理费按日计提",
    "招银理财招赢日日金 60 号：费率结构：不同份额费率差异化设计，B/C 类免申购费，计提销售服务费",
    "工银理财·天天鑫添益 1 号：费率结构： 固定管理费、托管费按日计提 申购费：0；持有满一定期限赎回费为 0",
    "大成现金增利货币市场基金 A：费率结构（年化）： 管理费：0.33% 托管费：0.05% 销售服务费：0.25% 申购 / 赎回费：0",
    "易方达易理财货币市场基金 B：费率结构（年化）： 管理费：0.15% 托管费：0.05% 销售服务费：0.01% 申购 / 赎回费：0",
    "广发景荣纯债：费率结构（年化）： A 类管理费：0.60%；托管费：0.20% C 类计提销售服务费 0.40%/ 年 赎回费随持有期限递减，持有满 2 年免赎回费",
    "易方达纯债 A：费率结构（年化）： 管理费：0.30% 托管费：0.20% A 类认购 / 申购费前端收取，C 类免申购费计提销售服务费",
    "平安惠铭纯债：费率结构： 管理费：0.30%/ 年 托管费：0.10%/ 年 持有满 7 天免赎回费",
]

RISK_TEXT = "投资有风险，不保证本金也不保证收益。极端市场行情下产品可能存在估值波动风险与巨额赎回引发的流动性风险。"

# 全部 fixture 产物统一挂在临时根目录下，跑完整体删除
TMP_ROOT = Path(tempfile.mkdtemp(prefix="_tmp_p2_"))


def _table_md(header: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|---" * len(header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _make_md_fixture(table_md: str, note: str, prose: list[str] | None = None) -> Path:
    """产品说明书风格 md：费率结构节 = 说明段 + 整表（+ 可选原文明细，把节长撑过 SECTION_MAX）"""
    body = "\n".join(f"- {p}" for p in prose) if prose else ""
    md = (
        "# 货币与债券类产品费用概览\n\n"
        "## 费率结构\n\n"
        f"{note}\n\n"
        f"{table_md}\n\n"
        f"{body}\n"
    )
    path = TMP_ROOT / "fee_example.md"
    path.write_text(md, encoding="utf-8")
    return path


def eval_table(header: tuple[str, ...], rows: list[tuple[str, ...]], label: str) -> list[str]:
    """表格链路断言：整表同块、行数无损、table 分类；长表额外断言节长确超 SECTION_MAX"""
    table_md = _table_md(header, rows)
    note = FEES_LONG_NOTE if label == "长表" else f"基金管理费与托管费按下列标准年化计提：{rows[0][0]}。"
    prose = FEE_PROSE if label == "长表" else None
    path = _make_md_fixture(table_md, note, prose)
    doc = read_document(path)
    drafts = quality.annotate(chunk_document(doc))

    failures: list[str] = []
    if label == "长表":
        holder = next((d for d in drafts if table_md in d.original_text), None)
        if holder is None or len(holder.original_text) <= SECTION_MAX_CHARS:
            failures.append(
                f"{label}：fixture 节长未超 SECTION_MAX_CHARS={SECTION_MAX_CHARS}，未真正测到长表保护路径"
            )
    if not any(table_md in d.original_text for d in drafts):
        owners = {ln: sum(1 for d in drafts if ln in d.original_text) for ln in table_md.splitlines()}
        missing = [ln for ln, n in owners.items() if n == 0]
        frag = [ln for ln, n in owners.items() if n > 1]
        failures.append(f"{label}：整表未完整出现在同一片（缺失行 {len(missing)}，跨片重复行 {len(frag)}）")
    holder = next((d for d in drafts if table_md in d.original_text), drafts[0] if drafts else None)
    if holder is None:
        failures.append(f"{label}：无任何 chunk 产出")
        return failures
    n_lines = sum(1 for ln in holder.original_text.splitlines() if ln.startswith("|"))
    total_rows = len(rows) + 2  # 表头 + 分隔行 + 数据行
    if n_lines < total_rows:
        failures.append(f"{label}：table chunk 行数 {n_lines} < 原表 {total_rows}，疑被切碎")
    if not any(d.chunk_type == "table" for d in drafts):
        failures.append(f"{label}：无 chunk_type='table' 分类结果")
    print(f"  [{label}] 表 {len(rows)} 行 / {len(table_md)} 字 → chunk {len(drafts)} 片，"
          f"整表同块={'是' if any(table_md in d.original_text for d in drafts) else '否'}，"
          f"类型={sorted({d.chunk_type for d in drafts})}")
    return failures


def _make_pptx_fixture() -> Path:
    """python-pptx 生成 3 页产品介绍：封面 / 费率表页 / 风险提示页"""
    from pptx import Presentation
    from pptx.util import Inches, Pt

    prs = Presentation()
    blank = prs.slide_layouts[6]

    slide = prs.slides.add_slide(blank)
    tb = slide.shapes.add_textbox(Inches(1), Inches(2), Inches(8), Inches(2)).text_frame
    tb.text = "易方达易理财货币市场基金 B"
    p = tb.add_paragraph()
    p.text = "基金代码：008733  基金类型：货币市场基金  风险等级：低风险（R1）"
    p.font.size = Pt(18)

    slide = prs.slides.add_slide(blank)
    tb = slide.shapes.add_textbox(Inches(1), Inches(0.5), Inches(8), Inches(0.8)).text_frame
    tb.text = "费率结构（年化）"
    table = slide.shapes.add_table(len(FEES_SHORT) + 1, 2, Inches(1), Inches(1.5), Inches(8), Inches(3)).table
    table.cell(0, 0).text = "费用项目"
    table.cell(0, 1).text = "费率"
    for i, (name, rate) in enumerate(FEES_SHORT, start=1):
        table.cell(i, 0).text = name
        table.cell(i, 1).text = rate

    slide = prs.slides.add_slide(blank)
    tb = slide.shapes.add_textbox(Inches(1), Inches(2), Inches(8), Inches(3)).text_frame
    tb.text = "风险提示"
    p = tb.add_paragraph()
    p.text = RISK_TEXT
    p.font.size = Pt(14)

    path = TMP_ROOT / "fund_intro.pptx"
    prs.save(path)
    return path


def eval_ppt() -> list[str]:
    """PPT 链路断言：页数、正文逐字覆盖、表格页整块、风险提示页类型"""
    failures: list[str] = []
    path = _make_pptx_fixture()
    doc = read_document(path)
    if doc.page_count != 3:
        failures.append(f"PPT：page_count={doc.page_count}，期望 3")
    drafts = quality.annotate(chunk_document(doc))
    joined = "\n".join(d.original_text for d in drafts)
    stripped = re.sub(r"[\s\u3000\u2002]", "", joined)
    for must in ("008733", "货币市场基金", "管理费", "0.15%", RISK_TEXT[:30]):
        if re.sub(r"[\s\u3000\u2002]", "", must) not in stripped:
            failures.append(f"PPT：关键文本未进入任何 chunk：{must!r}")
    table_md = _table_md(("费用项目", "费率"), FEES_SHORT)
    if table_md not in joined:
        failures.append("PPT：表格页整表未出现在同一 chunk")
    if not any(d.chunk_type == "risk" for d in drafts):
        failures.append("PPT：风险提示页未被分类为 chunk_type='risk'")
    print(f"  [PPT] page_count={doc.page_count}，chunks={len(drafts)}，"
          f"路径={[d.heading_path for d in drafts]}，类型={sorted({d.chunk_type for d in drafts})}")
    return failures


def main():
    print("=" * 60)
    print("Phase 2 验证：表格整块保护 + PPT 解析通路")
    print("=" * 60)
    failures: list[str] = []

    failures += eval_table(("费用项目", "费率（年化）"), FEES_SHORT, "短表")
    failures += eval_table(FEES_LONG_HEADER, FEES_LONG_ROWS, "长表")
    failures += eval_ppt()

    shutil.rmtree(TMP_ROOT, ignore_errors=True)

    print("\n" + "=" * 60)
    if failures:
        print("结果 FAIL：")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("结果 PASS：表格整块无损（含超节长长表）、PPT 三页通路可用")
    return 0


if __name__ == "__main__":
    sys.exit(main())
