"""Phase 3 质量门控 + 扫描件通道验证（只读正式数据，fixture 建于临时目录自动清理，不写任何库）

断言四检一控：
1. 纯扫描件 PDF（pypdf 白页）端到端：空文本检出 → 整篇隔离，全片 need_review 零 valid；
2. 半扫描件（mock 10 页 7 空页、均值 90 字/页）：均值检不触发、空页率检命中——证明检②必要性；
3. 乱码文本（控制字符污染字节）端到端：文档级乱码检出 → 整篇隔离（防单片压线漏网混写 valid）；
4. 丢章文本（目录声明八章、正文仅两章）端到端：章节对账检出缺失章 → 整篇隔离；
5. 零误报对照：23 份正式文档（4 监管 txt + 深交所 PDF + 北交所 DOCX + 7 内部制度 + 10 产品说明书）
   文档级四检全部 0 问题（含 txt/PDF 双载体目录对账不误报）。

启动方式（项目根目录）：
    python -X utf8 tests/quality_gate_eval.py
"""
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RAW_DATA_DIR  # noqa: E402
from src.data.ingestion import loaders, quality  # noqa: E402
from src.data.ingestion.chunking import chunk_document  # noqa: E402
from src.data.ingestion.readers import read_document  # noqa: E402
from src.data.ingestion.schemas import RawDocument  # noqa: E402
from src.data.ingestion.structuring import split_sections  # noqa: E402

TMP_ROOT = Path(tempfile.mkdtemp(prefix="l1_quality_gate_"))
DOC_DIR = RAW_DATA_DIR / "正式版文档"


def _quarantine_pipeline(path: Path) -> tuple[list[str], int, int]:
    """端到端跑一遍门控链，返回（文档级 issues，valid 片数，need_review 片数）"""
    doc = read_document(path)
    issues = quality.check_document(doc)
    sections = split_sections(doc)
    issues += quality.check_chapters(doc, sections)
    drafts = quality.annotate(chunk_document(doc, sections), issues)
    rows = loaders.to_db_rows(drafts, doc_version_id=f"TEST:{path.stem}")
    valid = sum(1 for r in rows if r["status"] == "valid")
    review = sum(1 for r in rows if r["status"] == "need_review")
    return issues, valid, review


def eval_scanned_pdf(failures: list[str]) -> None:
    """检①端到端：pypdf 生成 5 页无任何文本的「扫描版」PDF"""
    from pypdf import PdfWriter

    path = TMP_ROOT / "scanned_blank.pdf"
    writer = PdfWriter()
    for _ in range(5):
        writer.add_blank_page(width=612, height=792)
    with path.open("wb") as fh:
        writer.write(fh)

    issues, valid, review = _quarantine_pipeline(path)
    if not issues:
        failures.append("纯扫描 PDF 未被检出（应报空文本/扫描件）")
    if valid:
        failures.append(f"纯扫描 PDF 混写 valid={valid} 片（应零 valid，有片则全进隔离带，实际 need_review={review}）")


def eval_half_scanned(failures: list[str]) -> None:
    """检②函数级：10 页里 7 页空白，但 3 个文字页把均值拉到 90 字/页——检①漏网，检②必须兜住"""
    body = "第一条　测试条款内容。" * 60  # 900 字，均值 90 ≥ 50 不触发检①
    doc = RawDocument(source_path=TMP_ROOT / "half.pdf", title="半扫描", text=body,
                      page_count=10, empty_pages=7)
    issues = quality.check_document(doc)
    avg_hits = [i for i in issues if "平均每页" in i]
    ratio_hits = [i for i in issues if "空页率" in i]
    if avg_hits:
        failures.append("半扫描样例均值 90 字/页不应触发检①，实际触发（阈值失效）")
    if not ratio_hits:
        failures.append(f"半扫描样例（空页率 70%）未被检②捕获：issues={issues}")


def eval_garbled_txt(failures: list[str]) -> None:
    """编码损坏端到端：控制字符污染字节流（各编码解码后仍为乱字符）"""
    path = TMP_ROOT / "garbled.txt"
    path.write_bytes(b"report\x00\x01\x02\x03\xfe\xff" * 300)

    issues, valid, review = _quarantine_pipeline(path)
    if not any("乱码" in i for i in issues):
        failures.append(f"乱码文本未触发文档级乱码检：issues={issues[:2]}")
    if valid:
        failures.append(f"乱码文本仍混写 valid={valid}，应整篇隔离")


def eval_missing_chapters(failures: list[str]) -> None:
    """章节对账端到端：目录声明八章，正文只解析出两章"""
    toc = "\n".join(
        f"第{c}章 测试章节................................{i * 5}"
        for i, c in enumerate("一二三四五六七八", start=1)
    )
    body = "第一章 总则\n1.1　本规则为测试样例，用于验证目录与正文的章节对账逻辑。\n第二章 交易\n2.1.1　正文仅解析出前两章，其余六章整章丢失。"
    path = TMP_ROOT / "missing_chapters.txt"
    path.write_text(f"目 录\n{toc}\n{body}", encoding="utf-8")

    issues, valid, review = _quarantine_pipeline(path)
    missing = [i for i in issues if "整章丢失" in i]
    if len(missing) != 6:
        failures.append(f"章节对账应检出 6 章缺失，实际 {len(missing)}：{missing[:2]}")
    if valid:
        failures.append(f"丢章文本仍混写 valid={valid}，应整篇隔离")


def eval_no_false_positive(failures: list[str]) -> int:
    """零误报对照：23 份正式文档文档级四检必须全空"""
    dirs = [DOC_DIR / "监管规则类", DOC_DIR / "内部制度类", DOC_DIR / "产品说明书类"]
    files = [f for d in dirs if d.is_dir() for f in sorted(d.iterdir())
             if f.is_file() and f.suffix.lower() in (".txt", ".pdf", ".docx")]
    for f in files:
        doc = read_document(f)
        issues = quality.check_document(doc) + quality.check_chapters(doc, split_sections(doc))
        if issues:
            failures.append(f"{f.name} 正常文档被误拦：{issues[0]}")
    return len(files)


def main():
    print("=" * 60)
    print("Phase 3 质量门控 + 扫描件通道验证")
    print("=" * 60)
    failures: list[str] = []
    try:
        eval_scanned_pdf(failures)
        print("  [1] 纯扫描 PDF 端到端 …", "FAIL" if failures else "OK")
        n = len(failures)
        eval_half_scanned(failures)
        print("  [2] 半扫描空页率检 ……", "FAIL" if len(failures) > n else "OK")
        n = len(failures)
        eval_garbled_txt(failures)
        print("  [3] 乱码文本整篇隔离 …", "FAIL" if len(failures) > n else "OK")
        n = len(failures)
        eval_missing_chapters(failures)
        print("  [4] 目录章节对账 ………", "FAIL" if len(failures) > n else "OK")
        n = len(failures)
        total = eval_no_false_positive(failures)
        print(f"  [5] {total} 份正式文档零误报 …", "FAIL" if len(failures) > n else "OK")
    finally:
        shutil.rmtree(TMP_ROOT, ignore_errors=True)

    print("\n" + "=" * 60)
    if failures or total < 21:
        print(f"结果 FAIL（{len(failures)} 项）：")
        for x in failures:
            print(f"  - {x}")
        return 1
    print("结果 PASS：四类劣质文档全部检出并整篇隔离，正式文档零误报")
    return 0


if __name__ == "__main__":
    sys.exit(main())
