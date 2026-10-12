"""摄取层 CLI：--check 空跑（只打印解析与切块结果，绝不写库）/ 默认模式正式摄取

启动方式（项目根目录执行，正式模式需 L2/L3 服务未运行以释放 MilvusLite 锁）：
    python -m src.data.ingestion                    # 正式摄取 data/incoming/（增量写库+版本递增+同步）
    python -m src.data.ingestion <投放目录>          # 指定其他投放目录（测试/隔离演练用）
    python -m src.data.ingestion --check [文件或目录]...  # 空跑预览（如 data/raw）
正式摄取的增量检测/版本递增/superseded 链路见 runner.py 模块文档。
"""
import argparse
import json
import sys
from pathlib import Path

from src.data.ingestion import chunking, keywords, loaders, quality, readers, structuring
from src.data.ingestion.config import INCOMING_DIR
from src.data.ingestion.schemas import RawDocument

# --check 模式下单文件默认预览的切片数（其余仅计数，防刷屏；--preview 可调，0=只看统计）
PREVIEW_CHUNKS = 3
# 切片预览文本最大长度
PREVIEW_CHARS = 80


def _collect_targets(paths: list[str]) -> list[Path]:
    """命令行路径参数 → 待处理文件列表（目录递归；不支持的后缀在 check_file 内提示）"""
    files: list[Path] = []
    for p in [Path(x) for x in paths] or ([INCOMING_DIR] if INCOMING_DIR.exists() else []):
        if p.is_dir():
            files.extend(sorted(f for f in p.rglob("*") if f.is_file()))
        elif p.is_file():
            files.append(p)
        else:
            print(f"[跳过] 路径不存在：{p}")
    return files


def _check_file(path: Path, preview_n: int, kw_mode: str = "rule") -> str:
    """单文件空跑：解析 → 文档级检查 → 切块 → 切片级质检 → 关键词抽取 → 打印将写入的行结构摘要"""
    if not readers.is_supported(path):
        return "skipped"
    try:
        doc: RawDocument = readers.read_document(path)
    except ImportError as e:
        print(f"  [跳过] {e}")
        return "skipped"
    except Exception as e:  # 解析失败即劣质文档，空跑仅报告不中断
        print(f"  [失败] 解析异常：{e!r}")
        return "failed"

    doc_issues = quality.check_document(doc)
    sections = structuring.split_sections(doc)
    doc_issues += quality.check_chapters(doc, sections)
    drafts = quality.annotate(chunking.chunk_document(doc, sections), doc_issues)
    # 空跑缺省走规则路（确定性、零联网）；--keywords llm 显式选择时才真实调用 LLM 预览效果；
    # 整篇隔离文档不参与（不写 valid，抽了也不进索引）——与 runner 同口径用 doc_issues 判
    kw_note = None if doc_issues else keywords.assign_keywords(drafts, kw_mode)
    rows = loaders.to_db_rows(drafts, doc_version_id=f"PREVIEW:{path.stem}")

    print(f"  抽取文本：{len(doc.text)} 字符 | 页数：{doc.page_count or '-'} | 章节切块：{len(drafts)} 片")
    if doc_issues:
        for issue in doc_issues:
            print(f"  [文档级] {issue}")
        # 扫描件/乱码/丢章通道：整篇强制隔离，不做自动 OCR（护栏不允许改写原文），转人工/OCR 后重新投放
        print(f"  ⚠ 整篇隔离：{len(rows)} 片全部 → need_review（不写 valid，检索不召回）")
        return "quarantined"
    n_review = sum(1 for r in rows if r["status"] == "need_review")
    types: dict[str, int] = {}
    for r in rows:
        types[r["chunk_type"]] = types.get(r["chunk_type"], 0) + 1
    print(f"  chunk_type 分布：{types} | 质检后进 need_review：{n_review} 片")
    if kw_note:
        print(f"  keywords 自动抽取：{kw_note}")

    for r in rows[:preview_n]:
        preview = r["content"].replace("\n", " ")[:PREVIEW_CHARS]
        kw = "，".join(json.loads(r["keywords"])) if r["keywords"] else "-"
        print(f"    - {r['knowledge_id']} [{r['chunk_type']}] 条款={r['clause_position'] or '-'} "
              f"| kw={kw[:60]} | {r['heading_path'][:40]} | {preview}…")
    if len(rows) > preview_n:
        print(f"    - …（其余 {len(rows) - preview_n} 片略，--check 不写库）")
    if n_review:
        sample = next(d.issues for d in drafts if d.issues)
        print(f"  [切片级] 质检样例：{sample[0]}（共 {n_review} 片待 need_review）")
    return "ok"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.data.ingestion", description="L1 自动摄取流水线")
    parser.add_argument("--check", action="store_true", help="空跑：仅打印解析与切块结果，不写任何库")
    parser.add_argument("--full-sync", action="store_true",
                        help="正式摄取时走全量重建（FTS 全重建 + 向量全库重嵌），用于存量漂移修复；默认增量 delta")
    parser.add_argument("--keywords", choices=["llm", "rule", "off"], default=None,
                        help="keywords 自动抽取路：正式摄取缺省读 INGEST_KEYWORDS_MODE（默认 llm，失败退 rule）；"
                             "--check 缺省 rule 保空跑零联网，显式 llm 才真实调用")
    parser.add_argument("--preview", type=int, default=PREVIEW_CHUNKS,
                        help=f"--check 每文件预览的切片数（默认 {PREVIEW_CHUNKS}，0=只看统计）")
    parser.add_argument("paths", nargs="*", help="待处理文件或目录（缺省扫描 data/incoming/）")
    args = parser.parse_args(argv)

    if not args.check:
        # 正式摄取（Phase 4）：增量检测 → 版本化写库 → FTS/向量全量同步（仅动 data/incoming/ 入口）
        from src.data.ingestion.runner import run_ingest
        directory = Path(args.paths[0]) if args.paths else INCOMING_DIR
        INCOMING_DIR.mkdir(parents=True, exist_ok=True)
        print("=" * 60)
        print(f"L1 正式摄取：{directory}（增量写 SQLite + "
              f"{'全量重建 FTS/Milvus' if args.full_sync else 'FTS/向量增量同步'}）")
        print("=" * 60)
        stats = run_ingest(directory, sync_mode="full" if args.full_sync else "delta",
                           keywords_mode=args.keywords)
        failed = sum(1 for r in stats["records"] if r["outcome"] == "failed")
        return 1 if failed else 0

    INCOMING_DIR.mkdir(parents=True, exist_ok=True)
    files = _collect_targets(args.paths)
    print("=" * 60)
    print("L1 摄取层 --check 空跑（不写 SQLite / Milvus / 快照）")
    print("=" * 60)
    if not files:
        print(f"投放目录为空：{INCOMING_DIR}（将文档放入后重试，或传路径参数指定样例）")
        return 0

    stats = {"ok": 0, "skipped": 0, "failed": 0, "quarantined": 0}
    for f in files:
        print(f"\n[{f}]")
        stats[_check_file(f, args.preview, kw_mode=args.keywords or "rule")] += 1

    print("\n" + "=" * 60)
    print(f"空跑完成：处理 {stats['ok']} 个，整篇隔离 {stats['quarantined']} 个（转人工/OCR），"
          f"跳过格式 {stats['skipped']} 个，解析失败 {stats['failed']} 个")
    return 0


if __name__ == "__main__":
    sys.exit(main())
