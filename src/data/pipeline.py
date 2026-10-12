"""L1 知识库全量重建入口：清空双库与摄取状态 → 从投放目录整库重新自动摄取 → 对账与连通性抽测

数据来源已统一为投放目录的原始文档（不再有 Excel 人工切片通道），切片全部由摄取层自动产出。
本命令用于「从零重建整库」：换 embedding 模型、表结构演进、投放目录搬家、或想把存量清干净重做时用；
日常增量入库请直接用 `python -m src.data.ingestion`（不必整库重建）。

启动方式（项目根目录执行，需 L2/L3 服务均未运行以释放 MilvusLite 锁）：
    python -m src.data.pipeline                       # 重建 data/incoming/
    python -m src.data.pipeline "data/raw/正式版文档"  # 重建指定目录（位置参数）
    python -m src.data.pipeline --keywords rule       # 重建时 keywords 走规则路（零联网，省 LLM 调用）
    python -m src.data.pipeline --probe 8             # 检索连通性抽测条数（0=跳过）

重建范围：doc_version / knowledge_chunk / ingest_state 三表 + FTS5 索引 + Milvus 集合。
会话记忆库 l1_memory.db 不在范围内（审计存档永不删除约定）。
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.config  # noqa: E402  统一入口先加载全局路径配置（含 HF 离线标志）
from src.data.hybrid_retriever import hybrid_retriever  # noqa: E402
from src.data.ingestion.config import INCOMING_DIR  # noqa: E402
from src.data.ingestion.runner import run_ingest  # noqa: E402
from src.data.milvus_client import milvus_store  # noqa: E402
from src.data.sqlite_client import sqlite_client  # noqa: E402


def step1_reset() -> tuple[int, int]:
    """清空主库三表与关键词索引，并把向量集合清干净（主库空 → 全量同步即清空）"""
    n_doc, n_chunk = sqlite_client.reset_knowledge_base()
    milvus_store.sync_from_sqlite()
    print(f"已清空存量：doc_version {n_doc} 条，knowledge_chunk {n_chunk} 条，"
          f"ingest_state 与 FTS5 索引全部重置，向量集合已清空")
    return n_doc, n_chunk


def live_chunk_count() -> int:
    """文档与切片双有效（即真正参与检索）的切片行数"""
    conn = sqlite_client._get_conn()
    try:
        return conn.execute(
            "SELECT COUNT(*) FROM knowledge_chunk k JOIN doc_version d "
            "ON k.doc_version_id = d.doc_version_id "
            "WHERE k.status='valid' AND d.status='valid'").fetchone()[0]
    finally:
        conn.close()


def step4_stats_and_reconcile() -> bool:
    """按 doc_type 打印存量，并做双库行数对账（不一致即增量同步有漏，直接判失败）"""
    conn = sqlite_client._get_conn()
    rows = conn.execute(
        "SELECT d.doc_type, COUNT(DISTINCT d.doc_version_id), COUNT(k.knowledge_id) "
        "FROM doc_version d LEFT JOIN knowledge_chunk k "
        "ON k.doc_version_id = d.doc_version_id AND k.status = 'valid' "
        "WHERE d.status = 'valid' GROUP BY d.doc_type ORDER BY 3 DESC").fetchall()
    conn.close()

    print(f"\n{'doc_type':<12}{'文档':>6}{'valid 切片':>12}")
    for doc_type, n_doc, n_chunk in rows:
        print(f"{doc_type or '未分类':<12}{n_doc:>6}{n_chunk:>12}")
    n_live = live_chunk_count()
    n_vec = milvus_store.count_rows()
    ok = n_live == n_vec
    print(f"\n对账：主库双有效切片 {n_live} 条 vs 向量集合 {n_vec} 条 → "
          f"{'一致 ✓' if ok else '不一致 ✗（可用 --full-sync 修复或重跑本命令）'}")
    return ok


def step5_probe(probe_n: int) -> None:
    """检索连通性抽测：从库内切片自带的 keywords/heading 造查询，看能否召回原切片。

    只验证「数据进得去也出得来」这条链是否通，不作质量评测（质量指标见 tests/retrieval_eval.py）。
    查询取自库内自身，因此不依赖任何写死的业务主题，换一批文档照样可跑。
    """
    if probe_n <= 0:
        print("\n[Step 5] 已跳过检索抽测（--probe 0）")
        return
    conn = sqlite_client._get_conn()
    pool = conn.execute(
        "SELECT k.knowledge_id, k.keywords, k.heading_path, d.doc_name FROM knowledge_chunk k "
        "JOIN doc_version d ON k.doc_version_id = d.doc_version_id "
        "WHERE k.status='valid' AND d.status='valid'").fetchall()
    conn.close()
    if not pool:
        print("\n[Step 5] 库内无 valid 切片，跳过检索抽测")
        return

    print(f"\n[Step 5] 检索连通性抽测（随机 {min(probe_n, len(pool))} 条，库内自采样查询）")
    rng = random.Random(20261006)  # 固定种子：重建报告可复现
    hit = 0
    for kid, kw_raw, heading, doc_name in rng.sample(pool, min(probe_n, len(pool))):
        words = [w for w in (json.loads(kw_raw) if kw_raw else []) if isinstance(w, str) and len(w) >= 2]
        query = words[0] if words else (heading or "").split(">")[-1].strip() or doc_name
        results = hybrid_retriever.search(query, top_k=10)
        rank = next((i for i, h in enumerate(results, 1) if h["knowledge_id"] == kid), 0)
        hit += 1 if rank else 0
        top1 = results[0]["doc_name"] if results else "无结果"
        print(f"  {('命中 rank=%d' % rank) if rank else '未命中':<12}| 查询「{query[:18]}」"
              f" → 期望 {kid}｜top1 文档={top1[:24]}")
    print(f"  抽测命中 {hit}/{min(probe_n, len(pool))}（低于全中不代表检索退化："
          f"库内孪生主题会让同内容切片互相顶位，指标以评测集为准）")


def rebuild(directory: Path, keywords_mode: str | None, probe_n: int) -> int:
    t0 = time.time()
    print("=" * 62)
    print("L1 知识库全量重建（自动摄取唯一通道）")
    print(f"投放目录：{directory}")
    print(f"目标库：{src.config.SQLITE_DB_PATH.name} + {src.config.MILVUS_DB_PATH.name}")
    print("=" * 62)

    print("\n[Step 1/5] 清空存量（主库三表 + FTS5 + 向量集合）...")
    step1_reset()

    print("\n[Step 2/5] 从投放目录整库重新摄取...")
    stats = run_ingest(directory, sync_mode="delta", keywords_mode=keywords_mode)
    failed = [r for r in stats["records"] if r["outcome"] == "failed"]
    quarantined = [r for r in stats["records"] if r["outcome"] == "quarantined"]

    print("\n[Step 3/5] 原始文档在位对账...")
    report = sqlite_client.verify_doc_files()
    missing = [r for r in report if r["result"] == "missing"]
    print(f"对账完成：在位 {sum(1 for r in report if r['result'] == 'ok')} 条，"
          f"文件名规范化后匹配 {sum(1 for r in report if r['result'] == 'fuzzy')} 条，"
          f"缺失 {len(missing)} 条")
    for r in missing[:10]:
        print(f"  [x] [{r['doc_version_id']}] 未找到原始文档：{r['file_path']}")
    if len(missing) > 10:
        print(f"  ... 另有 {len(missing) - 10} 条缺失（完整清单查 doc_version.file_path）")

    print("\n[Step 4/5] 存量统计与双库对账...")
    reconciled = step4_stats_and_reconcile()

    step5_probe(probe_n)

    print("\n" + "=" * 62)
    print(f"重建完成（{time.time() - t0:.0f}s）：新增 {stats['added']}，隔离 {len(quarantined)}，"
          f"失败 {len(failed)}，未变 {stats['unchanged']}")
    if failed:
        for r in failed[:5]:
            print(f"  [x] {r['path']}：{r.get('error') or r.get('detail')}")
    print("=" * 62)
    return 0 if (reconciled and not failed) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="L1 知识库全量重建（清空存量后从投放目录整库重新自动摄取）")
    parser.add_argument("directory", nargs="?", default=str(INCOMING_DIR),
                        help=f"投放目录（缺省 {INCOMING_DIR}）")
    parser.add_argument("--keywords", choices=["llm", "rule", "off"], default=None,
                        help="keywords 抽取路覆盖（缺省读 INGEST_KEYWORDS_MODE，默认 llm；"
                             "整库重建用 rule 可零联网并省下 LLM 调用）")
    parser.add_argument("--probe", type=int, default=5,
                        help="检索连通性抽测条数（默认 5，0=跳过）")
    args = parser.parse_args(argv)
    return rebuild(Path(args.directory), args.keywords, args.probe)


if __name__ == "__main__":
    sys.exit(main())
