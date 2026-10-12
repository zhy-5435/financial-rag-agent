"""检索质量全量指标评测：Recall@K / Precision@K / MRR / NDCG@K / MAP@K + 检索时延分布

对生产检索链路（hybrid_retriever，双路混合召回 + Reranker 精排，开关态与线上一致，
运行头行会打印当前 RERANK 态）做标准化信息检索评测，回答三个问题：召回够不够准
（ Precision/Recall/MRR）、排得够不够好（NDCG/MAP）、响应够不够快（端到端与分路时延分布）。

指标口径（标准 IR 定义，与旧报告的「Recall@3=前3名纯度」口径不同，不可直接对比）：
  gold 为文档级标注（retrieval_eval_set_v2.json，运行期按 doc_name 子串解析 doc_id 集合），
  候选切片所属 doc_id 落在 gold 集合内即计为「相关切片」：
  - Hit@K       前 K 条中存在相关切片的查询占比（能不能拿到答案）
  - Precision@K 前 K 条中相关切片数 / K（头部结果纯度，衡量杂质）
  - Recall@K    前 K 条覆盖到的 gold 文档数 / gold 文档总数（文档覆盖率）
  - MRR         首个相关切片排名的倒数均值（全候选集内计算，不设截断）
  - MRR@10      首相关切片排进前 10 才计分，否则记 0
  - NDCG@10     二元相关性的归一化折损累计增益，IDCG 按候选集内可见的相关切片数
                （文档级标注无法知晓全库相关切片真值，取列表内归一，属保守口径）
  - MAP@10      前 10 条内逐位置精度的均值（对多相关切片场景补充区分度）
  - 时延        每查询预热 1 次后连续 R 次全链路检索取【中位数】，再对全查询取
                mean/P50/P90/P95/max；分路时延同法测向量路（含查询向量化）与
                关键词路（FTS5），用于定位耗时瓶颈。

运行（项目根目录，需 L2/L3 服务未运行——Milvus Lite 单写者锁；
     首查询加载 embedding 模型约 10-30s，测量本身约 2-4 分钟）：
    python -X utf8 tests/retrieval_metrics_eval.py
可选参数：--eval-set 路径 / --runs 计时次数(默认5) / --top-n 候选深度(默认20) /
          --out 结果 JSON 输出路径（默认 tests/data/retrieval_metrics_result.json，
          为报告留档产物，重跑即覆盖）。
退出码 0=检查项全部 PASS（S1 gold 全解析 / S2 候选非空 / S3 对账一致）。
"""
import argparse
import json
import math
import sqlite3
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_EVAL_SET = PROJECT_ROOT / "tests" / "data" / "retrieval_eval_set_v2.json"
KS = (1, 3, 5, 10)   # 指标截断深度；须 ≤ top-n

_checks: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" | {detail}" if detail else ""))
    if not cond:
        _checks.append(name)


# --------------------------------------------------------------------------- gold 解析

def load_doc_rows(sqlite_path: Path) -> list[tuple[str, str]]:
    conn = sqlite3.connect(sqlite_path)
    try:
        return conn.execute(
            "SELECT doc_id, doc_name FROM doc_version WHERE status='valid'").fetchall()
    finally:
        conn.close()


def resolve_gold(doc_rows: list[tuple[str, str]], accept_names: list[str], auto_only: bool) -> set[str]:
    """文档级 gold：valid 文档中 doc_name 含任一主题子串的 doc_id；auto_only 仅保留 AUTO- 前缀"""
    ids = {doc_id for doc_id, name in doc_rows if any(s in (name or "") for s in accept_names)}
    return {d for d in ids if d.startswith("AUTO-")} if auto_only else ids


# --------------------------------------------------------------------------- 单查询指标

def query_metrics(cands: list[dict], gold: set[str]) -> dict:
    """按 KS 计算单查询的 Hit/Precision/Recall 与 MRR/MRR@10/NDCG@10/MAP@10（二元相关）"""
    rel = [c["doc_id"] in gold for c in cands]
    first_rank = next((i for i, r in enumerate(rel, 1) if r), 0)
    gold_docs_total = len(gold) or 1
    out = {"first_relevant_rank": first_rank or None}
    for k in KS:
        top = rel[:k]
        n_rel = sum(top)
        docs_covered = len({c["doc_id"] for c, r in zip(cands[:k], top) if r})
        out[f"hit@{k}"] = 1.0 if n_rel else 0.0
        out[f"precision@{k}"] = n_rel / k
        out[f"recall@{k}"] = docs_covered / gold_docs_total
    # MRR 全候选集口径；MRR@10 截断
    out["rr"] = 1.0 / first_rank if first_rank else 0.0
    out["rr@10"] = 1.0 / first_rank if first_rank and first_rank <= 10 else 0.0
    # NDCG@10：IDCG 取候选集内相关切片数 min(rel_total,10) 的理想排名（列表内归一，保守口径）
    dcg = sum(1.0 / math.log2(i + 1) for i, r in enumerate(rel[:10], 1) if r)
    ideal_n = min(sum(rel), 10)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_n + 1))
    out["ndcg@10"] = dcg / idcg if idcg else 0.0
    # MAP@10：相关位置精度的均值（分母取 min(相关切片数,10)）
    hits = prec_sum = 0
    for i, r in enumerate(rel[:10], 1):
        if r:
            hits += 1
            prec_sum += hits / i
    out["map@10"] = prec_sum / min(sum(rel), 10) if rel[:10] and sum(rel) else 0.0
    return out


def aggregate(rows: list[dict]) -> dict:
    """对查询级指标取均值（rows 为 query_metrics 输出与元信息的合并记录）"""
    n = len(rows) or 1
    agg = {"n": len(rows)}
    for key in (*[f"{m}@{k}" for k in KS for m in ("hit", "precision", "recall")],
                "rr", "rr@10", "ndcg@10", "map@10"):
        agg[key] = round(sum(r[key] for r in rows) / n, 3)
    return agg


# --------------------------------------------------------------------------- 时延统计

def latency_stats(values: list[float]) -> dict:
    """对每查询中位时延序列做分布统计（P90/P95 用最近秩法）"""
    if not values:
        return {}
    s = sorted(values)
    rank = lambda p: s[min(len(s) - 1, max(0, round(p / 100 * len(s)) - 1))]  # noqa: E731
    return {"mean": round(statistics.mean(s), 1), "p50": round(rank(50), 1),
            "p90": round(rank(90), 1), "p95": round(rank(95), 1),
            "min": round(s[0], 1), "max": round(s[-1], 1),
            "stdev": round(statistics.stdev(s), 1) if len(s) > 1 else 0.0}


# --------------------------------------------------------------------------- 主流程

def timed(fn, runs: int) -> tuple[list, list[float]]:
    """连续 runs 次执行 fn，返回 (最后一次结果, 每次耗时 ms 列表)"""
    out, ts = None, []
    for _ in range(runs):
        t0 = time.perf_counter()
        out = fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return out, ts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-set", default=str(DEFAULT_EVAL_SET))
    ap.add_argument("--runs", type=int, default=5, help="每查询计时次数（预热 1 次不计）")
    ap.add_argument("--top-n", type=int, default=20, help="候选深度（与 CANDIDATE_TOP_K×两路上限对齐）")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "tests" / "data" / "retrieval_metrics_result.json"))
    args = ap.parse_args()

    from src.data import config as data_config
    from src.data.config import CANDIDATE_TOP_K
    from src.data.hybrid_retriever import hybrid_retriever
    from src.data.milvus_client import milvus_store
    from src.data.sqlite_client import sqlite_client
    from src.config import MILVUS_DB_PATH, SQLITE_DB_PATH

    eval_set = json.loads(Path(args.eval_set).read_text(encoding="utf-8"))["queries"]
    doc_rows = load_doc_rows(SQLITE_DB_PATH)
    conn = sqlite3.connect(SQLITE_DB_PATH)
    n_chunks = conn.execute(
        "SELECT COUNT(*) FROM knowledge_chunk k JOIN doc_version d ON k.doc_version_id=d.doc_version_id "
        "WHERE k.status='valid' AND d.status='valid'").fetchone()[0]
    n_fts = conn.execute("SELECT COUNT(*) FROM fts_l1_knowledge").fetchone()[0]
    conn.close()
    n_vectors = milvus_store.count_rows()

    print(f"环境：{len([1 for _, s in doc_rows])} valid 文档 / {n_chunks} 双有效切片 / "
          f"向量 {n_vectors} / FTS {n_fts}｜权重 vec {data_config.VECTOR_WEIGHT}/kw {data_config.KEYWORD_WEIGHT}"
          f"｜RERANK={data_config.RERANK_ENABLED}｜候选深度 top_n={args.top_n}｜runs={args.runs}")

    # 预热：首查询触发 embedding 模型加载与集合 load，不计入任何测量
    t0 = time.time()
    hybrid_retriever.search("预热查询：融资融券", top_k=args.top_n)
    print(f"预热完成（模型加载 {time.time() - t0:.1f}s，未计入测量）\n")

    per_query: list[dict] = []
    unresolved: list[str] = []
    empty_cands: list[str] = []
    lat_full, lat_vec, lat_kw = [], [], []

    for item in eval_set:
        gold = resolve_gold(doc_rows, item["accept_doc_names"], item.get("auto_only", False))
        if not gold:
            unresolved.append(item["id"])
        # 计时全链路：runs 次取中位数；候选集取最后一次（各次内容一致，排序确定）
        cands_raw, ts_full = timed(lambda: hybrid_retriever.search(item["query"], top_k=args.top_n), args.runs)
        _, ts_vec = timed(lambda: milvus_store.search(item["query"], top_k=CANDIDATE_TOP_K), args.runs)
        _, ts_kw = timed(lambda: sqlite_client.keyword_search(item["query"], top_k=CANDIDATE_TOP_K), args.runs)
        med_full = statistics.median(ts_full)
        lat_full.append(med_full)
        lat_vec.append(statistics.median(ts_vec))
        lat_kw.append(statistics.median(ts_kw))

        cands = [{"knowledge_id": h["knowledge_id"],
                  "doc_id": h["doc_version_id"].split(":")[0],
                  "similarity": h["similarity"], "keyword_score": h["keyword_score"],
                  "content_head": h["content"][:60]} for h in cands_raw]
        if not cands:
            empty_cands.append(item["id"])
        m = query_metrics(cands, gold)
        per_query.append({
            **{k: item[k] for k in ("id", "group", "query", "note")},
            "gold_doc_ids": sorted(gold), "n_candidates": len(cands),
            **m,
            "latency_ms": {"full": round(med_full, 1),
                           "full_all_runs": [round(t, 1) for t in ts_full],
                           "vector_path": round(statistics.median(ts_vec), 1),
                           "keyword_path": round(statistics.median(ts_kw), 1)},
            "top3": [c["content_head"] for c in cands[:3]],
        })
        mark = "✓" if m["hit@3"] else "✗"
        print(f"  {mark} {item['id']} rank={m['first_relevant_rank'] or '-'} "
              f"P@3={m['precision@3']:.2f} 中位时延={med_full:.0f}ms | {item['query'][:30]}")

    # ---- 汇总 ----
    groups = sorted({q["group"] for q in per_query})
    overall = aggregate(per_query)
    by_group = {g: aggregate([q for q in per_query if q["group"] == g]) for g in groups}

    header = f"{'口径':<14}" + "".join(f"{'Hit@' + str(k):>8}" for k in KS) + \
             "".join(f"{'P@' + str(k):>8}" for k in KS) + "".join(f"{'R@' + str(k):>8}" for k in KS) + \
             f"{'MRR':>8}{'MRR@10':>8}{'NDCG@10':>9}{'MAP@10':>8}"

    def row(tag: str, a: dict) -> str:
        return (f"{tag:<14}" + "".join(f"{a[f'hit@{k}']:>8.3f}" for k in KS)
                + "".join(f"{a[f'precision@{k}']:>8.3f}" for k in KS)
                + "".join(f"{a[f'recall@{k}']:>8.3f}" for k in KS)
                + f"{a['rr']:>8.3f}{a['rr@10']:>8.3f}{a['ndcg@10']:>9.3f}{a['map@10']:>8.3f}")

    print("\n" + "=" * len(header))
    print("检索指标汇总（文档级 gold，标准 IR 口径）")
    print("=" * len(header))
    print(header)
    print(row("总体", overall))
    for g in groups:
        print(row(g, by_group[g]))

    lat = {"full": latency_stats(lat_full), "vector_path": latency_stats(lat_vec),
           "keyword_path": latency_stats(lat_kw)}
    print("\n检索时延（ms，每查询 {} 次取中位后跨 {} 查询分布）".format(args.runs, len(per_query)))
    print(f"{'链路':<12}{'mean':>8}{'P50':>8}{'P90':>8}{'P95':>8}{'min':>8}{'max':>8}{'stdev':>8}")
    for tag, key in (("全链路(双路+归一加权)", "full"), ("向量路(含查询向量化)", "vector_path"),
                     ("关键词路(FTS5 BM25)", "keyword_path")):
        s = lat[key]
        print(f"{tag:<12}{s['mean']:>8.1f}{s['p50']:>8.1f}{s['p90']:>8.1f}{s['p95']:>8.1f}"
              f"{s['min']:>8.1f}{s['max']:>8.1f}{s['stdev']:>8.1f}")

    # ---- 检查项 ----
    print("\n检查项：")
    check("S1 gold 全部可解析（评测集与当前库对齐）", not unresolved,
          f"未解析：{unresolved or '无'}")
    check("S2 所有查询候选集非空", not empty_cands, f"空结果：{empty_cands or '无'}")
    check("S3 向量库与主库有效行数对账一致", n_vectors == n_chunks,
          f"milvus {n_vectors} vs 主库 {n_chunks}")

    misses = [q for q in per_query if not q["hit@10"]]
    if misses:
        print("\n未命中明细（top10 内无相关切片）：")
        for q in misses:
            print(f"  {q['id']} {q['query']} | gold={len(q['gold_doc_ids'])}文档 "
                  f"| top1={q['top3'][0] if q['top3'] else '无结果'}")

    payload = {
        "meta": {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"), "eval_set": str(Path(args.eval_set).name),
            "n_queries": len(per_query), "top_n": args.top_n, "runs": args.runs,
            "valid_docs": len(doc_rows), "live_chunks": n_chunks,
            "milvus_rows": n_vectors, "fts_rows": n_fts,
            "embedding_model": data_config.EMBEDDING_MODEL_NAME,
            "weights": [data_config.VECTOR_WEIGHT, data_config.KEYWORD_WEIGHT],
            "rerank_enabled": data_config.RERANK_ENABLED,
            "candidate_top_k": CANDIDATE_TOP_K,
        },
        "overall": overall, "by_group": by_group, "latency_ms": lat,
        "checks": {"pass": not _checks, "failed": _checks,
                   "unresolved_gold": unresolved, "empty_candidates": empty_cands},
        "queries": per_query,
    }
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n结果 JSON 已写入 {Path(args.out).name}")
    return 0 if not _checks else 1


if __name__ == "__main__":
    sys.exit(main())
