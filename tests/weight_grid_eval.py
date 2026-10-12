"""P2 权重网格复测：在孪生密集的现状库上重扫 向量/关键词 加权比例，校准生产权重

背景：现网 VECTOR_WEIGHT/KEYWORD_WEIGHT=0.8/0.2 系旧 14 条评测集网格搜索所得；整库重建为
131 文档 / 21105 切片后跨机构/跨版本孪生显著增多，权重最优点可能漂移。本脚本一次性抓取每条
查询的候选池（原始向量相似度 + 关键词得分，min-max 归一化与权重无关，故只需抓一次），随后在
网格上仅重算加权分并排序，套用与主评测完全一致的文档级 gold 与指标口径，输出各权重组合的
Hit@1/3、Precision@3、MRR、NDCG@10、MAP@10，定位当前库真实最优配比。

召回集不随权重变化（权重只影响排序），故 Hit@10 恒定；网格只重排头部，衡量的是排序收益。

运行（项目根目录，需 L2/L3 服务未运行——Milvus Lite 单写者锁；含模型加载约 1-2 分钟）：
    python -X utf8 tests/weight_grid_eval.py
可选：--step 权重步长(默认0.1) / --eval-set 路径 / --out 结果 JSON 路径。
"""
import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tests.retrieval_metrics_eval import (  # 复用 gold 解析与指标口径，保证与主评测同源
    DEFAULT_EVAL_SET,
    aggregate,
    load_doc_rows,
    query_metrics,
    resolve_gold,
)

# 与主评测一致的候选深度；网格在完整候选池上重排
TOP_N = 20


def fetch_pool(query, retriever_mod, cand_top_k):
    """复刻一次混合检索到加权排序前，返回完整候选池 [{doc_id, similarity, keyword_score}]"""
    milvus_store = retriever_mod.milvus_store
    sqlite_client = retriever_mod.sqlite_client
    HybridRetriever = retriever_mod.HybridRetriever
    vec = milvus_store.search(query, top_k=cand_top_k)
    kw = sqlite_client.keyword_search(query, top_k=cand_top_k)
    r = HybridRetriever()
    merged = r._merge_candidates(query, vec, kw) if kw else [{**h, "keyword_score": 0.0} for h in vec]
    # 归一化与权重无关：预先算好 sim_norm/kw_norm，网格上仅线性组合
    sim_norm = HybridRetriever._minmax([h["similarity"] for h in merged])
    kw_norm = HybridRetriever._minmax([h["keyword_score"] for h in merged])
    return [
        {"doc_id": h["doc_version_id"].split(":")[0],
         "similarity": h["similarity"], "keyword_score": h["keyword_score"],
         "sn": sim_norm[h["similarity"]], "kn": kw_norm[h["keyword_score"]]}
        for h in merged
    ]


def rank_at_weight(pool, vw, kw_w):
    """按给定权重重算综合得分并降序（并列回退原始相似度），返回指标用的候选列表"""
    scored = [dict(p, score=round(p["sn"] * vw + p["kn"] * kw_w, 6)) for p in pool]
    scored.sort(key=lambda x: (x["score"], x["similarity"]), reverse=True)
    return scored


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-set", default=str(DEFAULT_EVAL_SET))
    ap.add_argument("--step", type=float, default=0.1, help="向量权重网格步长（关键词权重=1-向量权重）")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "tests" / "data" / "weight_grid_result.json"))
    args = ap.parse_args()

    import src.data.config as data_config
    from src.data.hybrid_retriever import HybridRetriever
    from src.data.milvus_client import milvus_store
    from src.data.sqlite_client import sqlite_client
    from src.config import SQLITE_DB_PATH

    retriever_mod = type("M", (), {"milvus_store": milvus_store, "sqlite_client": sqlite_client,
                                   "HybridRetriever": HybridRetriever})()

    eval_set = json.loads(Path(args.eval_set).read_text(encoding="utf-8"))["queries"]
    doc_rows = load_doc_rows(SQLITE_DB_PATH)
    cand_top_k = data_config.CANDIDATE_TOP_K

    # 预热（加载模型/集合），不计入抓取
    HybridRetriever().search("预热查询：融资融券", top_k=TOP_N)

    # 每查询抓一次候选池（权重无关），gold 一并解析
    print(f"抓取 {len(eval_set)} 条查询的完整候选池（每查询一次双路检索）...")
    t0 = time.time()
    query_pools = []
    for item in eval_set:
        gold = resolve_gold(doc_rows, item["accept_doc_names"], item.get("auto_only", False))
        pool = fetch_pool(item["query"], retriever_mod, cand_top_k)
        query_pools.append((item, gold, pool))
    print(f"抓取完成（{time.time() - t0:.1f}s），网格重扫（仅重算加权排序，不再触发检索）\n")

    # 网格：向量权重 0.0~1.0，关键词权重 = 1 - 向量权重
    steps = round(1.0 / args.step) + 1
    grid = [round(i * args.step, 3) for i in range(steps)]
    current_vw = data_config.VECTOR_WEIGHT
    rows = []
    for vw in grid:
        kw_w = round(1 - vw, 3)
        per_query = []
        rank_dist = {}
        for item, gold, pool in query_pools:
            ranked = rank_at_weight(pool, vw, kw_w)
            m = query_metrics(ranked, gold)
            per_query.append(m)
            rk = m["first_relevant_rank"]
            rank_dist[rk or ">10"] = rank_dist.get(rk or ">10", 0) + 1
        agg = aggregate(per_query)
        rows.append({"vector_weight": vw, "keyword_weight": kw_w, **agg, "rank_dist": rank_dist})

    header = (f"{'vec':>5}{'kw':>6}{'Hit@1':>8}{'Hit@3':>8}{'Hit@5':>8}{'P@3':>8}"
              f"{'MRR':>8}{'NDCG@10':>9}{'MAP@10':>8}")
    print("=" * len(header))
    print("权重网格搜索（现状库·文档级 gold·召回集恒定，仅重排头部）")
    print("=" * len(header))
    print(header)
    best = max(rows, key=lambda r: (r["rr"], r["hit@1"]))
    for r in rows:
        tag = ""
        if abs(r["vector_weight"] - current_vw) < 1e-6:
            tag += "  ←现网"
        if r is best:
            tag += "  ★最优(按MRR)"
        print(f"{r['vector_weight']:>5.1f}{r['keyword_weight']:>6.1f}"
              f"{r['hit@1']:>8.3f}{r['hit@3']:>8.3f}{r['hit@5']:>8.3f}{r['precision@3']:>8.3f}"
              f"{r['rr']:>8.3f}{r['ndcg@10']:>9.3f}{r['map@10']:>8.3f}{tag}")

    cur = next(r for r in rows if abs(r["vector_weight"] - current_vw) < 1e-6)
    print(f"\n现网 vec {current_vw}：MRR={cur['rr']} Hit@1={cur['hit@1']} NDCG@10={cur['ndcg@10']}")
    print(f"最优 vec {best['vector_weight']}：MRR={best['rr']} Hit@1={best['hit@1']} NDCG@10={best['ndcg@10']}")
    print("→ 最优相对现网增益："
          f"MRR {best['rr'] - cur['rr']:+.3f}｜Hit@1 {best['hit@1'] - cur['hit@1']:+.3f}｜"
          f"NDCG@10 {best['ndcg@10'] - cur['ndcg@10']:+.3f}")

    payload = {
        "meta": {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "eval_set": Path(args.eval_set).name,
                 "n_queries": len(eval_set), "top_n": TOP_N,
                 "current_weights": [data_config.VECTOR_WEIGHT, data_config.KEYWORD_WEIGHT],
                 "step": args.step},
        "grid": rows,
        "current": cur, "best": best,
    }
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n结果 JSON 已写入 {Path(args.out).name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
