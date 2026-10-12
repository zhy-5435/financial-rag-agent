"""Phase 5 检索质量复测：36 条评测集 × 双环境对照 + 双路权重网格复测

评测问题（切块对齐评测报告第 9 节遗留）：自动摄取通道投满 23 份文档后，
既有检索质量是否被稀释？新通道内容是否可召回？0.8/0.2 权重平台是否仍稳？

方法：
  基线环境 = 正式库现状（人工通道 596 切片），只读查询；
  投放后环境 = 正式库【临时副本】+ run_ingest(data/raw/正式版文档)（AUTO 通道
  ~600 片孪生/独有内容入库）后查询——与基线逐指标对照即「稀释」检验。
  gold 为文档级宽松标注（命中人工版或 AUTO 同名版均算），评测集见
  tests/data/retrieval_eval_set.json；权重网格在投放后候选集上内存重排复测，
  不改动生产配置。

断言组（E1-E7）：基线达标 / 稀释不越界 / 逐条回退限量 / AUTO 版自身可召回（另报宽松
口径拆分孪生压制）/ gold 交叉验证 / 0.8-0.2 处于网格最优平台 / 投放后向量对账一致；
A-12 费率残留失败为诊断项不计分。

运行（项目根目录，需 L2/L3 服务未运行——Milvus Lite 单写者锁；
     两阶段子进程各加载一次 embedding 模型，投放阶段含 AUTO 全量入库，全程约 4-8 分钟）：
    python -X utf8 tests/retrieval_eval.py
退出码 0=全部 PASS。--phase 子模式仅供父进程编排调用。
"""
import argparse
import json
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

EVAL_SET_PATH = PROJECT_ROOT / "tests" / "data" / "retrieval_eval_set.json"
INCOMING_DIR = PROJECT_ROOT / "data" / "raw" / "正式版文档"
TOP_N = 20        # 与 CANDIDATE_TOP_K(10)×两路去重上限对齐：search(top_k=20) 即完整候选集
GRID = [(vw, kw) for vw in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5) for kw in (0.0, 0.1, 0.2, 0.3)]
CUR_WEIGHT = (0.8, 0.2)

_failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" | {detail}" if detail else ""))
    if not cond:
        _failures.append(name)


# --------------------------------------------------------------------------- 查询阶段

def resolve_accept(sqlite_path: Path, accept_names: list[str], auto_only: bool = False) -> set[str]:
    """文档级 gold：valid 版本中 doc_name 含任一主题子串的 doc_id 集合（孪生双通道自动并入）；
    auto_only=True 时只保留 AUTO- 前缀——B 组考「自动通道切片自身可召回」，而非人工孪生版"""
    conn = sqlite3.connect(sqlite_path)
    try:
        rows = conn.execute("SELECT doc_id, doc_name FROM doc_version WHERE status='valid'").fetchall()
    finally:
        conn.close()
    ids = {doc_id for doc_id, name in rows if any(s in (name or "") for s in accept_names)}
    return {d for d in ids if d.startswith("AUTO-")} if auto_only else ids


def run_queries(sqlite_path: Path, eval_set: list[dict]) -> list[dict]:
    """全评测集走生产检索链路（hybrid_retriever.search，RERANK 关闭态与线上一致）"""
    from src.data.hybrid_retriever import hybrid_retriever
    results = []
    for item in eval_set:
        accept_ids = resolve_accept(sqlite_path, item["accept_doc_names"], item.get("auto_only", False))
        accept_any = resolve_accept(sqlite_path, item["accept_doc_names"])  # 宽松口径：孪生任意版本
        hits = hybrid_retriever.search(item["query"], top_k=TOP_N)
        cands = [{
            "knowledge_id": h["knowledge_id"],
            "doc_id": h["doc_version_id"].split(":")[0],
            "similarity": h["similarity"],
            "keyword_score": h["keyword_score"],
            "content_head": h["content"][:50],
        } for h in hits]
        results.append({**{k: item[k] for k in ("id", "group", "query", "note")},
                        "accept_doc_ids": sorted(accept_ids),
                        "accept_all_doc_ids": sorted(accept_any), "candidates": cands})
    return results


def phase_main(args) -> None:
    """子进程模式：baseline=正式库只读查询；after=副本投放 AUTO 通道后查询"""
    import src.config
    if args.phase == "after":
        src.config.SQLITE_DB_PATH = Path(args.sqlite)
        src.config.MILVUS_DB_PATH = Path(args.milvus)
    from src.data.ingestion.runner import run_ingest

    t0 = time.time()
    payload: dict = {"phase": args.phase, "top_n": TOP_N}
    if args.phase == "after":
        # patch 路径必须先于一切 src.data.* 导入（全局实例按副本路径构建）
        from src.data.ingestion.runner import run_ingest
        # 与 Phase 4 演练同口径：keywords 固定 rule 档，评测确定性零联网
        stats = run_ingest(INCOMING_DIR, keywords_mode="rule")
        stats.pop("records", None)
        payload["ingest_stats"] = {k: v for k, v in stats.items() if isinstance(v, (int, bool, tuple, type(None)))}

    eval_set = json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))["queries"]
    payload["results"] = run_queries(src.config.SQLITE_DB_PATH, eval_set)

    if args.phase == "after":
        # 向量库 ↔ 主库双有效行数对账（P0 增量同步的逐轮对账约定）
        from src.data.milvus_client import milvus_store
        conn = sqlite3.connect(src.config.SQLITE_DB_PATH)
        n_live = conn.execute("SELECT COUNT(*) FROM knowledge_chunk k JOIN doc_version d "
                              "ON k.doc_version_id=d.doc_version_id "
                              "WHERE k.status='valid' AND d.status='valid'").fetchone()[0]
        conn.close()
        payload["reconcile"] = {"milvus_rows": milvus_store.count_rows(), "live_chunks": n_live}
    payload["elapsed_s"] = round(time.time() - t0, 1)
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


# --------------------------------------------------------------------------- 指标计算

def _first_accept_rank(cands: list[dict], accept: set[str]) -> int:
    for r, c in enumerate(cands, 1):
        if c["doc_id"] in accept:
            return r
    return 0


def metrics(results: list[dict], weights: tuple | None = None, gold_key: str = "accept_doc_ids") -> dict:
    """文档级 gold 指标；weights 非 None 时在候选集上按该权重内存重排（网格复测用）；
    gold_key 可指向宽松口径（accept_all_doc_ids，孪生任意版本均计命中）"""
    from src.data.hybrid_retriever import HybridRetriever
    out = {"n": 0, "hit@1": 0.0, "hit@3": 0.0, "recall@3": 0.0, "mrr@3": 0.0, "miss": []}
    h1 = h3 = r3 = mrr = 0.0
    for q in results:
        cands = q["candidates"]
        if weights is not None:
            sim_norm = HybridRetriever._minmax([c["similarity"] for c in cands])
            kw_norm = HybridRetriever._minmax([c["keyword_score"] for c in cands])
            cands = sorted(
                cands,
                key=lambda c: (sim_norm[c["similarity"]] * weights[0] + kw_norm[c["keyword_score"]] * weights[1],
                               c["similarity"]),
                reverse=True,
            )
        accept = set(q[gold_key])
        rank = _first_accept_rank(cands, accept)
        n_top3 = sum(1 for c in cands[:3] if c["doc_id"] in accept)
        h1 += 1.0 if rank == 1 else 0.0
        h3 += 1.0 if 0 < rank <= 3 else 0.0
        r3 += n_top3 / 3
        mrr += 1.0 / rank if 0 < rank <= 3 else 0.0
        if not 0 < rank <= 3:
            out["miss"].append({"id": q["id"], "rank": rank or None,
                                "top1": cands[0]["content_head"] if cands else "无结果"})
        out["n"] += 1
    n = out["n"] or 1
    out.update({"hit@1": round(h1 / n, 3), "hit@3": round(h3 / n, 3),
                "recall@3": round(r3 / n, 3), "mrr@3": round(mrr / n, 3)})
    return out


def fmt(m: dict) -> str:
    return (f"n={m['n']} Hit@1 {m['hit@1']:.3f} Hit@3 {m['hit@3']:.3f} "
            f"Recall@3 {m['recall@3']:.3f} MRR@3 {m['mrr@3']:.3f}")


# --------------------------------------------------------------------------- 父进程编排

def spawn(phase: str, out: Path, sqlite: Path | None = None, milvus: Path | None = None) -> None:
    cmd = [sys.executable, "-X", "utf8", str(Path(__file__).resolve()),
           "--phase", phase, "--out", str(out)]
    if sqlite:
        cmd += ["--sqlite", str(sqlite), "--milvus", str(milvus)]
    t0 = time.time()
    print(f"[{phase}] 子进程运行中…")
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if proc.returncode != 0:
        raise SystemExit(f"[{phase}] 阶段失败（exit={proc.returncode}）")
    print(f"[{phase}] 完成（{time.time() - t0:.0f}s）")


def main_parent() -> int:
    t0 = time.time()
    tmp_root = Path(tempfile.mkdtemp(prefix="retrieval_eval_"))
    try:
        return evaluate(tmp_root, t0)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def evaluate(tmp_root: Path, t0: float) -> int:
    import src.config
    # 副本须在打开正式库之前拷贝：投放后环境的底座
    cp_sqlite = tmp_root / "l1_core.db"
    cp_milvus = tmp_root / "l1_milvus.db"
    shutil.copy2(src.config.SQLITE_DB_PATH, cp_sqlite)
    shutil.copytree(src.config.MILVUS_DB_PATH, cp_milvus)

    base_json, after_json = tmp_root / "base.json", tmp_root / "after.json"
    spawn("baseline", base_json)
    spawn("after", after_json, cp_sqlite, cp_milvus)
    base = json.loads(base_json.read_text(encoding="utf-8"))
    after = json.loads(after_json.read_text(encoding="utf-8"))

    b_a = [q for q in base["results"] if q["group"] == "A"]
    a_a = [q for q in after["results"] if q["group"] == "A"]
    a_b = [q for q in after["results"] if q["group"] == "B"]
    m_base, m_after_a, m_after_b = metrics(b_a), metrics(a_a), metrics(a_b)
    # B 组宽松口径：同主题孪生人工版占据 top 位也算用户拿到了答案（孪生压制的量化拆分）
    m_after_b_loose = metrics(a_b, gold_key="accept_all_doc_ids")
    # B 组在基线环境同样可查（内容缺失时应全数落空），作 gold 有效性交叉验证
    b_b = [q for q in base["results"] if q["group"] == "B"]
    m_base_b = metrics(b_b)

    print("\n" + "=" * 66)
    print("Phase 5 检索质量复测（A=人工通道存量问法 / B=AUTO 通道新问法）")
    print("=" * 66)
    print(f"基线·A 组（正式库现状）        {fmt(m_base)}")
    print(f"投放后·A 组（+AUTO 23 文档）   {fmt(m_after_a)}")
    print(f"投放后·B 组（AUTO 版自身可召回）  {fmt(m_after_b)}")
    print(f"投放后·B 组（宽松口径含孪生人工版）{fmt(m_after_b_loose)}")
    print(f"基线·B 组（auto_only 应全数落空）    {fmt(m_base_b)}")

    # ---- 权重网格复测（投放后合并集，内存重排） ----
    grid_rows = sorted(
        ((vw, kw, metrics(after["results"], weights=(vw, kw))) for vw, kw in GRID),
        key=lambda r: (-r[2]["mrr@3"], -r[2]["hit@1"]))
    print("\n权重网格（投放后·合并 36 条，Top5 + 现值）：")
    for vw, kw, m in grid_rows[:5]:
        print(f"  vec {vw:.1f} / kw {kw:.1f} -> Hit@1 {m['hit@1']:.3f} Hit@3 {m['hit@3']:.3f} MRR@3 {m['mrr@3']:.3f}")
    cur = next(m for vw, kw, m in grid_rows if (vw, kw) == CUR_WEIGHT)
    best = grid_rows[0][2]
    print(f"  当前 {CUR_WEIGHT[0]}/{CUR_WEIGHT[1]} -> Hit@1 {cur['hit@1']:.3f} Hit@3 {cur['hit@3']:.3f} MRR@3 {cur['mrr@3']:.3f}")

    # ---- 诊断项（不计分）：旧报告残留失败 A-12 ----
    diag = next(q for q in after["results"] if q["id"] == "A-12")
    rank12 = _first_accept_rank(diag["candidates"], set(diag["accept_doc_ids"]))
    print(f"\n[诊断] A-12 信银199费率：基线 rank={_first_accept_rank(next(q for q in base['results'] if q['id']=='A-12')['candidates'], set(next(q for q in base['results'] if q['id']=='A-12')['accept_doc_ids'])) or '>20'}"
          f" → 投放后 rank={rank12 or '>20'}（AUTO 前缀切片是否解锁跨切片割裂）")

    # ---- 断言 ----
    print("\n断言：")
    check("E1 基线 A 组 Hit@3 ≥ 0.75", m_base["hit@3"] >= 0.75, fmt(m_base))
    check("E2 稀释检验：投放后 A 组 Hit@3 不低于基线 − 0.05",
          m_after_a["hit@3"] >= m_base["hit@3"] - 0.05,
          f"{m_after_a['hit@3']:.3f} vs {m_base['hit@3']:.3f}")
    check("E2 稀释检验：投放后 A 组 MRR@3 不低于基线 − 0.05",
          m_after_a["mrr@3"] >= m_base["mrr@3"] - 0.05,
          f"{m_after_a['mrr@3']:.3f} vs {m_base['mrr@3']:.3f}")
    regress = [q["id"] for q, bq in zip(a_a, b_a)
               if _first_accept_rank(q["candidates"], set(q["accept_doc_ids"])) > 3
               and 0 < _first_accept_rank(bq["candidates"], set(bq["accept_doc_ids"])) <= 3]
    check("E3 逐条对照：基线 top3 命中→投放后跌出 top3 的回退问法 ≤ 1 条",
          len(regress) <= 1, f"回退：{regress or '无'}")
    check("E4 新通道可召回：B 组 AUTO 版自身投放后 Hit@3 ≥ 0.70（孪生压制下实测水平）",
          m_after_b["hit@3"] >= 0.70, fmt(m_after_b))
    check("E4 新通道可召回：B 组 AUTO 版自身投放后 Hit@1 ≥ 0.50", m_after_b["hit@1"] >= 0.50, fmt(m_after_b))
    check("E4 孪生压制拆解：B 组宽松口径 Hit@3 不低于严格口径，且两者差即被人工版代打条数",
          m_after_b_loose["hit@3"] >= m_after_b["hit@3"],
          f"宽松 {m_after_b_loose['hit@3']:.3f} vs 严格 {m_after_b['hit@3']:.3f}")
    check("E5 gold 交叉验证：B 组（auto_only）在基线环境全部落空（基线无 AUTO 文档）",
          m_base_b["hit@3"] == 0.0, fmt(m_base_b))
    check("E6 权重复测：现值 0.8/0.2 MRR@3 处于网格最优 − 0.02 平台内",
          cur["mrr@3"] >= best["mrr@3"] - 0.02,
          f"现值 {cur['mrr@3']:.3f}，最优 {best['mrr@3']:.3f}（{grid_rows[0][0]}/{grid_rows[0][1]}）")
    rec = after.get("reconcile") or {}
    check("E7 投放后向量库与主库双有效行数对账一致（delta 链路）",
          rec.get("milvus_rows") == rec.get("live_chunks"),
          f"milvus {rec.get('milvus_rows')} vs 主库 {rec.get('live_chunks')}")

    # 失败明细辅助排查
    for tag, m in (("基线A", m_base), ("投放后A", m_after_a), ("投放后B", m_after_b)):
        if m["miss"]:
            print(f"\n未命中明细[{tag}]：" + "; ".join(f"{x['id']}(rank={x['rank'] or '-'}, top1={x['top1']})" for x in m["miss"]))

    print("\n" + "=" * 66)
    print(f"复测结束（{time.time() - t0:.0f}s）：" + ("全部 PASS ✓" if not _failures else f"FAIL {len(_failures)} 项：{_failures}"))
    print("=" * 66)
    return 0 if not _failures else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=["baseline", "after"], help=argparse.SUPPRESS)
    ap.add_argument("--out", help=argparse.SUPPRESS)
    ap.add_argument("--sqlite", help=argparse.SUPPRESS)
    ap.add_argument("--milvus", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.phase:
        phase_main(args)
        return 0
    return main_parent()


if __name__ == "__main__":
    sys.exit(main())
