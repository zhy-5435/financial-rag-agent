"""全量重建入口评测：pipeline 清库重投 + 幂等 + 正式库零污染

Excel 人工切片通道移除后，`python -m src.data.pipeline` 是唯一的整库重建入口：
清空三表与双索引 → 从投放目录重新自动摄取 → 原始文档对账 → 双库对账 → 检索连通性抽测。
本脚本在正式库的临时副本上跑两遍重建，验证上述链路，并确认正式库全程只读。

运行（项目根目录，需 L2/L3 服务未运行以释放 MilvusLite 锁；纯本地，keywords 走规则路不调 LLM）：
    python -X utf8 tests/rebuild_pipeline_eval.py
退出码 0=全部 PASS。
"""
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

# 用于重建演练的真实文档（小份制度文书，走 rule 档零联网）
SEED_DOCS = ("内部制度类/考勤与休假管理制度.txt", "内部制度类/岗位权责规范.txt")
SRC_DIR = PROJECT_ROOT / "data" / "raw" / "正式版文档"

_failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" | {detail}" if detail else ""))
    if not cond:
        _failures.append(name)


# --------------------------------------------------------------------------- 子进程（沙箱内重建）

def phase_child(sandbox: Path, incoming: Path) -> None:
    """在子进程里把全局路径指向沙箱后执行重建：src.data.* 的全局单例按导入时路径绑定，
    因此必须先改 src.config 再首次导入 pipeline（正式库在这一进程里从未被打开）"""
    import src.config
    src.config.SQLITE_DB_PATH = sandbox / "l1_core.db"
    src.config.MILVUS_DB_PATH = sandbox / "milvus" / "l1_milvus.db"
    from src.data.pipeline import rebuild
    from src.data.ingestion.runner import run_ingest
    from src.data.milvus_client import milvus_store
    from src.data.sqlite_client import sqlite_client

    out: dict = {}
    rc = rebuild(incoming, keywords_mode="rule", probe_n=3)
    out["rebuild_rc"] = rc
    out["after_rebuild"] = snapshot(src.config.SQLITE_DB_PATH, milvus_store)

    # 幂等：无改动重跑应零写库、零索引操作，存量不变
    stats = run_ingest(incoming, keywords_mode="rule")
    out["rerun"] = {k: stats[k] for k in ("added", "changed", "removed", "unchanged")}
    out["after_rerun"] = snapshot(src.config.SQLITE_DB_PATH, milvus_store)
    # 沙箱库的 FTS 行数与 valid 切片对账（_ensure_fts_index 的自动补建是否生效）
    conn = sqlite3.connect(src.config.SQLITE_DB_PATH)
    out["fts_rows"] = conn.execute("SELECT COUNT(*) FROM fts_l1_knowledge").fetchone()[0]
    conn.close()
    (sandbox / "child.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")


def snapshot(db_path: Path, milvus_store) -> dict:
    conn = sqlite3.connect(db_path)
    try:
        docs = conn.execute("SELECT COUNT(*) FROM doc_version").fetchone()[0]
        live = conn.execute(
            "SELECT COUNT(*) FROM knowledge_chunk k JOIN doc_version d "
            "ON k.doc_version_id=d.doc_version_id "
            "WHERE k.status='valid' AND d.status='valid'").fetchone()[0]
        prefixes = sorted({r[0].split("-")[0] for r in
                           conn.execute("SELECT doc_id FROM doc_version").fetchall()})
        ingest_rows = conn.execute("SELECT COUNT(*) FROM ingest_state").fetchone()[0]
    finally:
        conn.close()
    return {"docs": docs, "live_chunks": live, "doc_id_prefixes": prefixes,
            "ingest_state": ingest_rows, "milvus_rows": milvus_store.count_rows()}


# --------------------------------------------------------------------------- 父进程

def prod_snapshot() -> dict:
    import src.config
    conn = sqlite3.connect(src.config.SQLITE_DB_PATH)
    try:
        return {"docs": conn.execute("SELECT COUNT(*) FROM doc_version").fetchone()[0],
                "chunks": conn.execute("SELECT COUNT(*) FROM knowledge_chunk").fetchone()[0],
                "ingest_state": conn.execute("SELECT COUNT(*) FROM ingest_state").fetchone()[0]}
    finally:
        conn.close()


def main() -> int:
    print("=" * 62)
    print("全量重建入口评测（正式库副本上执行，正式库只读）")
    print("=" * 62)
    t0 = time.time()
    import src.config

    prod_before = prod_snapshot()
    tmp_root = Path(tempfile.mkdtemp(prefix="rebuild_eval_"))
    try:
        # 1) 沙箱底座：复制正式库双库（必须在打开正式库 Milvus 之前完成拷贝）
        sandbox = tmp_root / "sandbox"
        (sandbox / "milvus").mkdir(parents=True, exist_ok=True)
        shutil.copy2(src.config.SQLITE_DB_PATH, sandbox / "l1_core.db")
        shutil.copytree(src.config.MILVUS_DB_PATH, sandbox / "milvus" / "l1_milvus.db")

        # 2) 投放目录：两份真实制度文书
        incoming = tmp_root / "incoming"
        incoming.mkdir()
        for rel in SEED_DOCS:
            shutil.copy2(SRC_DIR / rel, incoming / Path(rel).name)

        # 3) 子进程内重建
        proc = subprocess.run([sys.executable, "-X", "utf8", str(Path(__file__).resolve()),
                               "--child", str(sandbox), str(incoming)], cwd=PROJECT_ROOT)
        check("C1 重建子进程退出码为 0", proc.returncode == 0, f"exit={proc.returncode}")
        if proc.returncode != 0:
            return report(1)
        data = json.loads((sandbox / "child.json").read_text(encoding="utf-8"))
        first, again = data["after_rebuild"], data["after_rerun"]

        # 4) 重建结果断言
        check("C2 重建返回码 0（含双库对账通过）", data["rebuild_rc"] == 0)
        check("C3 重建后主库与向量库行数一致", first["live_chunks"] == first["milvus_rows"],
              f"主库 {first['live_chunks']} vs 向量 {first['milvus_rows']}")
        check("C4 存量 Excel 数据已被整库替换（doc_id 只剩 AUTO- 前缀）",
              first["doc_id_prefixes"] == ["AUTO"], str(first["doc_id_prefixes"]))
        check("C5 投放文件全部入库且登记摄取状态",
              first["docs"] == len(SEED_DOCS) and first["ingest_state"] == len(SEED_DOCS),
              f"文档 {first['docs']}，状态行 {first['ingest_state']}")
        check("C6 FTS5 关键词索引与 valid 切片同范围", data["fts_rows"] == first["live_chunks"],
              f"FTS {data['fts_rows']} vs 切片 {first['live_chunks']}")
        check("C7 切片全部由自动通道产出（Excel 通道断开后仍可全库重建）",
              first["live_chunks"] > 0, f"{first['live_chunks']} 片")

        # 5) 幂等断言
        check("C8 无改动重跑零写库", data["rerun"] == {"added": 0, "changed": 0,
                                                       "removed": 0, "unchanged": len(SEED_DOCS)},
              str(data["rerun"]))
        check("C9 重跑后存量与对账不变", again == first, f"{again['live_chunks']}/{again['milvus_rows']}")

        # 6) 正式库零污染
        prod_after = prod_snapshot()
        check("C10 正式库全程未被触碰", prod_after == prod_before,
              f"before={prod_before} after={prod_after}")
        return report(0)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
        print(f"\n（耗时 {time.time() - t0:.0f}s）")


def report(rc: int) -> int:
    print("\n" + "=" * 62)
    if _failures:
        print(f"结果 FAIL：{len(_failures)} 项未通过 → {_failures}")
        return 1
    print("结果 PASS：全量重建入口可用，幂等且正式库零污染 ✓")
    return rc


if __name__ == "__main__":
    if "--child" in sys.argv:
        i = sys.argv.index("--child")
        phase_child(Path(sys.argv[i + 1]), Path(sys.argv[i + 2]))
        sys.exit(0)
    sys.exit(main())
