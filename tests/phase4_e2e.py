"""Phase 4 端到端演练：投放 → 自动入库 → 混合检索命中新版 → 旧版自动退出检索

在正式库的【临时副本】（l1_core.db + l1_milvus.db 目录）上执行全链路，正式库零污染；
每轮 run_ingest 走真实链路：增量检测 → 切块质检 → 版本化写库 → 索引同步（默认 P0 增量
delta：只删退出向量/只嵌新批切片）→ hybrid_retriever 双路检索断言；末尾另验全量回退
（--full-sync 链路）与增量结果等价。

演练剧本（18 项断言，摄取均固定 keywords_mode=rule 保持离线确定性）：
  R1 投放 1 份真实规则 PDF（深交所交易规则）+ 1 份合成细则 txt V1（独有条款代号
     ALPHA7391）→ 自动入库，混合检索分别命中 AUTO 版新切片；增量向量数=新增 valid 数，
     向量库行数与主库双有效行数对账一致
  R2 同路径覆盖为 V2（删 ALPHA7391、增 BRAVO8123）→ 版本递增 V2、V1 superseded、
     V1 切片 expired；向量增量只删 V1 只插 V2；检索不再召回 V1 独有内容、命中 V2
  R3 不做任何修改重跑 → 全部 unchanged，零写库、FTS/向量零操作（幂等）
  R4 删除细则投放文件 → 整版 expired，向量同步退出，检索退出，对账仍一致
  R5 投放伪扫描件（5 页空白 PDF）→ 门控隔离：doc_version=draft、切片无 valid、
     检索无召回，且隔离入库不触碰索引（增量零操作）
  R6 重投细则 V3 并强制 sync_mode=full → 全量回退通道可用，结果与增量路径等价

运行（项目根目录，需 L2/L3 服务未运行；增量模式下全程仅 1 轮全库重嵌（R6），约 2-4 分钟）：
    python -X utf8 tests/phase4_e2e.py
退出码 0=全部 PASS。
"""
import shutil
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import src.config  # 仅先导入配置模块，patch 路径后才触发 src.data.* 加载

REAL_PDF = PROJECT_ROOT / "data" / "raw" / "正式版文档" / "监管规则类" / "深圳证券交易所交易规则.pdf"
DOC_ID_DRILL = "AUTO"  # 演练文档 doc_version_id 前缀锚点（AUTO-自动化摄取演练用暂行细则:V{n}）

_failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" | {detail}" if detail else ""))
    if not cond:
        _failures.append(name)


def query_db(db_path: Path, sql: str, params: tuple = ()) -> list[tuple]:
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


def write_regulation(path: Path, token: str, article_no: str) -> None:
    """合成制度文书（中文制度通道格式）：正文含带独有代号的条款，用于检索命中/退出验证"""
    path.write_text(
        "自动化摄取演练用暂行细则\n\n"
        "第一条　本细则为验证自动摄取流水线的全链路行为而临时制定，正文条款均为演练专用内容，"
        "不构成任何真实业务规则，演练结束后随临时数据库副本一并销毁。\n"
        "第二条　投放目录中的文档经解析、切块与质量门控后自动写入知识库；未通过门控的整篇隔离，"
        "隔离切片全部进入待复核状态，检索链路对隔离内容保持完全不可见，直至人工复核放行。\n"
        f"第{article_no}条　本版本独有的验证条款，检索代号为 {token}，"
        "该条款仅存在于当前版本正文中，上一版本或下一版本均不包含此代号，"
        "用于证明新版本切片进入检索、旧版本切片退出检索的完整闭环。\n"
        "最后一条　本细则自发布之日起施行，由知识库运营团队负责解释与后续版本修订。\n",
        encoding="utf-8",
    )


def valid_db_count(db_path: Path) -> int:
    """主库「切片 valid ∧ 文档 valid」行数——向量库应有的规模"""
    return query_db(db_path,
                    "SELECT COUNT(*) FROM knowledge_chunk k JOIN doc_version d "
                    "ON k.doc_version_id = d.doc_version_id "
                    "WHERE k.status='valid' AND d.status='valid'")[0][0]


def main() -> int:
    t0 = time.time()
    tmp_root = Path(tempfile.mkdtemp(prefix="phase4_e2e_"))
    try:
        return run_drill(tmp_root, t0)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def run_drill(tmp_root: Path, t0: float) -> int:
    # ---- 环境与剧本 ----
    sqlite_path = tmp_root / "l1_core.db"
    milvus_path = tmp_root / "l1_milvus.db"
    shutil.copy2(src.config.SQLITE_DB_PATH, sqlite_path)
    shutil.copytree(src.config.MILVUS_DB_PATH, milvus_path)
    src.config.SQLITE_DB_PATH = sqlite_path   # 必须早于一切 src.data.* 导入：全局实例按副本路径构建
    src.config.MILVUS_DB_PATH = milvus_path

    from src.data.hybrid_retriever import hybrid_retriever
    from src.data.ingestion.runner import run_ingest
    from src.data.milvus_client import milvus_store

    incoming = tmp_root / "incoming" / "监管规则类"
    incoming.mkdir(parents=True)
    drill_txt = incoming / "自动化摄取演练用暂行细则.txt"
    scan_pdf = incoming / "扫描件演练.pdf"

    print("=" * 62)
    print("Phase 4 端到端演练（正式库临时副本，正式库零污染）")
    print("=" * 62)

    # ---- R1：新文档投放 → 自动入库 → 混合检索命中 ----
    print("\nR1 投放真实规则 PDF + 细则 txt V1，自动摄取入库…")
    shutil.copy2(REAL_PDF, incoming / REAL_PDF.name)
    write_regulation(drill_txt, "ALPHA7391", "九百九十七")
    stats = run_ingest(incoming, keywords_mode="rule")
    check("R1 两份文档均正常入库", stats["added"] == 2 and stats["vectors_synced"])
    v1_rows = query_db(sqlite_path,
                       "SELECT doc_version_id, status FROM doc_version WHERE doc_version_id LIKE 'AUTO-%'")
    check("R1 doc_version 写入 AUTO 版本且 valid", len(v1_rows) == 2 and all(s == "valid" for _, s in v1_rows),
          str(v1_rows))

    hits = hybrid_retriever.search("深交所收盘集合竞价 尾盘 成交价格 规则", top_k=10)
    check("R1 混合检索命中 PDF 新切片",
          any(h["doc_version_id"].startswith("AUTO-深圳证券交易所交易规则") for h in hits),
          f"top1={hits[0]['doc_version_id'] if hits else '无结果'}")
    hits = hybrid_retriever.search(f"独有验证条款 检索代号 ALPHA7391", top_k=10)
    check("R1 混合检索命中 txt V1 独有条款",
          any(h["doc_version_id"] == f"{DOC_ID_DRILL}-自动化摄取演练用暂行细则:V1" for h in hits))
    n_new_valid = sum(r["n_valid"] for r in stats["records"])
    check("R1 向量增量只嵌本批新切片（非全库重嵌）",
          stats["vector_delta"] == (0, n_new_valid) and n_new_valid > 0,
          f"vector_delta={stats['vector_delta']}，本批 valid={n_new_valid}")
    check("R1 向量库与主库双有效行数对账一致",
          milvus_store.count_rows() == valid_db_count(sqlite_path))
    kw_missing = query_db(sqlite_path,
                          "SELECT COUNT(*) FROM knowledge_chunk WHERE status = 'valid' "
                          "AND doc_version_id LIKE 'AUTO-%' AND (keywords IS NULL OR keywords = '')")
    check("R1 自动入库 valid 切片 keywords 全部非空（规则路抽取接线）", kw_missing[0][0] == 0,
          f"缺 keywords 的 valid 切片 {kw_missing[0][0]} 片")

    # ---- R2：同路径覆盖投放 → 版本递增 + 旧版 superseded 退出检索 ----
    print("\nR2 覆盖为 V2（删 ALPHA7391 / 增 BRAVO8123），重新摄取…")
    write_regulation(drill_txt, "BRAVO8123", "九百九十八")
    stats = run_ingest(incoming, keywords_mode="rule")
    vid1 = f"{DOC_ID_DRILL}-自动化摄取演练用暂行细则:V1"
    vid2 = f"{DOC_ID_DRILL}-自动化摄取演练用暂行细则:V2"
    check("R2 变更被增量检测捕获且未触碰未变 PDF", stats["changed"] == 1 and stats["unchanged"] == 1)
    st1 = query_db(sqlite_path, "SELECT status FROM doc_version WHERE doc_version_id=?", (vid1,))
    st2 = query_db(sqlite_path, "SELECT status FROM doc_version WHERE doc_version_id=?", (vid2,))
    n_live_v1 = query_db(sqlite_path,
                         "SELECT COUNT(*) FROM knowledge_chunk WHERE doc_version_id=? AND status='valid'", (vid1,))[0][0]
    check("R2 V1 superseded + V2 valid", st1 and st1[0][0] == "superseded" and st2 and st2[0][0] == "valid",
          f"V1={st1[0][0] if st1 else '?'}, V2={st2[0][0] if st2 else '?'}")
    check("R2 V1 切片全部退出 valid", n_live_v1 == 0, f"V1 残余 valid 切片 {n_live_v1} 片")

    hits = hybrid_retriever.search("独有验证条款 检索代号 ALPHA7391 仅存于第一版", top_k=10)
    # 内容级断言：V2 与 V1 正文句式高度相似，被召回属预期（孪生文本）；旧版退出的本质
    # 是「ALPHA7391 这段独有内容在全部召回切片中消失」且 V1 切片不再返回
    check("R2 旧版独有内容退出检索（ALPHA7391 不再出现在任何召回切片）",
          not any(h["doc_version_id"] == vid1 for h in hits)
          and not any("ALPHA7391" in (h["original_text"] or "") for h in hits),
          str([h["doc_version_id"] for h in hits][:3]))
    hits = hybrid_retriever.search("独有验证条款 检索代号 BRAVO8123", top_k=10)
    check("R2 新版独有内容进入检索（命中 V2）", any(h["doc_version_id"] == vid2 for h in hits))
    check("R2 向量增量精确到版本：只删 V1 只插 V2",
          stats["vector_delta"] == (3, 3), f"vector_delta={stats['vector_delta']}")
    check("R2 升版后对账仍一致", milvus_store.count_rows() == valid_db_count(sqlite_path))

    # ---- R3：无改动重跑 → 幂等零副作用 ----
    print("\nR3 无任何改动重跑，验证增量检测幂等…")
    stats = run_ingest(incoming, keywords_mode="rule")
    check("R3 全部 unchanged，零写库零同步",
          stats["added"] == stats["changed"] == stats["removed"] == 0
          and stats["unchanged"] == 2 and stats["fts_rebuilt"] is None
          and stats["fts_delta"] == (0, 0) and stats["vector_delta"] == (0, 0)
          and not stats["vectors_synced"])

    # ---- R4：撤出投放 → 整版 expired ----
    print("\nR4 删除细则投放文件（文档下线），重新扫描…")
    drill_txt.unlink()
    stats = run_ingest(incoming, keywords_mode="rule")
    check("R4 removed 检出且版本整版 expired",
          stats["removed"] == 1
          and query_db(sqlite_path, "SELECT status FROM doc_version WHERE doc_version_id=?", (vid2,))[0][0] == "expired")
    hits = hybrid_retriever.search("独有验证条款 检索代号 BRAVO8123", top_k=10)
    check("R4 撤回文档退出检索（BRAVO8123 不再出现在任何召回切片）",
          not any(h["doc_version_id"] == vid2 for h in hits)
          and not any("BRAVO8123" in (h["original_text"] or "") for h in hits))
    check("R4 撤回后向量同步退出，对账仍一致",
          stats["vector_delta"][0] == 3 and milvus_store.count_rows() == valid_db_count(sqlite_path))

    # ---- R5：扫描件投放 → 门控隔离，不惊动向量库 ----
    print("\nR5 投放 5 页空白伪扫描 PDF，验证隔离入库…")
    from pypdf import PdfWriter
    writer = PdfWriter()
    for _ in range(5):
        writer.add_blank_page(width=595, height=842)
    with scan_pdf.open("wb") as f:
        writer.write(f)
    stats = run_ingest(incoming, keywords_mode="rule")
    rec = next(r for r in stats["records"] if r["path"].endswith("扫描件演练.pdf"))
    check("R5 扫描件走隔离通道（quarantined 而非 ingested）", rec["outcome"] == "quarantined",
          str(rec["issues"][:1]))
    st = query_db(sqlite_path, "SELECT status FROM doc_version WHERE doc_version_id=?", (rec["doc_version_id"],))
    n_valid = query_db(sqlite_path,
                       "SELECT COUNT(*) FROM knowledge_chunk WHERE doc_version_id=? AND status='valid'",
                       (rec["doc_version_id"],))[0][0]
    check("R5 隔离版 doc_version=draft 且无 valid 切片", st and st[0][0] == "draft" and n_valid == 0)
    check("R5 纯隔离入库不触碰任何索引（FTS/向量增量零操作）",
          stats["fts_delta"] == (0, 0) and stats["vector_delta"] == (0, 0) and not stats["vectors_synced"])

    # ---- R6：全量回退通道（sync_mode=full）与增量结果等价 ----
    print("\nR6 重投细则 V3 并强制全量同步，验证回退通道可用且与增量等价…")
    write_regulation(drill_txt, "CHARLIE5544", "九百九十九")
    stats = run_ingest(incoming, sync_mode="full", keywords_mode="rule")
    vid3 = f"{DOC_ID_DRILL}-自动化摄取演练用暂行细则:V3"
    check("R6 全量模式：V3 入库（重新投放=新增）+ FTS 全重建 + 向量全量同步",
          stats["added"] == 1 and stats["fts_rebuilt"] is not None and stats["vectors_synced"])
    hits = hybrid_retriever.search("独有验证条款 检索代号 CHARLIE5544", top_k=10)
    check("R6 全量同步后检索命中 V3", any(h["doc_version_id"] == vid3 for h in hits))
    check("R6 全量重建后对账与增量路径终态一致",
          milvus_store.count_rows() == valid_db_count(sqlite_path))

    print("\n" + "=" * 62)
    print(f"演练结束（{time.time() - t0:.0f}s）：" + ("全部 PASS ✓" if not _failures else f"FAIL {len(_failures)} 项：{_failures}"))
    print("=" * 62)
    return 0 if not _failures else 1


if __name__ == "__main__":
    sys.exit(main())
