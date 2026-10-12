"""正式摄取流水线（Phase 4）：增量检测 → 切块质检 → 版本化写库 → FTS/向量同步

入口 run_ingest()：扫描投放目录，与 SQLite ingest_state 表比对指纹——
    added     → 新文档，doc_id 首版 V1 入库
    changed   → 同 doc_id 版本递增 V{n+1}，前版 doc_version 置 superseded、切片置 expired
    removed   → 撤回：对应版本整版置 expired（投放目录即待入库清单，删文件=下线信号）
    unchanged → 零写库零同步（幂等重跑安全）
整篇隔离（质量门控检出）不豁免写库：doc_version=draft + 全片 need_review 入库留痕，
检索两路（切片 valid ∧ 文档 valid）天然不召回，人工复核改 status 后随下次同步放量。
索引同步默认走增量 delta（P0）：FTS 按退出/新增 id 维护、Milvus 只删旧版向量只嵌新批
切片，成本 O(本批) 而非 O(全库)；--full-sync 保留全量重建作为漂移修复回退。
"""
import json
from pathlib import Path

import pandas as pd

from src.data.ingestion import chunking, keywords, loaders, quality, readers, structuring, versioning
from src.data.ingestion.config import INCOMING_DIR
from src.data.ingestion.schemas import FileSnapshot, RawDocument
from src.data.sqlite_client import sqlite_client


def _ingest_snapshot(snap: FileSnapshot, prev: dict | None, kw_mode: str | None = None) -> dict:
    """单文件摄取：解析→门控→切块→质检→关键词抽取→版本化写库→登记 ingest_state，返回处理记录"""
    path = snap.path
    rec: dict = {"path": str(path), "outcome": "failed", "doc_version_id": None,
                 "n_chunks": 0, "n_valid": 0, "n_review": 0, "issues": [], "prev": prev}
    try:
        doc: RawDocument = readers.read_document(path)
        doc_issues = quality.check_document(doc)
        sections = structuring.split_sections(doc)
        doc_issues += quality.check_chapters(doc, sections)
        drafts = quality.annotate(chunking.chunk_document(doc, sections), doc_issues)
    except Exception as e:  # 解析/切块异常：登记 failed 指纹，坏文件不反复冲击流水线
        rec["issues"] = [f"解析异常：{e!r}"]
        _register(snap, rec)
        return rec

    doc_id = versioning.derive_doc_id(path)
    version_n = sqlite_client.next_version(doc_id)
    vid = versioning.doc_version_id(doc_id, version_n)
    quarantined = bool(doc_issues)
    # 关键词抽取仅对将入索引的文档做（整篇隔离全片 need_review 不进检索，省掉无收益的 LLM 调用）
    kw_note = keywords.assign_keywords(drafts, kw_mode) if not quarantined else None
    rows = loaders.to_db_rows(drafts, vid)
    status = "draft" if quarantined else "valid"
    doc_row = loaders.build_doc_version_row(doc, vid, doc_id, version_n, status)
    supersede_vid = (prev or {}).get("doc_version_id")  # 前版（含前次隔离版）整版退出
    retired = sqlite_client.write_new_version(doc_row, rows, supersede_vid=supersede_vid)

    rec.update({
        "outcome": "quarantined" if quarantined else "ingested",
        "doc_version_id": vid,
        "n_chunks": len(rows),
        "n_valid": sum(1 for r in rows if r["status"] == "valid"),
        "n_review": sum(1 for r in rows if r["status"] == "need_review"),
        "issues": doc_issues,
        "superseded": supersede_vid,
        "retired": retired,  # 旧版退出的 valid 切片 id：FTS/向量增量删除的对象
        "keywords": kw_note,
    })
    _register(snap, rec)
    return rec


def _register(snap: FileSnapshot, rec: dict) -> None:
    """ingest_state 落指纹：无论成败都登记，unchanged 判定以内容为准"""
    detail = {k: rec[k] for k in ("doc_version_id", "n_chunks", "n_valid", "n_review")}
    detail["issues"] = rec["issues"][:5]
    sqlite_client.upsert_ingest_state(
        path=rec["path"], sha256=snap.sha256, mtime=snap.mtime,
        size=snap.path.stat().st_size if snap.path.exists() else 0,
        doc_version_id=rec["doc_version_id"], outcome=rec["outcome"],
        detail=json.dumps(detail, ensure_ascii=False),
    )


def _retire(path_str: str, prev: dict) -> dict:
    """removed：文件撤出投放目录 → 已入库版本整版 expired，指纹登记同步删除"""
    vid = prev.get("doc_version_id")
    retired = sqlite_client.expire_document(vid) if vid else []
    sqlite_client.delete_ingest_state(path_str)
    return {"path": path_str, "outcome": "expired", "doc_version_id": vid,
            "prev_outcome": prev.get("outcome"), "retired": retired}


def _touches_valid(rec: dict) -> bool:
    """本条处理是否改变 valid 切片集合（决定要不要重嵌 Milvus 向量）：
    正常入库必有增量；隔离新入库（draft）不动 valid 集合；supersede/expire 仅在
    前版是正常入库（含 valid 切片）时才有退出。"""
    if rec["outcome"] == "ingested":
        return True
    if rec["outcome"] == "expired":
        return rec.get("prev_outcome") == "ingested"
    if rec["outcome"] == "quarantined":
        return bool(rec.get("superseded")) and (rec.get("prev") or {}).get("outcome") == "ingested"
    return False


def run_ingest(directory: Path = INCOMING_DIR, sync_vectors: bool = True, verbose: bool = True,
               sync_mode: str = "delta", keywords_mode: str | None = None) -> dict:
    """正式摄取主入口。返回统计：added/changed/removed/unchanged 计数、逐文件处理记录、
    索引同步结果（fts_delta/vector_delta 或全量模式的 fts_rebuilt/vectors_synced）。
    sync_mode：delta（默认，P0 增量：只动本批退出/新增，成本 O(本批)）/ full（全库重建，
    漂移修复回退）；sync_vectors=False 交给调用方接管向量同步时机；
    keywords_mode：llm/rule/off 覆盖环境变量缺省（关键词自动抽取路）。"""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    snaps = versioning.snapshot_dir(directory)
    state = sqlite_client.load_ingest_state()
    diff = versioning.diff_state(snaps, {k: v["sha256"] for k, v in state.items()})

    records: list[dict] = []
    added_paths = {str(s.path) for s in diff["added"]}
    for snap in diff["added"] + diff["changed"]:
        rec = _ingest_snapshot(snap, state.get(str(snap.path)), kw_mode=keywords_mode)
        records.append(rec)
        if verbose:
            tag = {"ingested": "入库", "quarantined": "⚠ 整篇隔离（draft+need_review，转人工）",
                   "failed": "✗ 解析失败（登记指纹，修复后重投放）"}[rec["outcome"]]
            extra = (f"{rec['n_chunks']} 片（valid {rec['n_valid']} / need_review {rec['n_review']}）"
                     f" → {rec['doc_version_id']}" if rec["outcome"] != "failed" else "")
            if rec.get("keywords"):
                extra += f" | keywords: {rec['keywords']}"
            sup = f" | 前版退出: {rec.get('superseded')}" if rec.get("superseded") else ""
            kind = "新增" if rec["path"] in added_paths else "变更"
            print(f"[{kind}] {Path(rec['path']).name}: {tag} {extra}{sup}")
            for issue in rec["issues"]:
                print(f"    [门控] {issue}")
    for path_str in diff["removed"]:
        rec = _retire(path_str, state[path_str])
        records.append(rec)
        if verbose:
            print(f"[撤回] {Path(path_str).name}: 版本整版 expired → {rec['doc_version_id']}")

    wrote = any(r["outcome"] in ("ingested", "quarantined", "expired") for r in records)
    fts_delta = vector_delta = (0, 0)
    fts_rebuilt = None
    vectors_synced = False
    # 本批 delta 素材：退出 valid 的旧切片 id（FTS+向量都删）与新入库版本的 valid 行（都插）
    delete_ids = list(dict.fromkeys(kid for r in records for kid in r.get("retired") or []))
    new_vids = [r["doc_version_id"] for r in records if r["outcome"] == "ingested"]
    insert_df = (sqlite_client.get_valid_knowledge_for_vectors(new_vids)
                 if (new_vids or delete_ids) else pd.DataFrame())

    if wrote and sync_mode == "full":
        # 全量回退：FTS 全重建 + Milvus 全删重嵌（漂移修复/首次建库场景，O(全库)）
        fts_rebuilt = sqlite_client.rebuild_fts()
        if any(_touches_valid(r) for r in records) and sync_vectors:
            # 延迟导入：embedding 模型与 Milvus 锁仅正式同步时加载，--check / 纯隔离场景零副作用
            from src.data.milvus_client import milvus_store
            milvus_store.sync_from_sqlite()
            vectors_synced = True
    elif wrote and sync_mode == "delta":
        if delete_ids or not insert_df.empty:
            fts_delta = sqlite_client.fts_apply_delta(delete_ids, insert_df.to_dict("records"))
            if sync_vectors:
                from src.data.milvus_client import milvus_store
                vector_delta = milvus_store.apply_delta(delete_ids, insert_df)
                vectors_synced = True
    if verbose:
        if sync_mode == "full":
            tail = f"FTS 全量重建 {fts_rebuilt} 行，向量库{'已全量同步' if vectors_synced else '无需同步'}"
        else:
            tail = (f"FTS 增量 -{fts_delta[0]} +{fts_delta[1]}，"
                    f"向量库增量 -{vector_delta[0]} +{vector_delta[1]}" if vectors_synced or fts_delta != (0, 0)
                    else "索引零操作（无 valid 集合变化）")
        print(f"\n完成：新增 {len(diff['added'])}，变更 {len(diff['changed'])}，撤回 {len(diff['removed'])}，"
              f"未变 {len(diff['unchanged'])}；{tail}")
    return {
        "added": len(diff["added"]), "changed": len(diff["changed"]),
        "removed": len(diff["removed"]), "unchanged": len(diff["unchanged"]),
        "records": records, "fts_rebuilt": fts_rebuilt, "vectors_synced": vectors_synced,
        "fts_delta": fts_delta, "vector_delta": vector_delta,
    }
