"""生效期字段 NULL 兜底评测：向量同步不再崩 + 「长期有效」语义双路口径一致

背景：表结构与 README 均允许 effective_date / expire_date 为 NULL（expire_date 为空=长期有效），
SQL（FTS）路写了 `IS NULL OR ...` 兜底，但向量路的 effective_ts / expire_ts 必须是可比较整数，
原实现对 None 直接调 .timestamp() 会抛 AttributeError——表现即「全量构建在生成向量那一步崩掉，
而 SQLite 已经写好了数据」。本脚本在隔离沙箱库上验证兜底后的语义：空值不崩、NULL 生效期切片
双路均可召回、明确过期的切片不被兜底放宽、主库清空后向量集合不留脏索引。

运行（项目根目录，需 L2/L3 服务未运行以释放 MilvusLite 锁；纯本地计算，无 LLM 调用）：
    python -X utf8 tests/date_boundary_eval.py
退出码 0=全部 PASS。全程只读写临时目录，正式库零接触。
"""
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

_failures: list[str] = []

DOC_SQL = """
CREATE TABLE doc_version (
    doc_version_id  TEXT PRIMARY KEY, doc_id TEXT NOT NULL, doc_name TEXT NOT NULL,
    version TEXT NOT NULL, doc_type TEXT, publish_dept TEXT, publish_date TEXT,
    effective_date TEXT, expire_date TEXT, status TEXT NOT NULL DEFAULT 'valid',
    file_path TEXT, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, UNIQUE (doc_id, version))
"""
CHUNK_SQL = """
CREATE TABLE knowledge_chunk (
    knowledge_id TEXT PRIMARY KEY, doc_version_id TEXT NOT NULL, clause_position TEXT,
    heading_path TEXT, chunk_type TEXT NOT NULL DEFAULT 'clause', content TEXT NOT NULL,
    original_text TEXT NOT NULL, keywords TEXT, status TEXT NOT NULL DEFAULT 'valid',
    FOREIGN KEY (doc_version_id) REFERENCES doc_version(doc_version_id))
"""

# 四份文档覆盖四种生效期组合，主题词各异以便检索侧区分
# (knowledge_id, doc_id, effective_date, expire_date, 主题词, 期望是否可召回)
CASES = [
    ("DB-0001", "NULL-EXP", "2020-01-01", None, "涨跌幅限制", True),        # 长期有效：expire 为空
    ("DB-0002", "NULL-EFF", None, "2099-12-31", "融资融券", True),          # 立即生效：effective 为空
    ("DB-0003", "BOTH-NULL", None, None, "投资者适当性", True),             # 两列皆空
    ("DB-0004", "EXPIRED", "2019-01-01", "2020-01-01", "申购赎回费用", False),  # 明确过期，不得被放宽
]


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" | {detail}" if detail else ""))
    if not cond:
        _failures.append(name)


def build_sandbox(tmp_root: Path) -> Path:
    """临时目录内建一份只含本次用例的沙箱主库（纯 sqlite3 操作，不触发任何 src.data 导入）"""
    db_path = tmp_root / "date_boundary.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(DOC_SQL)
    conn.executescript(CHUNK_SQL)
    for i, (kid, doc_id, eff, exp, topic, _) in enumerate(CASES, start=1):
        vid = f"{doc_id}:V1"
        conn.execute(
            "INSERT INTO doc_version (doc_version_id, doc_id, doc_name, version, doc_type,"
            " effective_date, expire_date, status) VALUES (?,?,?,?,?,?,?,?)",
            (vid, doc_id, f"生效期测试文档{i}", "V1", "监管规则", eff, exp, "valid"),
        )
        conn.execute(
            "INSERT INTO knowledge_chunk (knowledge_id, doc_version_id, clause_position,"
            " heading_path, chunk_type, content, original_text, status) VALUES (?,?,?,?,?,?,?,?)",
            (kid, vid, f"1.{i}", f"第一章 > 1.{i}", "clause",
             f"关于{topic}的规定：本所对{topic}实行专门管理，具体比例与时限见正文。",
             f"第 1.{i} 条 关于{topic}的规定，{topic}相关安排自发布之日起施行。", "valid"),
        )
    conn.commit()
    conn.close()
    return db_path


def eval_unit() -> None:
    print("\nA _date_str_to_ts 纯函数（空值归到时间轴端点）")
    from src.data.milvus_client import OPEN_END_DATE, _date_str_to_ts

    end_ts = int(pd.to_datetime(OPEN_END_DATE).timestamp())
    rows = [
        ("A1 None + empty_as=end → 远期上界", _date_str_to_ts(None, empty_as="end"), end_ts),
        ("A2 None + 默认 → 0（立即生效）", _date_str_to_ts(None), 0),
        ("A3 float NaN 按空值处理", _date_str_to_ts(float("nan"), empty_as="end"), end_ts),
        ("A4 pandas NaT 按空值处理", _date_str_to_ts(pd.NaT, empty_as="end"), end_ts),
        ("A5 空串/纯空白按空值处理", _date_str_to_ts("   ", empty_as="end"), end_ts),
        ("A6 正常日期不受兜底影响", _date_str_to_ts("2020-01-01", empty_as="end"),
         int(pd.to_datetime("2020-01-01").timestamp())),
    ]
    for name, got, want in rows:
        check(name, got == want, f"{got} vs {want}")
    check("A7 兜底上界取有限日期（不用 int64 上界，留加法余量）",
          0 < _date_str_to_ts(None, empty_as="end") < 2 ** 62, f"OPEN_END_DATE={OPEN_END_DATE}")
    check("A8 返回值恒为 int（Milvus 表达式不接受 float/NaN）",
          all(isinstance(_date_str_to_ts(v, empty_as=e), int)
              for v in (None, float("nan"), pd.NaT, "2026-01-18") for e in ("start", "end")))


def eval_sync_and_recall(db_path: Path) -> None:
    print("\nB 向量同步与双路召回（沙箱库，正式库零接触）")
    from src.data.hybrid_retriever import hybrid_retriever
    from src.data.milvus_client import milvus_store
    from src.data.sqlite_client import sqlite_client

    try:
        milvus_store.sync_from_sqlite()
        crashed = None
    except Exception as e:  # noqa: BLE001 - 断言点：兜底缺失时这里抛 AttributeError
        crashed = f"{type(e).__name__}: {e}"
    check("B1 含 NULL 生效期的切片做向量同步不再崩溃", crashed is None, crashed or "无异常")
    check("B2 四份文档的 valid 切片全部入库", milvus_store.count_rows() == len(CASES),
          f"向量 {milvus_store.count_rows()} 条")

    for kid, doc_id, _, _, topic, recallable in CASES:
        label = "应命中" if recallable else "应被生效期过滤"
        kids = {h["knowledge_id"] for h in milvus_store.search(f"{topic}的规定是什么", top_k=10)}
        check(f"B3.{doc_id} 向量路 {label}", (kid in kids) == recallable, f"{kid} 命中={kid in kids}")
        kids = {h["knowledge_id"] for h in sqlite_client.keyword_search(f"{topic}的规定是什么", top_k=10)}
        check(f"B4.{doc_id} 关键词路与向量路同结论", (kid in kids) == recallable,
              f"{kid} 命中={kid in kids}")

    # 生产检索链路端到端（混合路，Rerank 关闭态与线上一致）
    top = {h["knowledge_id"] for h in hybrid_retriever.search("投资者适当性的规定是什么", top_k=5)}
    check("B5 混合检索可直接召回 expire_date 为 NULL 的切片", "DB-0003" in top, f"结果={sorted(top)}")

    # 主库清空后：全量同步须连带清空向量集合，否则残留向量参与召回却回查不到详情
    conn = sqlite3.connect(db_path)
    conn.execute("DELETE FROM knowledge_chunk")
    conn.execute("DELETE FROM doc_version")
    conn.commit()
    conn.close()
    milvus_store.sync_from_sqlite()
    check("B6 主库清空后向量集合同步清空（无脏索引）", milvus_store.count_rows() == 0,
          f"残留 {milvus_store.count_rows()} 条")


def main() -> int:
    print("=" * 62)
    print("生效期字段 NULL 兜底评测（向量同步 + 双路召回口径一致性）")
    print("=" * 62)
    tmp_root = Path(tempfile.mkdtemp(prefix="date_boundary_"))
    try:
        # 关键顺序：先建沙箱库并改写全局路径配置，再首次导入 src.data.*
        # （sqlite_client / milvus_store 是导入时构造的全局单例，按导入时的路径绑定，后补无效）
        import src.config
        db_path = build_sandbox(tmp_root)
        milvus_path = tmp_root / "milvus" / "date_boundary_milvus.db"
        milvus_path.parent.mkdir(parents=True, exist_ok=True)  # MilvusLite 不自动建父目录
        src.config.SQLITE_DB_PATH = db_path
        src.config.MILVUS_DB_PATH = milvus_path
        src.config.RAW_DATA_DIR = tmp_root / "raw"
        src.config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

        eval_unit()
        eval_sync_and_recall(db_path)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    print("\n" + "=" * 62)
    if _failures:
        print(f"结果 FAIL：{len(_failures)} 项未通过 → {_failures}")
        return 1
    print("结果 PASS：NULL 生效期语义在向量路与关键词路口径一致 ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
