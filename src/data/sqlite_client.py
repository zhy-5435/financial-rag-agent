"""SQLite 权威主库：建表 / 值域约束 / 版本化写库 / 原始文档在位对账 / 切片详情查询 / FTS5 关键词检索

数据来源已统一为自动摄取通道（src/data/ingestion）：本模块不再内置任何切片构造逻辑，
只做「行字典 → 表」的落库与检索，切片由摄取层产出后经 write_new_version 写入。
"""
import json
import re
import sqlite3
from pathlib import Path

import pandas as pd

from src.config import RAW_DATA_DIR, SQLITE_DB_PATH

# FTS5 关键词检索虚拟表（knowledge_id 不参与全文匹配，仅作关联主表外键）
FTS_TABLE = "fts_l1_knowledge"
# 检索用分词：中文按单字切分，英文/数字（含小数点、百分号，如 2.4.2 / 20%）连续串作整词
_TOKEN_PATTERN = re.compile(r"[0-9A-Za-z._%]+|[\u4e00-\u9fff]")


def tokenize_for_fts(text: str) -> list[str]:
    """FTS5 检索用分词：中文单字 + 英文数字整词，建索引与查询两侧共用同一套规则"""
    return _TOKEN_PATTERN.findall(text or "")


def parse_keywords(raw) -> list:
    """keywords 在 SQLite 中为 JSON 字符串，解析为列表便于上层直接使用"""
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        value = json.loads(raw)
        return value if isinstance(value, list) else []
    except json.JSONDecodeError:
        return []

# 与《表结构设计》建表语句保持一致的列顺序（摄取写库、SQL 查询均按此对齐）
DOC_COLUMNS = [
    "doc_version_id", "doc_id", "doc_name", "version", "doc_type",
    "publish_dept", "publish_date", "effective_date", "expire_date",
    "status", "file_path", "created_at", "updated_at",
]
CHUNK_COLUMNS = [
    "knowledge_id", "doc_version_id", "clause_position", "heading_path",
    "chunk_type", "content", "original_text", "keywords", "status",
]


def normalize_filename(name: str) -> str:
    """文件名规范化：去除全部空白字符、统一各类中点为半角中点，用于路径模糊匹配"""
    name = "".join(str(name).split())
    for dot in ("\u30fb", "\u2027", "\u2022"):  # 片假名中点/音节分隔符/圆点
        name = name.replace(dot, "\u00b7")
    return name


class SqliteClient:
    # 与表结构 CHECK 约束一致的合法值域（DDL 为唯一权威，此处为镜像常量，
    # 供摄取层写库前自检与人工改库前校验参考；表约束已能在写入时拦截非法值）
    DOC_STATUS_VALUES = ("draft", "valid", "superseded", "expired", "archived")
    CHUNK_STATUS_VALUES = ("valid", "expired", "deleted", "need_review")
    CHUNK_TYPE_VALUES = (
        "clause", "table", "exception", "principle", "definition",
        "product_element", "fee", "risk", "process", "other",
    )

    def __init__(self):
        self.db_path = str(SQLITE_DB_PATH)
        # FTS5 不可用时关键词检索路自动退化，上层退回纯向量检索
        self.fts_available = True
        self._init_tables()
        self._ensure_fts_index()

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")  # SQLite 外键约束需在每个连接上单独开启
        return conn

    def _init_tables(self):
        """按《表结构设计》初始化 doc_version / knowledge_chunk，幂等操作"""
        conn = self._get_conn()
        cursor = conn.cursor()

        # 清理旧版结构的表（先子表后父表）；本项目以投放文档为唯一数据源全量重建，旧结构数据不再保留
        cursor.execute("DROP TABLE IF EXISTS knowledge_point")
        cursor.execute("DROP TABLE IF EXISTS document_meta")

        # 表1：文档版本表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS doc_version (
            doc_version_id  TEXT PRIMARY KEY,           -- 如 REG-001:V2026
            doc_id          TEXT NOT NULL,              -- 如 REG-001
            doc_name        TEXT NOT NULL,
            version         TEXT NOT NULL,              -- 如 V2026
            doc_type        TEXT,                       -- 监管规则 / 内部制度 / 产品说明书
            publish_dept    TEXT,
            publish_date    TEXT,                       -- YYYY-MM-DD
            effective_date  TEXT,                       -- YYYY-MM-DD
            expire_date     TEXT,                       -- YYYY-MM-DD，NULL 表示长期有效
            status          TEXT NOT NULL DEFAULT 'valid'
                CHECK (status IN ('draft','valid','superseded','expired','archived')),
            file_path       TEXT,
            created_at      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (doc_id, version)
        )
        ''')

        # 表2：知识切片表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS knowledge_chunk (
            knowledge_id    TEXT PRIMARY KEY,           -- 如 L1-0001
            doc_version_id  TEXT NOT NULL,              -- 关联 doc_version.doc_version_id
            clause_position TEXT,                       -- 如 2.4.2，产品说明书可为 NULL
            heading_path    TEXT,                       -- 如 第二章 交易时间 > 2.4 交易时间 > 2.4.2
            chunk_type      TEXT NOT NULL DEFAULT 'clause'
                CHECK (chunk_type IN (
                    'clause','table','exception','principle','definition',
                    'product_element','fee','risk','process','other'
                )),
            content         TEXT NOT NULL,              -- 检索用文本
            original_text   TEXT NOT NULL,              -- 原文逐字摘录
            keywords        TEXT,                       -- JSON 字符串，如 ["科创板","20%"]
            status          TEXT NOT NULL DEFAULT 'valid'
                CHECK (status IN ('valid','expired','deleted','need_review')),
            FOREIGN KEY (doc_version_id) REFERENCES doc_version(doc_version_id)
        )
        ''')

        # 索引
        cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_doc_version_status
        ON doc_version(status, effective_date, expire_date)
        ''')
        cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_chunk_doc_version
        ON knowledge_chunk(doc_version_id, status)
        ''')
        cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_chunk_type
        ON knowledge_chunk(chunk_type)
        ''')

        # 表3：自动摄取状态表（增量检测）：投放文件指纹 → 已入库版本，
        # 重跑时 sha256 未变即跳过；本表是全库唯一的入库分界依据，reset_knowledge_base 必须连它一起清
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS ingest_state (
            file_path       TEXT PRIMARY KEY,   -- 投放文件绝对路径
            sha256          TEXT NOT NULL,      -- 内容指纹，变更检测依据
            mtime           REAL NOT NULL,
            size            INTEGER NOT NULL,
            doc_version_id  TEXT,               -- 本文件最近摄取产生的版本（quarantined/failed 也登记备查）
            outcome         TEXT NOT NULL       -- ingested=正常入库 / quarantined=整篇隔离 / failed=解析失败
                CHECK (outcome IN ('ingested','quarantined','failed')),
            detail          TEXT,               -- JSON：切片统计与门控 issues 样例，人工复核入口
            ingested_at     TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        # FTS5 关键词索引：仅索引 status=valid 切片（与向量路同步范围一致），随导入全量重建
        try:
            cursor.execute(f'''
            CREATE VIRTUAL TABLE IF NOT EXISTS {FTS_TABLE} USING fts5(
                knowledge_id UNINDEXED,
                body
            )
            ''')
        except sqlite3.OperationalError as e:
            self.fts_available = False
            print(f"[警告] 当前 SQLite 不支持 FTS5，关键词检索路不可用，将退化为纯向量检索：{e}")

        conn.commit()
        conn.close()

    # ---------- FTS5 关键词索引 ----------

    @staticmethod
    def _fts_body(content: str, keywords) -> str:
        """构造 FTS5 索引正文：content + keywords 分词后空格连接（keywords 的 JSON 语法字符被分词自然过滤）"""
        parts = [content]
        if isinstance(keywords, str) and keywords.strip():
            parts.append(keywords)
        return " ".join(tokenize_for_fts(" ".join(parts)))

    def _rebuild_fts_index(self, conn):
        """全量重建 FTS5 关键词索引，仅索引 status=valid 切片，与向量路同步范围一致"""
        cursor = conn.cursor()
        cursor.execute(f"DELETE FROM {FTS_TABLE}")
        rows = cursor.execute(
            "SELECT knowledge_id, content, keywords FROM knowledge_chunk WHERE status = 'valid'"
        ).fetchall()
        cursor.executemany(
            f"INSERT INTO {FTS_TABLE} (knowledge_id, body) VALUES (?, ?)",
            [(kid, self._fts_body(content, kw)) for kid, content, kw in rows],
        )
        return len(rows)

    def _ensure_fts_index(self):
        """FTS5 行数与主库有效切片行数对账，不一致（如旧库升级/索引缺失）时自动补建"""
        if not self.fts_available:
            return
        conn = self._get_conn()
        try:
            n_fts = conn.execute(f"SELECT COUNT(*) FROM {FTS_TABLE}").fetchone()[0]
            n_valid = conn.execute(
                "SELECT COUNT(*) FROM knowledge_chunk WHERE status = 'valid'"
            ).fetchone()[0]
            if n_fts != n_valid:
                self._rebuild_fts_index(conn)
                conn.commit()
                print(f"FTS5 关键词索引已自动重建，共索引 {n_valid} 条有效切片")
        finally:
            conn.close()

    # ---------- 全量重建与查询 ----------

    def reset_knowledge_base(self) -> tuple[int, int]:
        """清空知识库三表与关键词索引，返回清理前的 (文档数, 切片数)，供全量重建入口调用。

        必须连 ingest_state 一起清：该表按文件绝对路径记指纹，只清业务表而保留指纹状态时，
        重跑摄取会把所有投放文件判成 unchanged（内容未变）而零写库，表现为「数据删了但导不回来」。
        向量集合由调用方另行 sync_from_sqlite 处理（主库为空时会连带清空集合）；
        会话记忆库 l1_memory.db 不在范围内（审计存档永不删除约定）。
        """
        conn = self._get_conn()
        try:
            n_doc = conn.execute("SELECT COUNT(*) FROM doc_version").fetchone()[0]
            n_chunk = conn.execute("SELECT COUNT(*) FROM knowledge_chunk").fetchone()[0]
            conn.execute("DELETE FROM knowledge_chunk")
            conn.execute("DELETE FROM doc_version")
            conn.execute("DELETE FROM ingest_state")
            if self.fts_available:
                conn.execute(f"DELETE FROM {FTS_TABLE}")
            conn.commit()
            return n_doc, n_chunk
        finally:
            conn.close()

    # ---------- 自动摄取通道（Phase 4：增量检测 / 版本递增 / superseded） ----------

    def load_ingest_state(self) -> dict[str, dict]:
        """读全部摄取状态，{文件路径: 行字典}，供 versioning 增量比对"""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT file_path, sha256, mtime, size, doc_version_id, outcome, detail, ingested_at "
            "FROM ingest_state"
        ).fetchall()
        conn.close()
        cols = ["file_path", "sha256", "mtime", "size", "doc_version_id", "outcome", "detail", "ingested_at"]
        return {r[0]: dict(zip(cols, r)) for r in rows}

    def upsert_ingest_state(self, path: str, sha256: str, mtime: float, size: int,
                            doc_version_id: str | None, outcome: str, detail: str | None) -> None:
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO ingest_state "
            "(file_path, sha256, mtime, size, doc_version_id, outcome, detail, ingested_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now','localtime'))",
            (path, sha256, mtime, size, doc_version_id, outcome, detail),
        )
        conn.commit()
        conn.close()

    def delete_ingest_state(self, path: str) -> None:
        conn = self._get_conn()
        conn.execute("DELETE FROM ingest_state WHERE file_path = ?", (path,))
        conn.commit()
        conn.close()

    def next_version(self, doc_id: str) -> int:
        """同一文档实体的下一版本号：取现有 version 中数字最大值 + 1（无历史则 V1）"""
        conn = self._get_conn()
        rows = conn.execute("SELECT version FROM doc_version WHERE doc_id = ?", (doc_id,)).fetchall()
        conn.close()
        nums = [int(m.group(1)) for (v,) in rows if (m := re.search(r"(\d+)", v or ""))]
        return (max(nums) + 1) if nums else 1

    @staticmethod
    def _apply_supersede(cursor, doc_version_id: str, doc_status: str) -> list[str]:
        """版本退出链路：doc_version 置终态 + 其下 valid/need_review 切片全部置 expired。
        返回置前 status=valid 的 knowledge_id 列表（它们已在向量库中，供调用方增量删除向量）。
        检索两路均按「切片 valid ∧ 文档 valid」过滤，双重置后旧版自然退出召回；
        旧版遗留的 need_review 一并作废：已被新版取代的切片即使人工复核也不应再复活。"""
        live_ids = [row[0] for row in cursor.execute(
            "SELECT knowledge_id FROM knowledge_chunk WHERE doc_version_id = ? AND status = 'valid'",
            (doc_version_id,),
        ).fetchall()]
        cursor.execute(
            "UPDATE doc_version SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE doc_version_id = ?",
            (doc_status, doc_version_id),
        )
        cursor.execute(
            "UPDATE knowledge_chunk SET status = 'expired' "
            "WHERE doc_version_id = ? AND status IN ('valid','need_review')",
            (doc_version_id,),
        )
        return live_ids

    def write_new_version(self, doc_row: dict, chunk_rows: list[dict],
                          supersede_vid: str | None = None) -> list[str]:
        """事务内完成：旧版 superseded → 新 doc_version → 新切片（供 runner 调用）。
        返回旧版退出的 valid 切片 id（需从向量库增量删除）；FTS/向量同步由调用方
        在整批写完后统一触发（增量 delta 或全量回退）。"""
        retired: list[str] = []
        conn = self._get_conn()
        try:
            if supersede_vid:
                retired = self._apply_supersede(conn.cursor(), supersede_vid, "superseded")
            # 按行字典实际持有的列拼 INSERT（缺 created_at/updated_at 时走表 DEFAULT 而非写入 NULL）
            doc_cols = [c for c in DOC_COLUMNS if c in doc_row]
            conn.execute(
                f"INSERT INTO doc_version ({','.join(doc_cols)}) "
                f"VALUES ({','.join(['?'] * len(doc_cols))})",
                tuple(doc_row[c] for c in doc_cols),
            )
            conn.executemany(
                f"INSERT INTO knowledge_chunk ({','.join(CHUNK_COLUMNS)}) "
                f"VALUES ({','.join(['?'] * len(CHUNK_COLUMNS))})",
                [tuple(r[c] for c in CHUNK_COLUMNS) for r in chunk_rows],
            )
            conn.commit()
        except sqlite3.IntegrityError as e:
            conn.rollback()
            raise ValueError(f"自动摄取写库违反表约束：{e}") from e
        finally:
            conn.close()
        return retired

    def expire_document(self, doc_version_id: str) -> list[str]:
        """投放文件被移除 → 对应文档版本整版退出（doc_version.status=expired，切片置 expired）；
        返回退出的 valid 切片 id 供向量库增量删除"""
        conn = self._get_conn()
        try:
            retired = self._apply_supersede(conn.cursor(), doc_version_id, "expired")
            conn.commit()
            return retired
        finally:
            conn.close()

    def rebuild_fts(self) -> int:
        """公开的全量重建关键词索引入口（仅索引 valid 切片，幂等；千行级便宜，兼容回退用）"""
        if not self.fts_available:
            return 0
        conn = self._get_conn()
        try:
            n = self._rebuild_fts_index(conn)
            conn.commit()
            return n
        finally:
            conn.close()

    def fts_apply_delta(self, delete_ids: list[str], insert_rows: list[dict]) -> tuple[int, int]:
        """FTS 增量维护（替代每轮全量重建，支撑大批量文档流水线）：
        按 knowledge_id 删除退出 valid 的条目，插入本批新 valid 切片；仅动 delta，
        其余索引行不触碰（全量重建仅作为手动兼容回退通道）。返回 (删除数, 插入数)。"""
        if not self.fts_available or (not delete_ids and not insert_rows):
            return 0, 0
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            n_del = 0
            for kid in delete_ids:
                n_del += cursor.execute(f"DELETE FROM {FTS_TABLE} WHERE knowledge_id = ?", (kid,)).rowcount
            cursor.executemany(
                f"INSERT INTO {FTS_TABLE} (knowledge_id, body) VALUES (?, ?)",
                [(r["knowledge_id"], self._fts_body(r["content"], r["keywords"])) for r in insert_rows],
            )
            conn.commit()
            return n_del, len(insert_rows)
        finally:
            conn.close()

    def get_valid_knowledge_for_vectors(self, doc_version_ids: list[str]) -> pd.DataFrame:
        """查本批新入库版本中文档+切片双有效的行（含生效期元数据），供向量增量插入；
        与 sync_from_sqlite 同源同构（同一查询口径，保证增量与全量两条路径写入字段一致）"""
        if not doc_version_ids:
            return pd.DataFrame()
        conn = self._get_conn()
        placeholders = ",".join(["?"] * len(doc_version_ids))
        query = f'''
        SELECT
            k.knowledge_id, k.doc_version_id, k.chunk_type, k.content, k.original_text,
            k.keywords, d.doc_id, d.version, d.effective_date, d.expire_date, k.status
        FROM knowledge_chunk k
        JOIN doc_version d ON k.doc_version_id = d.doc_version_id
        WHERE k.status = 'valid' AND d.status = 'valid'
          AND k.doc_version_id IN ({placeholders})
        '''
        df = pd.read_sql(query, conn, params=doc_version_ids)
        conn.close()
        return df

    def verify_doc_files(self) -> list[dict]:
        """校验 doc_version.file_path 指向的原始文档是否在位，返回逐条对账报告。

        两种路径形态均支持：自动摄取写入的是绝对路径（直接存在性判定），历史/人工补录可能
        是相对 data/raw 的路径（拼接后判定）。模糊兜底按文件名在 data/raw 全量建索引，
        覆盖全部受支持格式（不仅 .txt），容忍空白/中点字符差异。
        本检查只报不阻断：missing 表示溯源原件不在位（切片仍可检索，但人工核验无据）。
        """
        suffixes = {".txt", ".md", ".pdf", ".docx", ".pptx"}
        norm_index = {normalize_filename(p.name): p
                      for p in RAW_DATA_DIR.rglob("*") if p.suffix.lower() in suffixes}

        conn = self._get_conn()
        df = pd.read_sql("SELECT doc_version_id, doc_name, file_path FROM doc_version", conn)
        conn.close()

        report = []
        for _, row in df.iterrows():
            fp = row["file_path"] or ""
            local = Path(fp)
            if not fp:
                local = None
            elif not local.is_absolute():
                local = RAW_DATA_DIR / fp.lstrip("/")
            if local and local.exists():
                result, matched = "ok", str(local)
            else:
                fuzzy_path = norm_index.get(normalize_filename(Path(fp).name))
                result, matched = ("fuzzy", str(fuzzy_path)) if fuzzy_path else ("missing", None)
            report.append({
                "doc_version_id": row["doc_version_id"],
                "doc_name": row["doc_name"],
                "file_path": fp,
                "result": result,
                "matched_path": matched,
            })
        return report

    def get_valid_knowledge_with_meta(self) -> pd.DataFrame:
        """获取文档与切片双有效的知识切片，携带文档元数据（生效/失效时间），供向量同步使用"""
        conn = self._get_conn()
        query = '''
        SELECT
            k.knowledge_id, k.doc_version_id, k.clause_position, k.heading_path,
            k.chunk_type, k.content, k.original_text, k.keywords, k.status,
            d.doc_id, d.doc_name, d.version, d.doc_type, d.publish_dept,
            d.publish_date, d.effective_date, d.expire_date
        FROM knowledge_chunk k
        JOIN doc_version d ON k.doc_version_id = d.doc_version_id
        WHERE k.status = 'valid' AND d.status = 'valid'
        '''
        df = pd.read_sql(query, conn)
        conn.close()
        return df

    def keyword_search(self, query: str, top_k: int = 10) -> list[dict]:
        """FTS5 关键词检索：BM25 精准召回，过滤切片/文档双有效且在生效期内。

        出参与向量路检索结果同构（不含 similarity），另附：
            keyword_score  BM25 原始分按当前查询最优值归一化到 [0,1]，保证不同查询间可比
            coverage       查询词元在切片正文（content+keywords）中的命中率，供弱命中诊断
        FTS5 不可用或分词后无有效词元时返回空列表，由上层退化为纯向量检索。
        """
        if not self.fts_available:
            return []
        tokens = tokenize_for_fts(query)
        if not tokens:
            return []
        # OR 连接扩大召回面，BM25 自然偏向命中词更多的切片；双引号包裹避免词元撞上 FTS5 查询语法
        match_expr = " OR ".join(f'"{t}"' for t in sorted(set(tokens)))
        query_tokens = set(tokens)

        conn = self._get_conn()
        try:
            rows = conn.execute(
                f'''
                SELECT k.knowledge_id, k.doc_version_id, k.clause_position, k.heading_path,
                       k.chunk_type, k.content, k.original_text, k.keywords,
                       d.doc_name, d.doc_type, d.version, d.publish_dept, d.effective_date,
                       bm25({FTS_TABLE}) AS raw_score
                FROM {FTS_TABLE} f
                JOIN knowledge_chunk k ON f.knowledge_id = k.knowledge_id
                JOIN doc_version d ON k.doc_version_id = d.doc_version_id
                WHERE {FTS_TABLE} MATCH ?
                  AND k.status = 'valid' AND d.status = 'valid'
                  AND (d.effective_date IS NULL OR d.effective_date <= date('now'))
                  AND (d.expire_date IS NULL OR d.expire_date >= date('now'))
                ORDER BY raw_score
                LIMIT ?
                ''',
                (match_expr, top_k),
            ).fetchall()
        except sqlite3.OperationalError as e:
            print(f"[警告] FTS5 关键词检索失败，退化为纯向量检索：{e}")
            return []
        finally:
            conn.close()

        if not rows:
            return []
        # BM25 原始分为负值（越小越相关），取绝对值后按当前查询最大值归一化
        max_raw = max(abs(r[-1]) for r in rows) or 1.0
        results = []
        for (kid, doc_vid, clause, heading, ctype, content, original, kw_raw,
             doc_name, doc_type, version, dept, eff_date, raw_score) in rows:
            body_tokens = set(tokenize_for_fts(f"{content} {kw_raw or ''}"))
            coverage = len(query_tokens & body_tokens) / len(query_tokens)
            results.append({
                "knowledge_id": kid,
                "doc_version_id": doc_vid,
                "chunk_type": ctype,
                "content": content,
                "original_text": original,
                "clause_position": clause,
                "heading_path": heading,
                "keywords": parse_keywords(kw_raw),
                "doc_name": doc_name,
                "doc_type": doc_type,
                "version": version,
                "publish_dept": dept,
                "effective_date": eff_date,
                "keyword_score": round(abs(raw_score) / max_raw, 4),
                "coverage": round(coverage, 4),
            })
        return results

    def get_knowledge_detail_by_ids(self, knowledge_ids: list[str]) -> pd.DataFrame:
        """根据切片 ID 批量查询完整详情，用于检索后补全信息"""
        if not knowledge_ids:
            return pd.DataFrame()
        conn = self._get_conn()
        placeholders = ",".join(["?"] * len(knowledge_ids))
        query = f'''
        SELECT
            k.knowledge_id, k.doc_version_id, k.clause_position, k.heading_path,
            k.chunk_type, k.content, k.original_text, k.keywords,
            d.doc_id, d.doc_name, d.version, d.doc_type, d.publish_dept,
            d.publish_date, d.effective_date, d.expire_date
        FROM knowledge_chunk k
        JOIN doc_version d ON k.doc_version_id = d.doc_version_id
        WHERE k.knowledge_id IN ({placeholders})
        '''
        df = pd.read_sql(query, conn, params=knowledge_ids)
        conn.close()
        return df


# 全局实例
sqlite_client = SqliteClient()
