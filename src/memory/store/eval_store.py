"""在线自动评估记录库（旁路观测闭环，纯增量）：把每次问答调用后的自动评估结果落库为独立表

与运行时链路完全解耦——本表只增不删、评估在响应返回后的后台任务中执行、不参与任何作答/拒答
决策，仅补齐在线质量观测数据：
    - 规则级护栏审计（generated 必须携带溯源切片）：护栏违规即「防线破坏」信号，回流排查优先级最高；
    - LLM judge（四维质量 + 幻觉声明审计）：按采样率执行，产出忠实度与质量分趋势，用于回归观测。

存储位置：与会话记忆库 l1_memory.db 同库、独立表（online_eval_records），与权威知识库物理隔离。
question/answer 入库前复用 masking 做 PII 脱敏，遵循 sqlite_store.py / feedback_store.py 的连接
与幂等建表惯例。
"""
import sqlite3
import uuid

from src.config import MEMORY_DB_PATH
from src.memory.schemas import EvalRecord

_OE_COLUMNS = [
    "eval_id", "session_id", "message_id", "channel", "question", "answer",
    "answer_type", "intent", "source_count", "elapsed_ms", "success",
    "guard_ok", "guard_reason", "judge_done", "quality_json", "faithfulness",
    "unsupported_claims", "judge_error", "created_at",
]


class EvalStore:
    def __init__(self):
        self.db_path = str(MEMORY_DB_PATH)
        self._init_tables()

    def _get_conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_tables(self):
        """幂等建表：online_eval_records（在线自动评估记录），供在线质量观测与护栏回归。"""
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute('''
        CREATE TABLE IF NOT EXISTS online_eval_records (
            eval_id            TEXT PRIMARY KEY,
            session_id         TEXT NOT NULL DEFAULT '',
            message_id         TEXT NOT NULL DEFAULT '',
            channel            TEXT NOT NULL DEFAULT 'chat',
            question           TEXT,
            answer             TEXT,
            answer_type        TEXT,
            intent             TEXT,
            source_count       INTEGER NOT NULL DEFAULT 0,
            elapsed_ms         INTEGER NOT NULL DEFAULT 0,
            success            INTEGER NOT NULL DEFAULT 1,
            guard_ok           INTEGER NOT NULL DEFAULT 1,
            guard_reason       TEXT,
            judge_done         INTEGER NOT NULL DEFAULT 0,
            quality_json       TEXT,
            faithfulness       REAL,
            unsupported_claims INTEGER NOT NULL DEFAULT 0,
            judge_error        TEXT,
            created_at         TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        cur.execute('''
        CREATE INDEX IF NOT EXISTS idx_eval_created
        ON online_eval_records(created_at)
        ''')
        cur.execute('''
        CREATE INDEX IF NOT EXISTS idx_eval_session
        ON online_eval_records(session_id)
        ''')
        conn.commit()
        conn.close()

    def record(self, rec: EvalRecord) -> EvalRecord:
        """落库一条评估记录：自动补 eval_id/created_at；question/answer 入库前 PII 脱敏后就地回写。"""
        from src.memory.store.masking import mask_pii  # 局部导入，避免模块级环依赖

        if not rec.eval_id:
            rec.eval_id = f"oev-{uuid.uuid4().hex[:12]}"
        if rec.question:
            rec.question = mask_pii(rec.question)
        if rec.answer:
            rec.answer = mask_pii(rec.answer)
        if not rec.created_at:
            rec.created_at = _now()

        conn = self._get_conn()
        try:
            conn.execute(
                f"INSERT OR REPLACE INTO online_eval_records "
                f"({','.join(_OE_COLUMNS)}) VALUES ({','.join(['?'] * len(_OE_COLUMNS))})",
                _row(rec),
            )
            conn.commit()
        finally:
            conn.close()
        return rec

    def update_judge(self, eval_id: str, quality_json: str | None, faithfulness: float | None,
                     unsupported_claims: int, judge_error: str | None) -> bool:
        """judge 结束后回写评估结论（judge_done 置 1），记录不存在时返回 False。"""
        conn = self._get_conn()
        try:
            cur = conn.execute(
                "UPDATE online_eval_records SET judge_done = 1, quality_json = ?, "
                "faithfulness = ?, unsupported_claims = ?, judge_error = ? WHERE eval_id = ?",
                (quality_json, faithfulness, unsupported_claims, judge_error, eval_id),
            )
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()

    def list_for_analysis(self, limit: int = 200) -> list[EvalRecord]:
        """取最近的评估记录（时间降序），供在线质量趋势与护栏违规回查。"""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                f"SELECT {', '.join(_OE_COLUMNS)} FROM online_eval_records "
                f"ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        finally:
            conn.close()
        return [_from_row(r) for r in rows]


def _row(rec: EvalRecord) -> tuple:
    """EvalRecord → 表行（布尔转 INTEGER，与 SQLite 存储对齐）"""
    return (
        rec.eval_id, rec.session_id, rec.message_id, rec.channel, rec.question, rec.answer,
        rec.answer_type, rec.intent, rec.source_count, rec.elapsed_ms, int(rec.success),
        int(rec.guard_ok), rec.guard_reason, int(rec.judge_done), rec.quality_json,
        rec.faithfulness, rec.unsupported_claims, rec.judge_error, rec.created_at,
    )


def _from_row(r: tuple) -> EvalRecord:
    return EvalRecord(
        eval_id=r[0], session_id=r[1], message_id=r[2], channel=r[3], question=r[4] or "",
        answer=r[5] or "", answer_type=r[6], intent=r[7], source_count=r[8],
        elapsed_ms=r[9], success=bool(r[10]), guard_ok=bool(r[11]), guard_reason=r[12],
        judge_done=bool(r[13]), quality_json=r[14], faithfulness=r[15],
        unsupported_claims=r[16], judge_error=r[17], created_at=r[18],
    )


def _now() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# 全局实例（懒初始化，首次访问建表）
eval_store = EvalStore()
