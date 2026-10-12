"""会话记忆门面：对 L3 graph/api 暴露「落库本轮 + 构建上下文 + 历史恢复」的高层入口

职责边界：
    - 只读/写持久化层（store），只调上下文引擎（context），不被 store/context 反向依赖；
    - 上层（graph 的 LLM 分支作答前）调 build() 取精简上下文；结果产出后调 remember_* 落库。
"""
import json
from collections import Counter

from src.memory.context.builder import ContextBuilder
from src.memory.context.tokenizer import estimate
from src.memory.schemas import (
    ContextBundle,
    EvalRecord,
    FeedbackCategory,
    FeedbackRecord,
    SecurityEventRecord,
    SessionSummary,
    StoredMessage,
)
from src.memory.store.eval_store import eval_store
from src.memory.store.feedback_store import feedback_store
from src.memory.store.security_store import security_store
from src.memory.store.sqlite_store import conversation_store
from src.memory.store.write_queue import memory_writer


class MemoryManager:
    def __init__(self, store=conversation_store, builder: ContextBuilder | None = None,
                 feedback=feedback_store, security=security_store, evals=eval_store):
        self.store = store
        self.builder = builder or ContextBuilder()
        self.feedback = feedback
        self.security = security
        self.eval = evals

    # ---------- 构建推理上下文（调用模型前） ----------

    def build(
        self,
        session_id: str,
        base_system_prompt: str,
        question: str,
        *,
        mode: str = "turns",
        extra_tokens_estimate: int = 0,
        persist_summary: bool = True,
        before_seq: int | None = None,
    ) -> ContextBundle:
        """加载会话历史 → 上下文引擎按档压缩 → 返回可直接 invoke 的消息视图。

        extra_tokens_estimate：本轮「新鲜内容」（如 kb 检索证据、news 网搜素材）的预估占用，
        纳入预算档位判断但不参与历史压缩。
        before_seq：仅加载该序号**之前**的历史（本轮用户消息已先落库时，用它排除本轮）。
        """
        history = self.store.load_messages(session_id, before_seq=before_seq)
        return self.builder.build_context(
            session_id,
            base_system_prompt,
            question,
            history,
            mode=mode,
            extra_tokens_estimate=extra_tokens_estimate,
            persist_summary=persist_summary,
        )

    # ---------- 落库本轮（审计存档，永不删除） ----------

    def next_seq(self, session_id: str) -> int:
        """预分配本轮 user 消息的 seq（廉价只读，供关键路径先行拿号、落库改走后台队列）。"""
        return self.store.next_seq(session_id)

    def begin_turn_user(self, session_id: str, content: str) -> int:
        """prepare 关键路径入口：先排空上一轮已提交的写入（保证本轮拿号/历史一致），
        再预分配本轮 user 的 seq 并把写入挂后台队列，立即返回 seq（不阻塞首 token）。

        稳态下上一轮写入早已落库，drain 瞬时返回；user 写入与后续 tool/assistant 写入同走
        单线程 FIFO 队列，seq 单调不乱。"""
        memory_writer.drain()
        seq = self.store.next_seq(session_id)
        memory_writer.submit(
            self._append, session_id=session_id, role="user", content=content, intent=None, seq=seq
        )
        return seq

    def compact_session(self, session_id: str, base_system_prompt: str) -> None:
        """后台主动压缩（"提早开始摘要"）：按当前历史体量提前走一次上下文组装，
        仅在会命中压缩档时才调摘要 LLM 并落库（direct 档无副作用），把重活从下一轮 prepare
        的关键路径上挪开。失败不阻断（下一轮 prepare 仍可同步按需摘要兼底）。

        必须在后台线程调用（内含可能的阻塞式摘要 LLM）；压缩落库走 memory_writer 队列保序。"""
        try:
            memory_writer.drain()  # 确保本轮 user/tool/assistant 已落库，再据完整历史判定档位
            history = self.store.load_messages(session_id)
            if not history:
                return
            self.builder.build_context(
                session_id, base_system_prompt, "", history,
                mode="turns", persist_summary=True,
            )
        except Exception as e:  # noqa: BLE001  后台压缩失败不影响主链路
            import logging

            logging.getLogger(__name__).warning("后台主动压缩失败（忽略）：%s", e)

    def remember_user(self, session_id: str, content: str, intent: str | None = None) -> StoredMessage:
        return self._append(
            session_id=session_id, role="user", content=content, intent=intent
        )

    def remember_user_async(
        self, session_id: str, content: str, seq: int, intent: str | None = None
    ) -> None:
        """prepare 关键路径用：以预分配 seq 把 user 消息写入提交后台队列，立即返回（不阻塞首 token）。"""
        memory_writer.submit(
            self._append, session_id=session_id, role="user", content=content,
            intent=intent, seq=seq,
        )

    def remember_tool(
        self, session_id: str, tool_name: str, content: str, payload: object | None = None
    ) -> StoredMessage:
        """落库工具结果：payload 存结构化 JSON（供上下文引擎精简渲染），content 存可读摘要文本"""
        tool_payload = json.dumps(payload, ensure_ascii=False) if payload is not None else None
        return self._append(
            session_id=session_id, role="tool", content=content,
            tool_name=tool_name, tool_payload=tool_payload,
        )

    def remember_tool_async(
        self, session_id: str, tool_name: str, content: str, payload: object | None = None
    ) -> None:
        """act 关键路径用：工具结果写入提交后台队列，与后续作答 LLM 调用重叠，不占墙上时间。"""
        tool_payload = json.dumps(payload, ensure_ascii=False) if payload is not None else None
        memory_writer.submit(
            self._append, session_id=session_id, role="tool", content=content,
            tool_name=tool_name, tool_payload=tool_payload,
        )

    def remember_assistant(
        self, session_id: str, content: str, answer_type: str, intent: str | None = None,
        message_id: str = "",
    ) -> StoredMessage:
        return self._append(
            session_id=session_id, role="assistant", content=content,
            intent=intent, answer_type=answer_type, message_id=message_id,
        )

    def remember_assistant_async(
        self, session_id: str, message_id: str, content: str, answer_type: str,
        intent: str | None = None,
    ) -> None:
        """finalize 出口用：助手回答写入提交后台队列（message_id 由调用方预生成，无需等落库）。"""
        memory_writer.submit(
            self._append, session_id=session_id, role="assistant", content=content,
            intent=intent, answer_type=answer_type, message_id=message_id,
        )

    def _append(self, **kwargs) -> StoredMessage:
        content = kwargs.get("content", "")
        msg = StoredMessage(
            message_id=kwargs.pop("message_id", "") or _gen_msg_id(),
            session_id=kwargs["session_id"],
            seq=kwargs.pop("seq", 0) or 0,  # 0=交由 store 自动分配；预分配时原样入库
            role=kwargs["role"],
            content=content,
            intent=kwargs.get("intent"),
            answer_type=kwargs.get("answer_type"),
            tool_name=kwargs.get("tool_name"),
            tool_payload=kwargs.get("tool_payload"),
            token_est=estimate(content),
        )
        return self.store.append_message(msg)

    # ---------- 回答反馈标注（HITL 反馈闭环，纯增量、不影响运行时） ----------

    def record_feedback(
        self, session_id: str, message_id: str, category: FeedbackCategory, comment: str | None = None
    ) -> FeedbackRecord:
        """落库一条回答反馈：answer_type/intent/提问由服务端从被反馈消息回填，不采信前端上送。"""
        answer_type: str | None = None
        intent: str | None = None
        question: str | None = None
        msg = self.store.get_message(message_id)
        if msg is not None and msg.session_id == session_id:
            answer_type = msg.answer_type
            intent = msg.intent
            question = self.store.preceding_user_question(session_id, msg.seq)
        rec = FeedbackRecord(
            feedback_id="", session_id=session_id, message_id=message_id,
            category=category, comment=comment, answer_type=answer_type,
            intent=intent, question=question,
        )
        return self.feedback.record(rec)

    def list_feedback(self, category: str | None = None, answer_type: str | None = None,
                      limit: int = 200) -> list[FeedbackRecord]:
        """按类别/作答类型取最近的反馈标注，供离线校准（拒答 vs 低置信回流分析）。"""
        return self.feedback.list_for_analysis(category=category, answer_type=answer_type, limit=limit)

    # ---------- 提示词注入安全事件（S3.3 审计闭环，纯增量、不参与运行时决策） ----------

    def record_security_event(self, rec: SecurityEventRecord) -> SecurityEventRecord:
        """落库一条注入/审计安全事件（detail 入库前 PII 脱敏），供红队回流与拦截率观测。"""
        return self.security.record(rec)

    def record_security_event_async(self, rec: SecurityEventRecord) -> None:
        """安全事件写入提交后台队列（审计留痕不占作答关键路径）。"""
        memory_writer.submit(self.security.record, rec)

    def list_security_events(self, rule: str | None = None, session_id: str | None = None,
                             limit: int = 200) -> list[SecurityEventRecord]:
        """按规则/会话取最近安全事件，供审计回流分析。"""
        return self.security.list_for_analysis(rule=rule, session_id=session_id, limit=limit)

    # ---------- 在线自动评估（每次问答调用后异步产出，纯增量、不参与运行时决策） ----------

    def record_eval(self, rec: EvalRecord) -> EvalRecord:
        """落库一条在线评估记录：规则级结论先落，judge 结论由后台任务完成后回写。"""
        return self.eval.record(rec)

    def update_eval_judge(self, eval_id: str, quality_json: str | None,
                          faithfulness: float | None, unsupported_claims: int,
                          judge_error: str | None) -> bool:
        """回写 LLM judge 评估结论（judge_done 置 1），记录不存在时返回 False。"""
        return self.eval.update_judge(
            eval_id, quality_json=quality_json, faithfulness=faithfulness,
            unsupported_claims=unsupported_claims, judge_error=judge_error,
        )

    def list_eval_records(self, limit: int = 200) -> list[EvalRecord]:
        """取最近的在线评估记录，供质量趋势与护栏违规回查。"""
        return self.eval.list_for_analysis(limit=limit)

    def summarize_eval(self, limit: int = 200) -> dict:
        """汇总最近 N 条评估：成功率、护栏合规率、忠实度与四维质量均值、延迟分位、作答类型分布。"""
        recs = self.eval.list_for_analysis(limit=limit)
        total = len(recs)
        if not total:
            return {"total": 0}
        guard_bad = [r for r in recs if not r.guard_ok]
        judged = [r for r in recs if r.judge_done]
        faith = [r.faithfulness for r in judged if r.faithfulness is not None]
        quality_keys = ("relevance", "accuracy", "completeness", "usefulness")
        quality_scores: dict[str, list[float]] = {k: [] for k in quality_keys}
        resolved: list[bool] = []
        judge_errors = 0
        for r in judged:
            if r.judge_error:
                judge_errors += 1
            if not r.quality_json:
                continue
            try:
                q = json.loads(r.quality_json)
            except json.JSONDecodeError:
                continue
            for k in quality_keys:
                if isinstance(q.get(k), (int, float)):
                    quality_scores[k].append(q[k])
            if isinstance(q.get("resolved"), bool):
                resolved.append(q["resolved"])
        elapsed = sorted(r.elapsed_ms for r in recs)

        def _avg(seq):
            return round(sum(seq) / len(seq), 4) if seq else None

        def _pct(seq: list[int], p: int):
            if not seq:
                return None
            idx = min(len(seq) - 1, max(0, round(p / 100 * (len(seq) - 1))))
            return seq[idx]

        return {
            "total": total,
            "success_rate": _avg([1 if r.success else 0 for r in recs]),
            "guard_violations": len(guard_bad),
            "guard_reasons": sorted({r.guard_reason for r in guard_bad if r.guard_reason}),
            "judge_done": len(judged),
            "judge_error_count": judge_errors,
            "faithfulness_avg": _avg(faith),
            "hallucination_record_count": sum(1 for v in faith if v < 1.0),
            "quality_avg": {k: _avg(v) for k, v in quality_scores.items()},
            "resolved_rate": _avg([1 if v else 0 for v in resolved]) if resolved else None,
            "elapsed_p50_ms": _pct(elapsed, 50),
            "elapsed_p90_ms": _pct(elapsed, 90),
            "answer_type_dist": dict(Counter(r.answer_type or "unknown" for r in recs)),
        }

    # ---------- 历史恢复 ----------

    def list_sessions(self, limit: int = 50) -> list[SessionSummary]:
        return self.store.list_sessions(limit=limit)

    def get_messages(self, session_id: str, limit: int | None = None) -> list[StoredMessage]:
        return self.store.load_messages(session_id, limit=limit)

    def delete_session(self, session_id: str):
        self.store.delete_session(session_id)


def _gen_msg_id() -> str:
    import uuid

    return f"msg-{uuid.uuid4().hex[:12]}"


# 全局单例
memory_manager = MemoryManager()
