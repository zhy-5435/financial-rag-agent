"""局部/全局摘要：压缩历史为业务要点，并做关键实体存活校验

- 摘要模型默认复用 L3 的 llm 实例（构造注入，便于单测 mock）；
- 生成后从被覆盖消息抽取关键实体，若摘要正文遗漏则强制补挂「关键实体清单」块，
  保证锁存实体不因压缩而丢失；
- 摘要与 covered_message_ids/version 登记进 compression_log，原始消息仅置状态不删除；
- 落库（compression_log + 被覆盖标记 + 锁存 system-summary）走后台单线程写队列，与本轮
  其它写入按 FIFO 保序（避免与 prepare 预分配的 user seq 碰撞），不占调用线程墙上时间。
"""
import uuid

from src.memory.config import MAX_SUMMARY_TOKENS
from src.memory.context.tagging import extract_entities
from src.memory.context.tokenizer import estimate
from src.memory.prompts import KEY_ENTITY_SUFFIX, SUMMARIZE_SYSTEM_PROMPT, render_transcript
from src.memory.schemas import CompressionRecord, PriorityTag, StoredMessage
from src.memory.store.sqlite_store import conversation_store
from src.memory.store.write_queue import memory_writer


class Summarizer:
    def __init__(self, llm=None):
        # 延迟注入：默认 None 时首次调用再取全局 llm，避免模块导入即拉起模型依赖
        self._llm = llm

    def _model(self):
        if self._llm is None:
            from src.agent.llm import llm  # 跨层取模型实例（仅摘要链路需要）

            self._llm = llm
        return self._llm

    def summarize(
        self,
        session_id: str,
        covered: list[StoredMessage],
        scope: str,
        persist: bool = True,
    ) -> CompressionRecord:
        """对 covered 消息生成摘要并登记；scope=local/global。persist=False 时仅返回不落库（供单测）"""
        covered_ids = [m.message_id for m in covered]
        key_entities = _collect_key_entities(covered)
        transcript = render_transcript([{"role": m.role, "content": m.content} for m in covered])

        system = SUMMARIZE_SYSTEM_PROMPT.format(max_tokens=MAX_SUMMARY_TOKENS)
        try:
            response = self._model().invoke([
                {"role": "system", "content": system},
                {"role": "user", "content": transcript},
            ])
            summary_text = (response.content or "").strip()
        except Exception as e:
            # 摘要失败不得阻断主链路：退化为「实体清单 + 首条截断」占位摘要
            summary_text = f"（摘要生成失败，降级保留关键实体）{e}"

        # 关键实体存活校验：摘要遗漏的实体补挂清单块
        missing = [ent for ent in key_entities if ent not in summary_text]
        if missing:
            summary_text += KEY_ENTITY_SUFFIX.format(entities="、".join(missing))

        rec = CompressionRecord(
            record_id="",
            session_id=session_id,
            scope=scope,  # type: ignore[arg-type]
            version=0,
            covered_message_ids=covered_ids,
            summary_text=summary_text,
            summary_token_est=estimate(summary_text),
            key_entities=key_entities,
        )

        if persist:
            # record_id 需同步可知（build 用它拼 summary-{record_id} 注入伪消息并回传 summary_ref），
            # 先落号再把三条写库提交后台队列（version 在队列内按序分配，避免并发摘要撞版本）。
            rec.record_id = f"cmp-{uuid.uuid4().hex[:12]}"
            memory_writer.submit(_persist_compression, rec, covered_ids)
        return rec


def _persist_compression(rec: CompressionRecord, covered_ids: list[str]) -> None:
    """后台队列内执行摘要落库：登记压缩日志 → 被覆盖原文标 replaced（不删除，仅退出直接注入）
    → 追加一条锁存级 system-summary 消息，使摘要在后续轮次持久可见（历史要点跨轮不丢失）。"""
    conversation_store.record_compression(rec)
    conversation_store.mark_compression_state(covered_ids, "replaced")
    conversation_store.append_message(StoredMessage(
        message_id=f"summary-{rec.record_id}",
        session_id=rec.session_id,
        seq=0,
        role="system-summary",
        content=rec.summary_text,
        priority_tag=PriorityTag.LATCHED_ENTITY,
        token_est=rec.summary_token_est,
    ))


def _collect_key_entities(messages: list[StoredMessage]) -> list[str]:
    """汇总被覆盖消息中的关键实体（代码/指数），去重保序，供存活校验"""
    seen: list[str] = []
    for m in messages:
        for ent in extract_entities(m.content):
            if ent not in seen:
                seen.append(ent)
    return seen
