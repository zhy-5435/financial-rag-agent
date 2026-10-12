"""在线自动评估管道（旁路）：每次问答调用返回后，对「问题-回答-证据」异步执行质量评估并落库

评估分两层，均在响应返回后的后台任务中执行、绝不阻塞用户响应、任何异常静默不影响主链路：
    Tier1 规则级（零 LLM 成本，每轮必做）：链路成败、护栏合规审计（generated 必须携带溯源切片，
          违规即「防线破坏」信号）、时延与来源数量登记；
    Tier2 LLM judge 级（按 ONLINE_EVAL_JUDGE_RATE 采样）：复用《系统评测报告》验证过的两个提示词——
          四维质量打分（相关性/准确性/完整性/有用性 + 是否解决）与幻觉声明审计（事实断言逐条对照
          证据，输出 supported 比例作忠实度）；仅 generated/searched/agentic 评忠实度（闲聊无事实
          断言、行情逐字直出天然有据、拒答无断言，与离线评测口径一致）。

落库：l1_memory.db 独立表 online_eval_records（经 memory_manager 门面），judge 结论异步回写。
并发：全局信号量限 judge 并发（防高 QPS 打满 LLM 额度）；任务 fire-and-forget，常驻引用防 GC。
"""
import asyncio
import json
import random
import uuid

from src.agent.config import (
    ONLINE_EVAL_ANSWER_MAX,
    ONLINE_EVAL_ENABLED,
    ONLINE_EVAL_JUDGE_CONCURRENCY,
    ONLINE_EVAL_JUDGE_RATE,
    ONLINE_EVAL_JUDGE_TIMEOUT,
)
from src.agent.llm import llm
from src.memory.memory_manager import memory_manager
from src.memory.schemas import EvalRecord

# judge 输入截断口径（与 tests/eval_e2e.py 离线评测一致）
_ANSWER_MAX_CHARS = 3000
_EVIDENCE_MAX_CHARS = 800

JUDGE_QUALITY_PROMPT = """你是严格的问答质量评审员。根据「用户问题、助手回答、证据」按四个维度打分（1~5 整数）：
- relevance 相关性：回答是否针对所问内容（答非所问=1，切题=4~5）
- accuracy 准确性：回答内容与证据是否一致、有无编造（无据编造=1，与证据一致=5）
- completeness 完整性：问题要求数量/类型的要素是否给全（遗漏关键要素=2~3，要素齐全=5）
- usefulness 有用性：用户拿到该回答能否直接解决问题（含来源、可执行性）
评分须知：
- 若回答为「知识库中未找到权威依据」类拒答：accuracy=5（合规行为）、relevance=5、completeness≤3、usefulness=3、resolved=true
- 行情类回答：数字为数据源直出，accuracy 依据自洽性；表格要素（最新价/涨跌幅/来源/免责）齐全则 completeness=5
- 闲聊类回答：accuracy 与 usefulness 按「语气得体、不误导」评，resolved=true 表示得体回应
- resolved：用户问题是否被完整解决（合规拒答也算正确解决）
只输出 JSON：{"relevance":x,"accuracy":x,"completeness":x,"usefulness":x,"resolved":true/false,"reason":"一句话"}

【用户问题】@@QUESTION@@
【助手回答】
@@ANSWER@@
【证据】
@@EVIDENCE@@"""

JUDGE_FAITHFULNESS_PROMPT = """你是幻觉审计员。从助手回答中抽取「事实性断言」（数字、比率、条款要求、费率、规则、日期等），
逐条对照证据判定：supported=证据可支持；unsupported=证据中不存在或矛盾；non-factual=寒暄/免责声明/指路性话术，不属于事实断言。
须知：
- 行情类回答的数字来自数据源直出（即回答本体），一律 supported
- 网搜简报中标注了白名单媒体 URL 的表述，视为 supported（来源可溯）
- 拒答固定话术不含事实断言
只输出 JSON：{"claims":[{"text":"断言摘要","verdict":"supported|unsupported|non-factual"}],"has_hallucination":true/false}
has_hallucination = 存在至少一条 unsupported。

【用户问题】@@QUESTION@@
【助手回答】
@@ANSWER@@
【证据】
@@EVIDENCE@@"""

# 评忠实度的作答类型：基于证据生成的三类（拒答/闲聊/行情直出无事实断言或天然有据）
_FAITH_ANSWER_TYPES = {"generated", "searched", "agentic"}

# judge 并发闸（进程级全局，防高 QPS 时评估调用打满 LLM 额度与连接池）
_judge_sem = asyncio.Semaphore(ONLINE_EVAL_JUDGE_CONCURRENCY)
# fire-and-forget 任务的常驻引用，防止任务被 GC 半路取消
_bg_tasks: set[asyncio.Task] = set()


def spawn_online_eval(
    *,
    channel: str,
    session_id: str,
    message_id: str,
    question: str,
    answer: str,
    answer_type: str | None,
    intent: str | None,
    source_count: int,
    elapsed_ms: int,
    success: bool,
    sources: list[dict] | None = None,
) -> None:
    """构造规则级评估记录并 spawn 后台评估任务（同步入口，响应返回前调用，不阻塞）。

    sources 传 dict 列表（SourceHit.model_dump() 或 graph 原始 dict），仅用于 judge 证据拼装。
    """
    if not ONLINE_EVAL_ENABLED:
        return
    guard_ok, guard_reason = _rule_guard(answer_type, source_count, success)
    rec = EvalRecord(
        eval_id=f"oev-{uuid.uuid4().hex[:12]}",
        session_id=session_id or "",
        message_id=message_id or "",
        channel=channel,
        question=question or "",
        answer=(answer or "")[:ONLINE_EVAL_ANSWER_MAX],
        answer_type=answer_type,
        intent=intent,
        source_count=source_count,
        elapsed_ms=elapsed_ms,
        success=success,
        guard_ok=guard_ok,
        guard_reason=guard_reason,
    )
    task = asyncio.create_task(_run(rec, sources or []))
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)


def _rule_guard(answer_type: str | None, source_count: int, success: bool) -> tuple[bool, str | None]:
    """Tier1 护栏合规：generated 必须携带溯源切片（金融防线不变量），异常链路视为待查违规。"""
    if not success:
        return False, "链路异常（未正常产出回答）"
    if answer_type == "generated" and source_count <= 0:
        return False, "generated 回答无溯源切片（护栏不变量被破坏）"
    return True, None


async def _run(rec: EvalRecord, sources: list[dict]) -> None:
    """后台评估主流程：Tier1 先落库 → Tier2 按采样执行 judge → 回写结论。任何异常静默退出。"""
    try:
        memory_manager.record_eval(rec)
    except Exception:
        return  # 评估是旁路：落库失败不打扰主链路，任务静默结束
    if rec.success and _should_judge():
        await _run_judge(rec, sources)


def _should_judge() -> bool:
    """Tier2 采样判定：rate=1.0 全量，rate=0.5 半量，rate=0 关闭 judge（规则级不受影响）。"""
    return ONLINE_EVAL_JUDGE_RATE > 0 and random.random() <= ONLINE_EVAL_JUDGE_RATE


async def _run_judge(rec: EvalRecord, sources: list[dict]) -> None:
    """Tier2：并行执行四维质量与忠实度两个 judge，结论一次性回写（单项失败记 judge_error）。"""
    evidence = _evidence_pack(rec.answer_type, sources)
    quality_prompt = _fill(JUDGE_QUALITY_PROMPT, rec.question, rec.answer, evidence)
    faith_prompt = (
        _fill(JUDGE_FAITHFULNESS_PROMPT, rec.question, rec.answer, evidence)
        if rec.answer_type in _FAITH_ANSWER_TYPES
        else None  # 拒答/闲聊/行情直出无事实断言或天然有据，与离线评测口径一致
    )

    try:
        async with _judge_sem:
            results = await asyncio.gather(
                _judge_json(quality_prompt),
                _judge_json(faith_prompt) if faith_prompt else _skip(),
                return_exceptions=True,
            )
    except Exception as e:  # 信号量等待被取消等极端情况：仍要把 judge_done 置位
        _safe_write_judge(rec.eval_id, None, None, 0, f"judge 调度失败：{_err_text(e)}")
        return

    quality = results[0] if isinstance(results[0], dict) else None
    faith = results[1] if isinstance(results[1], dict) else None
    errors = [_err_text(r) for r in results if isinstance(r, BaseException)]

    faithfulness: float | None = None
    unsupported = 0
    if faith is not None:
        claims = faith.get("claims") if isinstance(faith.get("claims"), list) else []
        verdicts = [c.get("verdict") for c in claims if isinstance(c, dict)]
        supported = verdicts.count("supported")
        unsupported = verdicts.count("unsupported")
        if supported + unsupported > 0:
            faithfulness = round(supported / (supported + unsupported), 4)

    quality_json = json.dumps(quality, ensure_ascii=False) if quality is not None else None
    _safe_write_judge(rec.eval_id, quality_json, faithfulness, unsupported,
                      "；".join(errors)[:300] if errors else None)


async def _skip() -> dict:
    """忠实度 judge 的占位结果（该作答类型无可评断言）。"""
    return {"claims": [], "has_hallucination": False, "skipped": True}


async def _judge_json(prompt: str) -> dict:
    """单次 judge 调用并解析 JSON。在线旁路不做重试（失败记 judge_error，下轮流量自然补样）。"""
    msg = await asyncio.wait_for(llm.ainvoke(prompt), timeout=ONLINE_EVAL_JUDGE_TIMEOUT)
    return _parse_json(getattr(msg, "content", "") or "")


def _safe_write_judge(eval_id: str, quality_json: str | None, faithfulness: float | None,
                      unsupported: int, judge_error: str | None) -> None:
    """judge 结论回写（judge_done 置 1）：失败静默，仅影响观测完整性不影响主链路。"""
    try:
        memory_manager.update_eval_judge(
            eval_id, quality_json=quality_json, faithfulness=faithfulness,
            unsupported_claims=unsupported, judge_error=judge_error,
        )
    except Exception:
        pass


def _fill(template: str, question: str, answer: str, evidence: str) -> str:
    """占位符替换（提示词内含 JSON 花括号示例，不能用 str.format）"""
    return (template
            .replace("@@QUESTION@@", question or "")
            .replace("@@ANSWER@@", (answer or "")[:_ANSWER_MAX_CHARS])
            .replace("@@EVIDENCE@@", evidence[:4000]))


def _evidence_pack(answer_type: str | None, sources: list[dict]) -> str:
    """汇总本轮可核验证据：知识库切片原文 + 行情/网搜语义说明（与离线评测口径一致）。"""
    parts: list[str] = []
    for s in sources or []:
        body = (s.get("original_text") or s.get("content") or "")[:_EVIDENCE_MAX_CHARS]
        parts.append(f"[{s.get('doc_name') or ''} {s.get('version') or ''} {s.get('clause_position') or ''}] {body}")
    if answer_type == "quoted":
        parts.append("[行情工具直出数据（即回答本体，视为有据）]")
    if answer_type == "searched":
        parts.append("[网搜简报：回答中引用的白名单媒体 URL 即证据来源]")
    return "\n".join(parts) if parts else "（无结构化证据）"


def _parse_json(text: str) -> dict:
    """宽容解析 judge 的 JSON 输出：剥离代码围栏，截取首尾花括号之间内容。"""
    s = (text or "").strip()
    if s.startswith("```"):
        s = s.split("```", 2)[1]
        if s.startswith("json"):
            s = s[4:]
        s = s.strip()
    start, end = s.find("{"), s.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"无 JSON 结构：{s[:80]}")
    return json.loads(s[start:end + 1])


def _err_text(e: BaseException) -> str:
    """压缩异常为简短错误串（类型 + 摘要），供 judge_error 落库观测。"""
    return f"{type(e).__name__}:{str(e)[:120]}"
