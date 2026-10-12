"""Ragas 离线自动化评估：在项目根目录输入命令即手动触发，一次跑完四项指标并落盘结果

用法（无需启动 L2/L3 服务，只读本地测试集，纯离线）：
    .venv1\\Scripts\\python.exe -m src.agent.ragas_eval                    # 全量评估 + 打分理由
    .venv1\\Scripts\\python.exe -m src.agent.ragas_eval --limit 3           # 只跑前 3 条（冒烟）
    .venv1\\Scripts\\python.exe -m src.agent.ragas_eval --no-reasons        # 只出分，不调 Judge 写理由
    .venv1\\Scripts\\python.exe -m src.agent.ragas_eval --testset data/eval/ragas_testset.json --json
    .venv1\\Scripts\\python.exe -m src.agent.ragas_eval --no-md            # 不生成 Markdown 报告（默认生成）
    .venv1\\Scripts\\python.exe -m src.agent.ragas_eval --help              # 查看全部参数

评估四指标（Ragas 官方核心检索/生成指标，0~1，越高越好）：
  faithfulness       忠实度：回答是否只讲检索切片里有的事实（幻觉代理）
  answer_relevancy   答案相关性：回答是否切题
  context_precision 上下文精确率：有用切片是否排在前面
  context_recall     上下文召回率：标准答案要点是否被检索覆盖

综合得分：四项按权重加权（权重取配置 EVAL_COMPOSITE_WEIGHTS，默认四项等权），
        单样本先加权、再对样本取平均；某项缺失时按剩余权重归一，不让缺项拉低总分。

数据流：本地 JSON 测试集（question / retrieved_contexts / answer / ground_truth）
        → ragas.evaluate 打分 → Judge 模型逐项生成「打分理由」→ 算综合得分
        → 终端报表 + 落盘 CSV（含得分/综合分/理由/系统回答/标准答案）与 Markdown 报告
        （--json 可另存同口径的完整结果 JSON）。

设计约束：
  1. 只有手动输入本命令才执行评估，无自动执行、无定时任务；一次运行串行跑完，重复执行请等上一次结束
  2. 评估异常不中断整体：单样本打分失败该项记 None，理由生成失败写占位文案，进程退出码 0/1 可判成败
  3. 配置项沿用 .env：Judge 模型（EVAL_JUDGE_MODEL）、测试集路径（EVAL_TESTSET_PATH）、结果保存路径（EVAL_RESULTS_DIR）
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import math
import os
import re
import sys
import types
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import httpx
from langchain_core.embeddings import Embeddings

from src.agent.config import (
    EVAL_COMPOSITE_WEIGHTS,
    EVAL_ENABLED,
    EVAL_JUDGE_CONCURRENCY,
    EVAL_JUDGE_DISABLE_THINKING,
    EVAL_JUDGE_MAX_TOKENS,
    EVAL_JUDGE_MODEL,
    EVAL_JUDGE_RETRY_MAX_TOKENS,
    EVAL_JUDGE_TIMEOUT,
    EVAL_MAX_RETRIES,
    EVAL_MAX_WORKERS,
    EVAL_MD_REPORT,
    EVAL_RESULTS_DIR,
    EVAL_TESTSET_PATH,
    EVAL_TIMEOUT,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_EXTRA_BODY,
    LLM_MODEL,
    LLM_TEMPERATURE,
)
from src.config import PROJECT_ROOT

# 指标键与对外中文标签（终端报表与 CSV 表头都用这套命名）
METRICS: dict[str, str] = {
    "faithfulness": "忠实度",
    "answer_relevancy": "答案相关性",
    "context_precision": "上下文精确率",
    "context_recall": "上下文召回率",
}

# 四项加权得到的综合同名指标（权重取配置 EVAL_COMPOSITE_WEIGHTS）
COMPOSITE_LABEL = "综合得分"


def _composite(scores: dict[str, Any]) -> float | None:
    """单样本加权综合得分：只对有值的指标计入，并按命中权重归一（全缺返回 None）

    归一处理是为了不让「某项没打到分」直接拉低综合分，而是退化成已打分项的加权平均。
    """
    acc = weight = 0.0
    for key, w in EVAL_COMPOSITE_WEIGHTS.items():
        v = scores.get(key)
        if v is None:
            continue
        acc += float(v) * w
        weight += w
    return round(acc / weight, 4) if weight > 0 else None


def _weights_text() -> str:
    """权重的对外展示文本，如「忠实度0.25 + 答案相关性0.25 + …」"""
    return " + ".join(f"{METRICS[k]}{w:g}" for k, w in EVAL_COMPOSITE_WEIGHTS.items())


def _now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


# ---------- ragas 依赖兼容 ----------

def _apply_compat_stubs() -> None:
    """ragas 0.4.3 兼容处理：vertexai 硬引用 + 遥测序列化崩溃

    1) ragas 无条件 from langchain_community.chat_models.vertexai import ChatVertexAI，
       而 langchain-community 0.4.2 已移除该模块 → 注入空实现桩（本项目不用 VertexAI）。
    2) 遥测事件 EmbeddingUsageEvent 序列化本地 SentenceTransformer 对象会崩溃并连带
       整个 Job 失败（answer_relevancy 全 NaN）→ 关闭遥测。
    """
    os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
    mod_name = "langchain_community.chat_models.vertexai"
    if mod_name not in sys.modules:
        try:
            import langchain_community.chat_models.vertexai  # noqa: F401  已存在则无需桩
        except Exception:
            stub = types.ModuleType(mod_name)

            class ChatVertexAI:  # 占位类型，仅为满足 ragas 的导入与类型引用
                pass

            stub.ChatVertexAI = ChatVertexAI
            sys.modules[mod_name] = stub
            import langchain_community.chat_models as cm

            cm.vertexai = stub


def _ensure_event_loop() -> None:
    """绑定一个事件循环：ragas 内部部分包装类仍调 asyncio.get_event_loop()"""
    try:
        asyncio.get_running_loop()
        return  # 已在循环内（被其他异步程序 import 调用时），不再新建
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)


class _BgeEmbeddings(Embeddings):
    """本地 bge-base-zh-v1.5 的 ragas Embeddings 适配（纯文本相似度，不加检索指令前缀）"""

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        from src.data.config import EMBEDDING_MODEL_PATH

        self._model = SentenceTransformer(EMBEDDING_MODEL_PATH)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(x) for x in vec] for vec in self._model.encode(list(texts))]

    def embed_query(self, text: str) -> list[float]:
        return [float(x) for x in self._model.encode([text])[0]]


def _judge_extra_body() -> dict[str, Any]:
    """评估调用的透传参数：默认关闭推理模型思考模式（不关时 glm-5 会把额度全花在 reasoning、content 为空）"""
    body: dict[str, Any] = dict(LLM_EXTRA_BODY or {})
    if EVAL_JUDGE_DISABLE_THINKING:
        body.setdefault("thinking", {"type": "disabled"})
    return body


def _judge_llm(judge_model: str):
    """评估用 LLM（Ragas 打分）：单独构造，不影响在线问答的业务模型实例与思考开关"""
    from langchain.chat_models import init_chat_model

    if judge_model == LLM_MODEL and not _judge_extra_body():
        # 与业务配置完全一致时直接复用全局实例，省一次连接池初始化
        from src.agent.llm import llm

        return llm
    return init_chat_model(
        model=judge_model,
        model_provider="openai",
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        temperature=LLM_TEMPERATURE,
        **({"extra_body": _judge_extra_body()} if _judge_extra_body() else {}),
    )


def _ragas_components(judge_model: str):
    """构造 ragas 所需的 LLM 与 Embeddings 包装"""
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    return LangchainLLMWrapper(_judge_llm(judge_model)), LangchainEmbeddingsWrapper(_BgeEmbeddings())


# ---------- 测试集 ----------

def _load_testset(testset: str | Path | None = None) -> list[dict[str, Any]]:
    """读取本地 JSON 测试集，补齐 ragas 所需字段并做最小校验"""
    rel = str(testset or EVAL_TESTSET_PATH)
    path = Path(rel) if os.path.isabs(rel) else PROJECT_ROOT / rel
    if not path.exists():
        raise FileNotFoundError(
            f"测试集不存在：{rel}（可先执行 .venv1\\Scripts\\python.exe tests\\build_ragas_testset.py 生成）"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload["samples"] if isinstance(payload, dict) else payload
    if not raw:
        raise ValueError(f"测试集为空：{rel}")

    samples: list[dict[str, Any]] = []
    for i, row in enumerate(raw, start=1):
        question = (row.get("question") or row.get("user_input") or "").strip()
        if not question:
            continue
        contexts = row.get("retrieved_contexts") or []
        if isinstance(contexts, str):
            contexts = [contexts]
        samples.append({
            "id": row.get("id") or f"rs-{i:02d}",
            "category": row.get("category") or "",
            "question": question,
            "answer": (row.get("answer") or row.get("response") or "").strip(),
            "retrieved_contexts": [str(c) for c in contexts],
            "ground_truth": (row.get("ground_truth") or row.get("reference") or "").strip(),
            "answer_type": row.get("answer_type"),
        })
    if not samples:
        raise ValueError(f"测试集无有效问题字段：{rel}")
    return samples


# ---------- 打分理由 ----------

REASON_PROMPT = """你是 RAG 评估解释专家。下面给出一条测试样本的原始信息与 Ragas 计算出的四项指标得分，
请逐项用中文解释「这个分数为什么是这样」，每条理由不超过 80 字，必须落到具体证据与内容上（引用关键短语），
不得复述指标定义，不得使用「总体良好」一类空话。

【用户问题】@@QUESTION@@
【检索到的知识库切片】
@@CONTEXTS@@
【系统回答】
@@ANSWER@@
【标准答案（ground truth）】
@@REFERENCE@@
【Ragas 得分】忠实度=@@FAITHFULNESS@@ 答案相关性=@@RELEVANCY@@ 上下文精确率=@@PRECISION@@ 上下文召回率=@@RECALL@@

只输出 JSON：
{"faithfulness":"理由","answer_relevancy":"理由","context_precision":"理由","context_recall":"理由"}"""


def _fill(template: str, **values: str) -> str:
    """占位符替换（提示词内含 JSON 花括号示例，不能用 str.format）"""
    out = template
    for key, value in values.items():
        out = out.replace(f"@@{key.upper()}@@", value)
    return out


def _parse_json(text: str) -> dict[str, Any] | None:
    """宽容解析模型输出的 JSON：剥代码围栏、截取首个花括号块"""
    if not text:
        return None
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _fmt(value: Any) -> str:
    """得分格式化：None/NaN → 「缺失」，其余两位小数"""
    if value is None:
        return "缺失"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "缺失"
    return "缺失" if f != f else f"{f:.2f}"


def _judge_chat(prompt: str, judge_model: str, *, max_tokens: int) -> str:
    """单次 Judge 调用（打分理由），返回消息正文"""
    resp = httpx.post(
        f"{LLM_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {LLM_API_KEY}"},
        json={
            "model": judge_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": max_tokens,
            **_judge_extra_body(),   # 含思考开关：与业务调用隔离，只影响评估
        },
        timeout=EVAL_JUDGE_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"].get("content") or ""


def _reason_call(
    sample: dict[str, Any], scores: dict[str, float | None], judge_model: str | None = None
) -> dict[str, str]:
    """调用 Judge 模型生成该样本的四项打分理由；失败返回占位文案，不抛出"""
    contexts = sample["retrieved_contexts"] or ["（本轮未检索到任何知识库切片）"]
    prompt = _fill(
        REASON_PROMPT,
        question=sample["question"],
        contexts="\n".join(f"- {c[:600]}" for c in contexts[:5]),
        answer=(sample["answer"] or "（空回答）")[:3000],
        reference=sample["ground_truth"] or "（测试集未提供标准答案）",
        faithfulness=_fmt(scores.get("faithfulness")),
        relevancy=_fmt(scores.get("answer_relevancy")),
        precision=_fmt(scores.get("context_precision")),
        recall=_fmt(scores.get("context_recall")),
    )
    judge = judge_model or EVAL_JUDGE_MODEL
    try:
        data = _parse_json(_judge_chat(prompt, judge, max_tokens=EVAL_JUDGE_MAX_TOKENS))
        if not data:
            # 强制思考端点会忽略关闭思考参数，reasoning 占满额度后正文为空 → 放宽上限重试一次
            data = _parse_json(
                _judge_chat(prompt, judge, max_tokens=max(EVAL_JUDGE_RETRY_MAX_TOKENS, EVAL_JUDGE_MAX_TOKENS))
            )
    except Exception as e:  # noqa: BLE001 理由缺失不应否定打分结果
        err = f"{type(e).__name__}: {str(e)[:120]}"
        return {k: f"（理由生成失败：{err}）" for k in METRICS}
    data = data or {}

    return {
        k: str(data.get(k) or "（模型未给出该项理由）").strip()[:400] for k in METRICS
    }


# ---------- 打分主流程 ----------

def _score_with_ragas(
    samples: list[dict[str, Any]], judge_model: str, max_workers: int | None = None
) -> list[dict[str, float | None]]:
    """执行 ragas.evaluate，返回与 samples 等长的得分列表（缺项为 None）"""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )
    from ragas.run_config import RunConfig

    ragas_llm, ragas_emb = _ragas_components(judge_model)
    dataset = Dataset.from_dict({
        "user_input": [s["question"] for s in samples],
        "retrieved_contexts": [s["retrieved_contexts"] for s in samples],
        "response": [s["answer"] for s in samples],
        "reference": [s["ground_truth"] for s in samples],
    })
    result = evaluate(
        dataset,
        metrics=[Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()],
        llm=ragas_llm,
        embeddings=ragas_emb,
        run_config=RunConfig(
            timeout=EVAL_TIMEOUT,
            max_retries=EVAL_MAX_RETRIES,
            max_workers=max(1, max_workers or EVAL_MAX_WORKERS),
        ),
    )
    df = result.to_pandas()

    scored: list[dict[str, float | None]] = []
    for i in range(len(samples)):
        row: dict[str, float | None] = {}
        for key in METRICS:
            value = df[key].iloc[i] if key in df.columns and i < len(df) else None
            try:
                f = float(value)
            except (TypeError, ValueError):
                f = None
            row[key] = None if f is None or math.isnan(f) else round(f, 4)
        scored.append(row)
    return scored


# ---------- 结果落盘 ----------

def _save_csv(items: list[dict[str, Any]], stamp: str, results_dir: str | Path | None = None) -> str:
    """评估明细保存为 CSV（utf-8-sig，Excel 直接打开不乱码），返回相对路径

    列：定位列 + 四项得分 + 综合得分 + 四项打分理由 + 系统回答 + 标准答案。
    """
    import pandas as pd

    rel = str(results_dir or EVAL_RESULTS_DIR)
    out_dir = Path(rel) if os.path.isabs(rel) else PROJECT_ROOT / rel
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"ragas_eval_{stamp}.csv"

    rows: list[dict[str, Any]] = []
    for it in items:
        row = {
            "样本ID": it["id"],
            "类别": it["category"],
            "问题": it["question"],
            "回答类型": it.get("answer_type"),
            "检索切片数": it["n_contexts"],
        }
        for key, label in METRICS.items():
            row[f"{label}({key})"] = _fmt(it["scores"][key])
        row[COMPOSITE_LABEL] = _fmt(it.get("composite"))
        for key, label in METRICS.items():
            row[f"{label}理由"] = (it.get("reasons") or {}).get(key, "")
        row["系统回答"] = it["answer"]
        row["标准答案"] = it["ground_truth"]
        rows.append(row)

    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")


def _save_json(payload: dict[str, Any], stamp: str, results_dir: str | Path | None = None) -> str:
    """完整结果另存 JSON（含逐样本得分/综合分/理由/系统回答/标准答案），便于二次分析"""
    rel = str(results_dir or EVAL_RESULTS_DIR)
    out_dir = Path(rel) if os.path.isabs(rel) else PROJECT_ROOT / rel
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"ragas_eval_{stamp}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")


def _md_cell(text: Any) -> str:
    """表格单元格文本转义：竖线与换行会破坏 Markdown 表格"""
    return _inline(text).replace("|", "\\|")


def _inline(text: Any) -> str:
    """压成单行：换行会断开 Markdown 列表项与标题"""
    return str(text if text is not None else "").replace("\r", " ").replace("\n", " ").strip()


def _block(text: Any, indent: str = "  > ") -> str:
    """长文本（系统回答/标准答案）在列表项内用引用块展示，保留原文换行"""
    body = str(text or "").strip() or "（空）"
    return "\n".join(f"{indent}{line}" for line in body.splitlines())


def _save_md(payload: dict[str, Any], stamp: str, results_dir: str | Path | None = None) -> str:
    """生成 Markdown 评估报告（总体指标 + 逐样本得分 + 弱项清单 + 逐样本明细），返回相对路径

    报告内容完全由本次评估结果构造，不手写数字，避免与 CSV/JSON 不一致。
    """
    rel = str(results_dir or EVAL_RESULTS_DIR)
    out_dir = Path(rel) if os.path.isabs(rel) else PROJECT_ROOT / rel
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"ragas_eval_{stamp}.md"

    o = payload["overall"]
    lines: list[str] = [
        "# Ragas 离线自动化评估报告",
        "",
        f"- 生成时间：{payload['generated_at']}",
        f"- 样本数：{o['n_samples']} 条 | Judge 模型：{o['judge_model']} | 耗时：{payload['duration_s']}s",
        f"- 测试集：{o['testset']}",
        f"- 综合得分权重：{_weights_text()}（缺失项不参与，按剩余权重归一）",
        "",
        "## 一、总体得分",
        "",
        "| 指标 | 平均分 | 最低 | 最高 | 有效样本 | 缺失 |",
        "|---|---|---|---|---|---|",
    ]
    for key in METRICS:
        m = o["metrics"][key]
        lines.append(
            f"| {m['label']}（{key}） | {_fmt(m['avg'])} | {_fmt(m['min'])} | {_fmt(m['max'])} "
            f"| {m['n']} | {m['missing']} |"
        )
    c = o["composite"]
    lines.append(
        f"| **{COMPOSITE_LABEL}**（加权） | **{_fmt(c['avg'])}** | {_fmt(c['min'])} | {_fmt(c['max'])} "
        f"| {c['n']} | {c['missing']} |"
    )

    lines += [
        "",
        "## 二、逐样本得分",
        "",
        "| 样本ID | 类别 | 问题 | 忠实度 | 答案相关性 | 上下文精确率 | 上下文召回率 | 综合得分 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for it in payload["items"]:
        sc = it["scores"]
        lines.append(
            f"| {it['id']} | {_md_cell(it['category'])} | {_md_cell(it['question'])} "
            f"| {_fmt(sc['faithfulness'])} | {_fmt(sc['answer_relevancy'])} "
            f"| {_fmt(sc['context_precision'])} | {_fmt(sc['context_recall'])} "
            f"| {_fmt(it.get('composite'))} |"
        )

    # 弱项清单：按综合得分升序取前 5，并指出拖分项（四项中最低的那项）
    weak = sorted(
        [it for it in payload["items"] if it.get("composite") is not None],
        key=lambda it: it["composite"],
    )[:5]
    weak_n = min(5, len([it for it in payload["items"] if it.get("composite") is not None]))
    lines += [
        "",
        f"## 三、弱项清单（综合得分最低 {weak_n} 条）",
        "",
        "| 样本ID | 问题 | 综合得分 | 最弱项 | 该弱项得分 |",
        "|---|---|---|---|---|",
    ]
    for it in weak:
        worst_key = min(
            (k for k in METRICS if it["scores"][k] is not None),
            key=lambda k: it["scores"][k],
            default=None,
        )
        lines.append(
            f"| {it['id']} | {_md_cell(it['question'])} | {_fmt(it.get('composite'))} "
            f"| {METRICS[worst_key] if worst_key else '—'} "
            f"| {_fmt(it['scores'][worst_key]) if worst_key else '—'} |"
        )

    lines += ["", "## 四、逐样本明细（得分与理由 + 系统回答 + 标准答案）", ""]
    for it in payload["items"]:
        sc = it["scores"]
        lines += [
            f"### {it['id']} {_inline(it['question'])}",
            "",
            f"- 类别：{it['category']} | 检索切片：{it['n_contexts']} 条 | 回答类型：{it.get('answer_type') or '—'}",
            f"- 得分：忠实度 {_fmt(sc['faithfulness'])}｜答案相关性 {_fmt(sc['answer_relevancy'])}｜"
            f"上下文精确率 {_fmt(sc['context_precision'])}｜上下文召回率 {_fmt(sc['context_recall'])}｜"
            f"**综合 {_fmt(it.get('composite'))}**",
        ]
        reasons = it.get("reasons") or {}
        if reasons:
            lines.append("- 打分理由：")
            for key in METRICS:
                lines.append(f"  - **{METRICS[key]}**（{_fmt(sc[key])}）：{_inline(reasons.get(key)) or '（无理由）'}")
        lines += [
            "- 系统回答：",
            _block(it["answer"]),
            "- 标准答案：",
            _block(it["ground_truth"]),
            "",
        ]

    lines += [
        "---",
        "",
        f"产物：CSV `{payload.get('csv_path')}` | JSON `{payload.get('json_path') or '未开 --json'}`",
        "",
        "指标口径：忠实度看回答是否均有检索切片支撑（防幻觉），答案相关性看回答与问题的贴合度，"
        "上下文精确率看检索切片中的有效信息占比与排序，上下文召回率看标准答案要点是否被检索覆盖。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")


def run_evaluation(
    *,
    testset: str | Path | None = None,
    results_dir: str | Path | None = None,
    limit: int | None = None,
    with_reasons: bool = True,
    judge_model: str | None = None,
    max_workers: int | None = None,
    concurrency: int | None = None,
    dump_json: bool = False,
    dump_md: bool | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """跑一次完整 Ragas 评估：加载测试集 → 四指标打分 → 生成打分理由 → 算综合得分 → 落盘结果

    Returns:
        结果字典：overall（四指标平均分与加权综合得分）+ items（逐样本得分/理由/回答/标准答案）
        + csv_path / json_path / md_path
    """
    def log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    if not EVAL_ENABLED:
        raise RuntimeError("评估功能已被配置关闭（EVAL_ENABLED=false，可在 .env 打开）")

    started = dt.datetime.now()
    rel_testset = str(testset or EVAL_TESTSET_PATH)
    judge = (judge_model or EVAL_JUDGE_MODEL).strip()

    log("① 加载测试集……")
    samples = _load_testset(rel_testset)
    if limit is not None and limit > 0:
        samples = samples[:limit]
    empty_ctx = [s["id"] for s in samples if not s["retrieved_contexts"]]
    if empty_ctx:
        log(f"  ! 以下样本无检索切片，忠实度/精确率/召回率会偏低：{', '.join(empty_ctx)}")
    total = len(samples)
    log(f"  样本 {total} 条 | Judge {judge} | 测试集 {rel_testset}")

    log("② Ragas 四指标打分中（逐条调用 Judge，耗时与样本数成正比）……")
    _apply_compat_stubs()
    _ensure_event_loop()
    scores = _score_with_ragas(samples, judge, max_workers)

    reasons: list[dict[str, str]] = [{} for _ in samples]
    if with_reasons:
        workers = max(1, concurrency or EVAL_JUDGE_CONCURRENCY)
        log(f"③ 生成逐项打分理由（并发 {workers}）……")
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_reason_call, s, sc, judge): idx for idx, (s, sc) in enumerate(zip(samples, scores))
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    reasons[idx] = fut.result()
                except Exception as e:  # noqa: BLE001 单样本理由失败不影响整体
                    reasons[idx] = {k: f"（理由生成失败：{e}）" for k in METRICS}
                done += 1
                log(f"  理由 {done}/{total}：{samples[idx]['id']}")
    else:
        log("③ 已跳过打分理由生成（--no-reasons）")

    log("④ 汇总结果并保存 CSV / Markdown 报告……")
    items: list[dict[str, Any]] = []
    for s, sc, rsn in zip(samples, scores, reasons):
        items.append({
            "id": s["id"],
            "category": s["category"],
            "question": s["question"],
            "answer_type": s.get("answer_type"),
            "n_contexts": len(s["retrieved_contexts"]),
            "scores": sc,
            "composite": _composite(sc),
            "reasons": rsn,
            "answer": s["answer"],
            "ground_truth": s["ground_truth"],
        })

    metric_summary: dict[str, dict[str, Any]] = {}
    for key in METRICS:
        vals = [it["scores"][key] for it in items if it["scores"][key] is not None]
        metric_summary[key] = {
            "label": METRICS[key],
            "avg": round(sum(vals) / len(vals), 4) if vals else None,
            "min": round(min(vals), 4) if vals else None,
            "max": round(max(vals), 4) if vals else None,
            "n": len(vals),
            "missing": len(items) - len(vals),
        }

    # 综合得分汇总：逐样本先加权，再对样本取平均（避免单条缺失改变整体含义）
    comp_vals = [it["composite"] for it in items if it["composite"] is not None]
    composite_summary = {
        "label": COMPOSITE_LABEL,
        "avg": round(sum(comp_vals) / len(comp_vals), 4) if comp_vals else None,
        "min": round(min(comp_vals), 4) if comp_vals else None,
        "max": round(max(comp_vals), 4) if comp_vals else None,
        "n": len(comp_vals),
        "missing": len(items) - len(comp_vals),
        "weights": dict(EVAL_COMPOSITE_WEIGHTS),
    }

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    payload: dict[str, Any] = {
        "overall": {
            "n_samples": len(items),
            "metrics": metric_summary,
            "composite": composite_summary,
            "judge_model": judge,
            "testset": rel_testset,
        },
        "items": items,
        "csv_path": _save_csv(items, stamp, results_dir),
        "generated_at": _now(),
        "duration_s": round((dt.datetime.now() - started).total_seconds(), 1),
    }
    if dump_json:
        payload["json_path"] = _save_json(payload, stamp, results_dir)
    if dump_md if dump_md is not None else EVAL_MD_REPORT:
        payload["md_path"] = _save_md(payload, stamp, results_dir)
    log(f"  明细 CSV：{payload['csv_path']}（含四项得分、综合得分、理由、系统回答、标准答案）")
    return payload


# ---------- 终端报表 ----------

def _print_report(payload: dict[str, Any], *, show_reasons: bool) -> None:
    """把结果打成中文终端报表：整体平均分 + 逐样本得分 + 逐样本理由"""
    o = payload["overall"]
    print("\n" + "=" * 72)
    print(f"Ragas 自动化评估结果  样本 {o['n_samples']} 条 | Judge {o['judge_model']} | 耗时 {payload['duration_s']}s")
    print(f"测试集 {o['testset']}")
    print("=" * 72)

    print("\n【整体平均分】")
    for key in METRICS:
        m = o["metrics"][key]
        rng = f"区间 {m['min']:.2f}~{m['max']:.2f}" if m["min"] is not None else "无有效得分"
        miss = f"，缺失 {m['missing']}" if m["missing"] else ""
        print(f"  {METRICS[key]:<12}{key:<18}{_fmt(m['avg']):>8}   有效 {m['n']} 条{miss}  {rng}")
    c = o["composite"]
    crng = f"区间 {c['min']:.2f}~{c['max']:.2f}" if c["min"] is not None else "无有效得分"
    print(f"  {COMPOSITE_LABEL:<11}{'composite':<18}{_fmt(c['avg']):>8}   有效 {c['n']} 条  {crng}")
    print(f"  权重：{_weights_text()}")

    print("\n【逐样本得分】")
    head = f"  {'ID':<8}{'忠实度':>8}{'相关性':>8}{'精确率':>8}{'召回率':>8}{'综合':>8}  问题"
    print(head)
    print("  " + "-" * 74)
    for it in payload["items"]:
        sc = it["scores"]
        print(
            f"  {it['id']:<8}{_fmt(sc['faithfulness']):>10}{_fmt(sc['answer_relevancy']):>10}"
            f"{_fmt(sc['context_precision']):>10}{_fmt(sc['context_recall']):>10}"
            f"{_fmt(it.get('composite')):>10}  {it['question'][:28]}"
        )

    if show_reasons:
        print("\n【逐样本打分理由】")
        for it in payload["items"]:
            print(f"\n  ▍{it['id']} {it['question']}")
            for key in METRICS:
                reason = (it.get("reasons") or {}).get(key) or "（无理由）"
                print(f"    {METRICS[key]}（{_fmt(it['scores'][key])}）：{reason}")

    print(f"\n明细已保存 CSV：{payload['csv_path']}（utf-8-sig，Excel 可直接打开）")
    if payload.get("json_path"):
        print(f"完整结果 JSON：{payload['json_path']}")
    if payload.get("md_path"):
        print(f"评估报告 Markdown：{payload['md_path']}")
    print("=" * 72)


# ---------- 命令入口 ----------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.agent.ragas_eval",
        description="Ragas 离线自动化评估：读本地测试集，跑忠实度/答案相关性/上下文精确率/上下文召回率四项指标"
                    "与加权综合得分，生成打分理由并保存 CSV / Markdown 报告（只有执行本命令才会评估，无自动执行）",
    )
    p.add_argument("--testset", default=None, help=f"测试集 JSON 路径（默认取配置 EVAL_TESTSET_PATH={EVAL_TESTSET_PATH}）")
    p.add_argument("--results-dir", default=None, help=f"结果保存目录（默认取配置 EVAL_RESULTS_DIR={EVAL_RESULTS_DIR}）")
    p.add_argument("--limit", type=int, default=None, help="只评估前 N 条样本（冒烟用），默认全量")
    p.add_argument("--no-reasons", action="store_true", help="不调用 Judge 生成打分理由，只输出得分")
    p.add_argument("--json", action="store_true", help="除 CSV 外，另存完整结果 JSON（含得分/综合分/理由/系统回答/标准答案）")
    p.add_argument("--no-md", action="store_true", help="不生成 Markdown 评估报告（默认生成，与 CSV 同目录同时间戳）")
    p.add_argument("--judge-model", default=None, help="临时指定 Judge 模型名（默认取配置 EVAL_JUDGE_MODEL）")
    p.add_argument("--max-workers", type=int, default=None, help="Ragas 打分并发数（默认取配置 EVAL_MAX_WORKERS）")
    p.add_argument("--concurrency", type=int, default=None, help="理由生成并发数（默认取配置 EVAL_JUDGE_CONCURRENCY）")
    return p


def main(argv: list[str] | None = None) -> int:
    """命令入口：打印进度与结果报表，成功返回 0，失败打印原因返回 1"""
    # ragas / langchain-community 大量弃用提示与评估结果无关，屏蔽后终端报表才可读
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", message=".*langchain-community.*")
    # Windows GBK 控制台下避免中文输出编码报错
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    args = build_parser().parse_args(argv)

    print("启动 Ragas 自动化评估……", flush=True)
    try:
        payload = run_evaluation(
            testset=args.testset,
            results_dir=args.results_dir,
            limit=args.limit,
            with_reasons=not args.no_reasons,
            judge_model=args.judge_model,
            max_workers=args.max_workers,
            concurrency=args.concurrency,
            dump_json=args.json,
            dump_md=False if args.no_md else None,
        )
    except Exception as e:  # noqa: BLE001 评估失败只打印原因并退出，不抛栈扰民
        print(f"\n评估失败：{type(e).__name__}: {str(e)[:500]}", flush=True)
        return 1

    _print_report(payload, show_reasons=not args.no_reasons)
    return 0


if __name__ == "__main__":
    sys.exit(main())
