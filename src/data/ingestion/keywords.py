"""关键词自动抽取：LLM 主路 + 规则兜底，为混合召回填 keywords 字段

链路价值（下游零改动，写完库即生效）：
- FTS 路：sqlite_client._fts_body 把 keywords 拼进索引 body，命中词 BM25 加权提升；
  FTS 中文按单字切分，正文已有的词天然可命中——关键词的真正增量在**正文未出现的
  同义词/通行简称**（正文写「上海证券交易所科创板」，用户搜「科创板」），此路由 LLM 路补齐；
- 向量路：milvus search 命中后从主库 detail 取 keywords 返回上层（get_knowledge_detail_by_ids），
  SQLite 填上即带出；
- keywords 是索引辅助字段，不参与作答引用，不触碰「原文逐字」防幻觉不变量。

两档设计：llm=主路（需 .env LLM 接入）；rule=纯本地兜底（实体词+章节词+高频 n-gram，
确定性、离线可复现，--check 空跑与常驻演练用这档保证零联网副作用）；off=不抽取。
llm 路任何一批失败（网络/解析）只让该批退 rule，绝不阻断入库。
"""
import json
import re
from collections import Counter

from src.data.ingestion.config import (
    KEYWORDS_LLM_BATCH_ITEMS,
    KEYWORDS_MIN_GRAM_FREQ,
    KEYWORDS_MODE,
    KEYWORDS_TOP_N,
    KW_LLM_API_KEY,
    KW_LLM_BASE_URL,
    KW_LLM_EXTRA_BODY,
    KW_LLM_MODEL,
    KW_LLM_TEMPERATURE,
)
from src.data.ingestion.schemas import ChunkDraft

# 字母实体（ETF/SSE/LOF 等代码与缩写）与数值指标（20%、0.5%、2.4.2 等，数字须成串带小数/百分号）
_ENTITY_RE = re.compile(r"[A-Za-z][A-Za-z0-9._+\-/]{1,15}|\d+(?:\.\d+)*%")
# 连续中文串（n-gram 候选来源），2 字起
_CN_RUN_RE = re.compile(r"[\u4e00-\u9fff]{2,}")
# 单字停用集：虚词/功能词/高频泛词，含该字的 n-gram 一律不取（宁缺毋滥，误杀「当日」类低频词可接受）
_STOP_CHARS = set(
    "的了和与及或是在为对按据关于等中以其该并将或被把向已未可应当得着过个此项之类至即若则都又只"
    "能从新老跟随会就还有着于第各每"
)
# 章节段前缀：「第X章/节/条/部分/编/篇」与数字编号「2.4.2」，剥离后剩主题词
_CHAPTER_PREFIX_RE = re.compile(r"^第[一二三四五六七八九十百千0-9]+(?:章|节|条|部分|编|篇)?\s*")
_NUM_PREFIX_RE = re.compile(r"^[0-9]+(?:\.[0-9]+)*\s*")

_KW_PROMPT = """你是金融知识库的索引员。为下列编号的文档切片各抽取检索关键词，用于混合召回的关键词路加料。
要求：
1. 每个切片 3~8 个词或短语；禁止整句、禁止解释；
2. 优先：主题实体、市场/产品/机构名称、关键数值与比例（如 20%、T+1）、章节主题词；
3. 可补 1~2 个正文未直接出现但用户检索时可能使用的同义词或通行简称（如正文写「上海证券交易所科创板」可给「科创板」）；
4. 只依据切片内容判断，不得引入切片之外的事实。
仅输出一个 JSON 对象，键为切片编号字符串，值为关键词数组，不要输出其他内容。
形如 {{"0": ["词1", "词2"], "1": ["词3"]}}

切片列表：
{items}"""


def _clean_words(words, cap: int = KEYWORDS_TOP_N) -> list[str]:
    """词表清洗：仅留 2~16 字符的纯文本词，去重保序，截上限"""
    out: list[str] = []
    for w in words:
        if not isinstance(w, str):
            continue
        w = w.strip().strip("\"'“”")
        if 2 <= len(w) <= 16 and w not in out:
            out.append(w)
        if len(out) >= cap:
            break
    return out


def _heading_terms(heading_path: str) -> list[str]:
    """章节路径各级剥编号取主题词（首级即文档名，本身是强关键词）"""
    out: list[str] = []
    for seg in (heading_path or "").split(">"):
        s = _NUM_PREFIX_RE.sub("", _CHAPTER_PREFIX_RE.sub("", seg.strip())).strip()
        if len(s) >= 2 and s not in out:
            out.append(s)
    return out


def _entity_terms(text: str) -> list[str]:
    """字母代码与数值指标实体（纯数字串太泛不取，带%/字母才有检索指向性）"""
    return _ENTITY_RE.findall(text or "")


def _ngram_terms(text: str, keep: int = 4) -> list[str]:
    """无分词库的高频中文术语兜底：2~4 字 n-gram 计数，剔停用字，
    按 频次×长度 排序；与已选词字符重叠率≥0.6 的滑窗变体（如选了「深圳证券」后的
    「圳证券交」）跳过，取前 keep 个"""
    counts: Counter[str] = Counter()
    for run in _CN_RUN_RE.findall(text or ""):
        for n in (2, 3, 4):
            for i in range(len(run) - n + 1):
                g = run[i:i + n]
                if not (set(g) & _STOP_CHARS):
                    counts[g] += 1
    cand = sorted(((g, c) for g, c in counts.items() if c >= KEYWORDS_MIN_GRAM_FREQ),
                  key=lambda x: (-x[1] * len(x[0]), -len(x[0])))
    picked: list[str] = []
    for g, _ in cand:
        gs = set(g)
        if any(g in p or p in g or len(gs & set(p)) / len(gs) >= 0.6 for p in picked):
            continue  # 子串或滑窗交叠变体不再重复占位
        picked.append(g)
        if len(picked) >= keep:
            break
    return picked


def extract_rule(text: str, heading_path: str = "", max_n: int = KEYWORDS_TOP_N) -> list[str]:
    """规则路（确定性、离线）：章节主题词 > 实体/数值词 > 高频 n-gram 术语，合并去重截断"""
    return _clean_words(_heading_terms(heading_path) + _entity_terms(text) + _ngram_terms(text), max_n)


# ---------- LLM 路 ----------

_llm = None  # 惰性单例：off/rule 模式与测试环境不构造客户端，零网络副作用


def _get_llm():
    global _llm
    if _llm is None:
        from langchain.chat_models import init_chat_model  # 延迟导入：rule 路不拉 langchain
        kwargs = {}
        if KW_LLM_EXTRA_BODY:
            kwargs["extra_body"] = KW_LLM_EXTRA_BODY
        _llm = init_chat_model(
            model=KW_LLM_MODEL, model_provider="openai",
            base_url=KW_LLM_BASE_URL, api_key=KW_LLM_API_KEY,
            temperature=KW_LLM_TEMPERATURE, **kwargs,
        )
    return _llm


def _parse_reply(content: str, n: int) -> dict[int, list[str]]:
    """容错解析模型回复：剥代码围栏取首个 {...} 块 → 按批内编号对齐；坏输出返空由上层退规则路"""
    m = re.search(r"\{.*\}", content or "", re.S)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}
    if not isinstance(obj, dict):
        return {}
    out: dict[int, list[str]] = {}
    for i in range(n):
        words = obj.get(str(i))
        if isinstance(words, list):
            cleaned = _clean_words(words)
            if cleaned:
                out[i] = cleaned
    return out


def llm_extract(texts: list[str], headings: list[str] | None = None) -> tuple[list[list[str]], int]:
    """LLM 批量抽取（每批 KEYWORDS_LLM_BATCH_ITEMS 片）。返回 (每片关键词, 回退规则路的批数)：
    整批调用/解析失败 → 该批逐片 extract_rule；个别编号缺失 → 该片单独兜底。"""
    headings = headings or [""] * len(texts)
    results: list[list[str]] = [[] for _ in texts]
    degraded_batches = 0
    for start in range(0, len(texts), KEYWORDS_LLM_BATCH_ITEMS):
        batch = texts[start:start + KEYWORDS_LLM_BATCH_ITEMS]
        items = "\n".join(f"[{j}] {t}" for j, t in enumerate(batch))
        try:
            resp = _get_llm().invoke(_KW_PROMPT.format(items=items))
            parsed = _parse_reply(getattr(resp, "content", ""), len(batch))
            if not parsed:
                raise ValueError("回复无可解析 JSON（截断或格式漂移）")
        except Exception:  # 网络/初始化/解析任一失败：本批整体退规则路，不阻断入库
            degraded_batches += 1
            parsed = {}
        for j in range(len(batch)):
            gi = start + j
            results[gi] = parsed.get(j) or extract_rule(batch[j], headings[gi])
    return results, degraded_batches


def assign_keywords(drafts: list[ChunkDraft], mode: str | None = None) -> str:
    """就意为合格切片填 draft.keywords；返回实际执行说明（供日志）。
    mode：llm/rule/off，缺省读环境变量 INGEST_KEYWORDS_MODE（默认 llm）。
    LLM 路建连失败整体降级 rule 并在返回串中说明原因。"""
    mode = (mode or KEYWORDS_MODE).lower()
    if mode == "off" or not drafts:
        return "off"
    if mode == "llm":
        try:
            _get_llm()  # 提前建连验证配置可用（api_key 缺失等在此暴露）
        except Exception as e:
            return f"rule(llm不可用:{type(e).__name__})"
        kws, degraded = llm_extract([d.content for d in drafts], [d.heading_path for d in drafts])
        for d, words in zip(drafts, kws):
            d.keywords = words
        note = f"llm({len(drafts)}片)"
        if degraded:
            note += f"，{degraded}批退规则"
        return note
    for d in drafts:
        d.keywords = extract_rule(d.content, d.heading_path)
    return f"rule({len(drafts)}片)"
