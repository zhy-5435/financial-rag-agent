"""Agent 层配置：加载项目根目录 .env，集中管理模型接入与检索服务参数"""
import json
import os

from dotenv import load_dotenv

from src.config import PROJECT_ROOT

# .env 位于项目根目录（不入库，存放大模型 API Key 等敏感信息）
load_dotenv(PROJECT_ROOT / ".env")

# ---------- LLM 配置（OpenAI 兼容接口，腾讯云 MaaS） ----------
LLM_MODEL = os.getenv("LLM_MODEL", "GLM-5.1")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://tokenhub.tencentmaas.com/v1")
# .env 中变量名为 API_KEY（兼容 OPENAI_API_KEY 命名）
LLM_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))

# ---------- 关闭推理模型「思考模式」（降低首 token 延迟）----------
# **默认关闭**（缺省 true）：思考链是规划步与作答步最主要的额外时延，关闭后端到端 TTFT 显著下降
# （deepseek 系列实测端到端 TTFT 33.9s→16.1s，见 docs/首Token耗时评测报告.md）。
# 仅对支持该参数的模型生效（实测 GLM-5.3-flash / flashx / glm-5-turbo / deepseek-v4 系列支持，
# GLM-5.1 强制思考会忽略此参数）；不支持的端点忽略该参数，不报错。
# 注意：关思考可能改变工具调用倾向（历史回归见报告 6.4），换模型版本后须重跑工具调用率回归；
# 如需临时回到「思考开」形态对比，在 .env 显式设 LLM_DISABLE_THINKING=false。
LLM_DISABLE_THINKING = os.getenv("LLM_DISABLE_THINKING", "true").lower() in ("1", "true", "yes")


def _build_extra_body() -> dict | None:
    """组装透传给 OpenAI 兼容接口的 extra_body：thinking 开关 + 通用 JSON 扩展合并"""
    body: dict = {}
    if LLM_DISABLE_THINKING:
        body["thinking"] = {"type": "disabled"}
    raw = os.getenv("LLM_EXTRA_BODY")  # 预留：额外自定义参数（JSON 字符串）
    if raw:
        try:
            body.update(json.loads(raw))
        except json.JSONDecodeError:
            pass
    return body or None


LLM_EXTRA_BODY = _build_extra_body()

# ---------- L2 检索服务 ----------
SEARCH_API_BASE = os.getenv("SEARCH_API_BASE", "http://127.0.0.1:8000")
SEARCH_TOP_K = int(os.getenv("SEARCH_TOP_K", "5"))

# ---------- 金融场景可靠性 ----------
# 相似度低于阈值的切片不作为作答依据；全部低于阈值时明确拒答，禁止编造条款与费率
# 0.4 为 bge-base-zh-v1.5 实测校准值（正例最低 0.4354，沿用 0.5 会误杀正确结果）
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.4"))

# ---------- 意图识别 ----------
# 分类置信度低于阈值的非 kb_qa 意图一律回落知识库分支（保留现有拒答底线，宁可拒答不乱分流）
INTENT_CONFIDENCE_THRESHOLD = float(os.getenv("INTENT_CONFIDENCE_THRESHOLD", "0.6"))

# ---------- 实时行情工具（M4） ----------
# 行情接口硬超时（秒）：单标的直连正常 <1s，超时即降级，不阻断其他分支
QUOTE_TIMEOUT_SECONDS = float(os.getenv("QUOTE_TIMEOUT_SECONDS", "10"))
# 错名/错别字 LLM 推断命中阈值：低于该值视为猜不准，走未识别话术而非强行作答
QUOTE_FUZZY_MIN_CONFIDENCE = float(os.getenv("QUOTE_FUZZY_MIN_CONFIDENCE", "0.6"))
# 「事前确认」触发信号：LLM 推断出的标准名再跑 suggest3，命中 ≥2 个不同标的/市场
# 即视为存在真实歧义，推候选让用户点选；唯一命中则直接作答 + 事后声明。
# （不再依赖 LLM 自报置信度分档——实测其值二元极化，要么 ≥0.95 要么直接 null，灰带为空集。）
# 确认条最多展示的候选标的数（suggest3 白名单命中的前 N 条，去重）
QUOTE_CONFIRM_CANDIDATE_LIMIT = int(os.getenv("QUOTE_CONFIRM_CANDIDATE_LIMIT", "5"))
# 行情数据源与免责固定话术（按要求必须注明来源且不得加 LLM 生成）
QUOTE_SOURCE_NAME = "新浪财经公开行情接口（hq.sinajs.cn 实时快照）"
QUOTE_DISCLAIMER = "行情数据来自第三方公开接口，仅供参考、不构成任何投资建议，可能有延迟，请以交易所公布数据为准；本条回答非 L1 知识库权威内容。"

# 关键词/规则预分类：命中 news/quote/chitchat 直接短路分流、跳过意图 LLM；未命中仍走 LLM
INTENT_RULES_ENABLED = os.getenv("INTENT_RULES_ENABLED", "true").lower() in ("1", "true", "yes")

# ---------- Agent Loop（统一自主循环） ----------
# 循环最大步数（reason→act→observe 一轮记 1 步）；=1 近似旧单轮，作应急降级
AGENT_MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "5"))
# 整轮墙钟上限（秒）：超限则 finalize 输出部分结论 + 未完成声明，杜绝无限循环
AGENT_LOOP_TIMEOUT_SECONDS = float(os.getenv("AGENT_LOOP_TIMEOUT_SECONDS", "60"))
# 整轮 token 预算（0=不限，预留）
AGENT_TOKEN_BUDGET = int(os.getenv("AGENT_TOKEN_BUDGET", "0"))
# Agent Loop 总开关：true=走统一自主循环；false=应急回退到「一步内按意图预过滤工具」的近单轮形态
AGENT_LOOP_ENABLED = os.getenv("AGENT_LOOP_ENABLED", "true").lower() in ("1", "true", "yes")
# 真流式开关（P0-1）：true=终答轮 LLM token 经 stream writer 直透 SSE（首 token 不再等全链路跑完）；
# false=应急回退旧形态（定稿后整段分片推送，协议兼容：无 custom 事件，前端零改动）
AGENT_STREAM_TOKENS = os.getenv("AGENT_STREAM_TOKENS", "true").lower() in ("1", "true", "yes")

# ---------- 接口限流（滑动窗口 QPS + Agent 长任务并发信号量，见 rate_limit.py） ----------
# 总开关（压测/调试时可短时下掉，不建议生产关闭）
RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").lower() in ("1", "true", "yes")
# 配 redis://host:port/db 启用 Redis 后端（多实例共享计数、Lua 原子）；留空=进程内存后端
RATE_LIMIT_REDIS_URL = os.getenv("RATE_LIMIT_REDIS_URL", "redis://127.0.0.1:6379/0")
# 单用户 QPS：窗口秒数内最多 N 次请求（所有受控端点共用此限额，按端点独立计数）
RATE_LIMIT_QPS_MAX = int(os.getenv("RATE_LIMIT_QPS_MAX", "5"))
RATE_LIMIT_QPS_WINDOW_SECONDS = float(os.getenv("RATE_LIMIT_QPS_WINDOW_SECONDS", "10"))
# 单用户同时进行的 Agent 长任务上限（/chat、/chat/stream，核心保命机制）：
# 防一人刷堆 70s 长请求打满服务
RATE_LIMIT_USER_CONCURRENCY = int(os.getenv("RATE_LIMIT_USER_CONCURRENCY", "2"))
# 全局总并发兜底闸
RATE_LIMIT_GLOBAL_CONCURRENCY = int(os.getenv("RATE_LIMIT_GLOBAL_CONCURRENCY", "80"))
# 并发槽租约（秒）：须大于单次最长任务（墙钟 60 + 补救生成 45），崩溃/断连残留由到期自动回收
RATE_LIMIT_SLOT_LEASE_SECONDS = float(os.getenv("RATE_LIMIT_SLOT_LEASE_SECONDS", "180"))

# ---------- 工具层容错（超时 / 重试 / 熔断） ----------
# 分工具硬超时（秒）：知识库检索复用 L2 网络调用、网搜复用 WEB_SEARCH_TIMEOUT、行情复用 QUOTE_TIMEOUT_SECONDS
TOOL_TIMEOUT_KB = float(os.getenv("TOOL_TIMEOUT_KB", "30"))
TOOL_TIMEOUT_WEB = float(os.getenv("TOOL_TIMEOUT_WEB", str(int(os.getenv("WEB_SEARCH_TIMEOUT", "10")) * 2)))
TOOL_TIMEOUT_QUOTE = float(os.getenv("TOOL_TIMEOUT_QUOTE", str(os.getenv("QUOTE_TIMEOUT_SECONDS", "10"))))
# 单次工具调用失败后的最大重试次数（不含首次尝试）
TOOL_MAX_RETRIES = int(os.getenv("TOOL_MAX_RETRIES", "1"))
# 指数退避基数（秒），第 n 次重试等待 base * 2**n + 抖动
TOOL_RETRY_BASE_SECONDS = float(os.getenv("TOOL_RETRY_BASE_SECONDS", "0.5"))
# 熔断：连续失败达阈值打开，打开后 reset 秒内直接返回「工具不可用」观测，到期转半开探活
BREAKER_FAILURE_THRESHOLD = int(os.getenv("BREAKER_FAILURE_THRESHOLD", "3"))
BREAKER_RESET_SECONDS = float(os.getenv("BREAKER_RESET_SECONDS", "20"))

# ---------- 联网搜索（权威信源）----------
# 调用方（后续接入的 Agent / 检索流程）通过 web_search 工具触发联网检索：
# Tavily Search API 以 include_domains 限定在下列白名单官方域名内检索，
# 原文打包进 Prompt 并引用 URL；白名单同时用于结果二次过滤，杜绝不可控来源。
WEB_SEARCH_TIMEOUT = int(os.getenv("WEB_SEARCH_TIMEOUT", "10"))          # 单次 HTTP 超时（秒）
WEB_SEARCH_MAX_PAGES = int(os.getenv("WEB_SEARCH_MAX_PAGES", "5"))       # 打包进 Prompt 的网页数上限
WEB_SEARCH_MAX_CHARS = int(os.getenv("WEB_SEARCH_MAX_CHARS", "1500"))    # 每篇正文截断字数
# Tavily API Key（https://tavily.com 注册获取；未配置时联网搜索优雅降级为空结果）
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()
# Tavily 检索深度：basic 快；advanced 正文提取更全（消耗 2 倍额度）
TAVILY_SEARCH_DEPTH = os.getenv("TAVILY_SEARCH_DEPTH", "advanced")
# 财经资讯媒体（新闻、解读、快讯，证监会指定信息披露媒体）
WEB_MAIN_SOURCES = {
    "中证网": "cs.com.cn",
    "证券时报网": "stcn.com",
    "财联社": "cls.cn",
    "新华财经": "cnfin.com",
    "上海证券报·中国证券网": "cnstock.com",
    "证券日报网": "zqrb.cn",
    "金融时报·中国金融新闻网": "financialnews.com.cn",
    "经济参考报": "jjckb.cn",
}

# 政策信源（监管、政府、交易所官方，原文、公告、统计数据）
WEB_POLICY_SOURCES = {
    "中国人民银行": "pbc.gov.cn",
    "中国证监会": "csrc.gov.cn",
    "国家统计局": "stats.gov.cn",
    "国家金融监督管理总局": "nfra.gov.cn",
    "国家外汇管理局": "safe.gov.cn",
    "中华人民共和国财政部": "mof.gov.cn",
    "上海证券交易所": "sse.com.cn",
    "深圳证券交易所": "szse.cn",
    "北京证券交易所": "bse.cn",
    "中国金融期货交易所": "cffex.com.cn",
    "上海期货交易所": "shfe.com.cn",
    "大连商品交易所": "dce.com.cn",
    "郑州商品交易所": "czce.com.cn",
    "广州期货交易所": "gfex.com.cn",
    "中国证券投资基金业协会": "amac.org.cn",
    "中国证券业协会": "sac.net.cn",
    "中国证券登记结算": "chinaclear.cn",
    "中证指数有限公司": "csindex.com.cn",
}


# ---------- 提示词注入纵深防御（S3.3：内容分级 + 结构化隔离 + 输出审计） ----------
# 总开关：true=对进入 prompt 的外部内容做定界符包裹与信任级标注（结构化隔离）
PROMPT_GUARD_ENABLED = os.getenv("PROMPT_GUARD_ENABLED", "true").lower() in ("1", "true", "yes")
# 低信任源（web/tool/MCP）正文的指令模式剥离开关：命中「忽略指令/输出系统提示词」等样句替换为占位符；
# 知识库切片属中信任源，**不受本项影响**（正文必须逐字保留以守「数字与原文一致」的金融不变量）
PROMPT_GUARD_FILTER_UNTRUSTED = os.getenv("PROMPT_GUARD_FILTER_UNTRUSTED", "true").lower() in ("1", "true", "yes")
# 输出审计开关：回答生成后规则检测 system prompt 泄漏 / 证据外 URL 与联系方式 / 指令执行迹象
OUTPUT_AUDIT_ENABLED = os.getenv("OUTPUT_AUDIT_ENABLED", "true").lower() in ("1", "true", "yes")
# 命中审计后是否拦截并替换为安全话术（false=仅记录安全事件不改答案，供灰度观察期使用）
OUTPUT_AUDIT_INTERCEPT = os.getenv("OUTPUT_AUDIT_INTERCEPT", "true").lower() in ("1", "true", "yes")
# 判定「逐字泄漏机密提示词」的最小连续匹配长度（字符）：低于此长度视为正常术语复用。
# 24 由实测定标：能力自介逐字复述铁律里的域名词枚举（如「条款费率期限风险等级监管规则等」）最长≈20，
# 应容忍；而真实提示词外泄（整段复述铁律/注入免疫）连续重合达 60+，远超此值仍可稳拦。
PROMPT_LEAK_MIN_LEN = int(os.getenv("PROMPT_LEAK_MIN_LEN", "24"))
# 安全事件是否落库审计（l1_memory.db 独立表 security_events，与权威库物理隔离）
SECURITY_EVENTS_PERSIST = os.getenv("SECURITY_EVENTS_PERSIST", "true").lower() in ("1", "true", "yes")

# ---------- 在线自动评估（每次问答调用后异步质量评估，落库 l1_memory.db，见 online_eval.py） ----------
# 总开关：false 时 /chat 与 /chat/stream 不触发任何评估；评估全程旁路，不阻塞响应、不影响主链路
ONLINE_EVAL_ENABLED = os.getenv("ONLINE_EVAL_ENABLED", "true").lower() in ("1", "true", "yes")
# LLM judge 采样率 0~1：1.0=每轮都评；调低可省 judge 额度。规则级护栏审计不受采样影响，每轮必做
ONLINE_EVAL_JUDGE_RATE = float(os.getenv("ONLINE_EVAL_JUDGE_RATE", "1.0"))
# judge 并发上限（进程级信号量）：防高 QPS 时评估调用打满 LLM 额度与连接池
ONLINE_EVAL_JUDGE_CONCURRENCY = int(os.getenv("ONLINE_EVAL_JUDGE_CONCURRENCY", "2"))
# 单次 judge 调用超时（秒）：超时记 judge_error，不重试不阻塞后续评估
ONLINE_EVAL_JUDGE_TIMEOUT = float(os.getenv("ONLINE_EVAL_JUDGE_TIMEOUT", "30"))
# answer 落库截断长度（字符）：评估明细回查只需可辨识前缀，防超大 blob
ONLINE_EVAL_ANSWER_MAX = int(os.getenv("ONLINE_EVAL_ANSWER_MAX", "2000"))

# ---------- Ragas 离线评估（命令行手动触发：python -m src.agent.ragas_eval，见 ragas_eval.py）----------
# 总开关：false 时执行评估命令直接报错退出，用于生产环境屏蔽评估入口
EVAL_ENABLED = os.getenv("EVAL_ENABLED", "true").lower() in ("1", "true", "yes")
# 测试集路径（相对项目根目录）：含 question / retrieved_contexts / answer / ground_truth 四要素
EVAL_TESTSET_PATH = os.getenv("EVAL_TESTSET_PATH", "data/eval/ragas_testset.json")
# 评估结果保存目录（相对项目根目录）：每次评估落一份 CSV（文件名含时间戳），可用 --results-dir 覆盖
EVAL_RESULTS_DIR = os.getenv("EVAL_RESULTS_DIR", "data/eval/results")
# Judge 模型：Ragas 打分与「打分理由」生成所用模型，留空=复用业务 LLM_MODEL
EVAL_JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", "").strip() or LLM_MODEL
# 理由生成并发数（Ragas 打分由 ragas 自身线程池控制，见 EVAL_MAX_WORKERS）
EVAL_JUDGE_CONCURRENCY = int(os.getenv("EVAL_JUDGE_CONCURRENCY", "2"))
# 评估调用是否关闭推理模型「思考模式」：实测 glm-5 不关思考时 800 token 全归 reasoning、content 为空，
# 打分理由解析不到；关掉后 176 token 直接输出 JSON。对强制思考端点无效（自动走放宽额度重试）
EVAL_JUDGE_DISABLE_THINKING = os.getenv("EVAL_JUDGE_DISABLE_THINKING", "true").lower() in ("1", "true", "yes")
# 单次评估调用的 token 上限：首次 / 解析失败后的重试上限（强制思考模型首次可能被 reasoning 占满）
EVAL_JUDGE_MAX_TOKENS = int(os.getenv("EVAL_JUDGE_MAX_TOKENS", "1500"))
EVAL_JUDGE_RETRY_MAX_TOKENS = int(os.getenv("EVAL_JUDGE_RETRY_MAX_TOKENS", "3000"))
# 单次理由调用超时（秒）：超时该样本理由记为「生成失败」，不阻断整体评估（强制思考模型重试一次会叠加此预算）
EVAL_JUDGE_TIMEOUT = float(os.getenv("EVAL_JUDGE_TIMEOUT", "90"))
# Ragas 单次 LLM 调用超时（秒）与最大重试次数
EVAL_TIMEOUT = int(os.getenv("EVAL_TIMEOUT", "180"))
EVAL_MAX_RETRIES = int(os.getenv("EVAL_MAX_RETRIES", "3"))
# Ragas 内部并发线程数：与在线问答共用 LLM 额度，默认压到 4 防打满端点
EVAL_MAX_WORKERS = int(os.getenv("EVAL_MAX_WORKERS", "4"))


def _default_composite_weights() -> dict[str, float]:
    """综合得分默认权重：四项等权（等价于简单平均，不预设哪一项更重要）"""
    return {
        "faithfulness": 0.25,
        "answer_relevancy": 0.25,
        "context_precision": 0.25,
        "context_recall": 0.25,
    }


def _parse_composite_weights(raw: str) -> dict[str, float]:
    """解析综合得分权重：非法 JSON / 未知指标名 / 全为 0 均回退到四项等权

    保证配置写错不会中断评估（评估是可重复运行的批处理，容错优先于报错）。
    金融合规场景想抬高防幻觉权重时，将 faithfulness 调大即可（如 0.4，其余相应调小）。
    """
    try:
        data = json.loads(raw) if raw else {}
        weights = {k: float(v) for k, v in data.items()}
    except Exception:  # noqa: BLE001 配置容错：回落默认值
        return _default_composite_weights()
    known = _default_composite_weights()
    weights = {k: v for k, v in weights.items() if k in known and v > 0}
    return weights or _default_composite_weights()


# 综合得分权重（JSON 字符串，.env 可覆盖），例：{"faithfulness":0.4,"answer_relevancy":0.2,"context_precision":0.2,"context_recall":0.2}
EVAL_COMPOSITE_WEIGHTS: dict[str, float] = _parse_composite_weights(
    os.getenv("EVAL_COMPOSITE_WEIGHTS", "").strip()
)
# 是否自动生成 Markdown 评估报告（与 CSV/JSON 同目录同时间戳），--no-md 可临时关
EVAL_MD_REPORT = os.getenv("EVAL_MD_REPORT", "true").lower() in ("1", "true", "yes")

# ---------- L3 Agent HTTP 服务 ----------
AGENT_API_HOST = os.getenv("AGENT_API_HOST", "0.0.0.0")
AGENT_API_PORT = int(os.getenv("AGENT_API_PORT", "8001"))
# 允许跨域访问前端来源，逗号分隔；生产同域反代时可置空关闭 CORS
AGENT_CORS_ORIGINS = [
    o.strip()
    for o in os.getenv("AGENT_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
    if o.strip()
]
