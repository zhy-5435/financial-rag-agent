# 首 Token 耗时（TTFT）评测与优化报告

> 测于 2026-10-06，环境：Windows / RTX 4060 Laptop GPU / L2 检索服务 :8000 / L3 Agent 服务 :8001 /
> LLM：deepseek-v4-flash（腾讯云 MaaS，OpenAI 兼容接口，`.env` 现状 `LLM_DISABLE_THINKING=false`）。
> 口径：TTFT = 前端发起 `POST /chat/stream` → 收到第一个 `token` 事件的时刻；每场景 3 轮取中位数。
>
> **终态更新（2026-10-07）**：模型升级 DeepSeek-V4.1-Flash，默认关思考 + prepare 写 SQLite 全异步化 +
> 上下文窗口/摘要阈值收紧 + 后台主动压缩；真网 KB 单轮 TTFT **33.9s→1.77s**（见「七、终态优化」）。

## 一、结论先行

1. **知识库问答的 TTFT 实测中位 33.9 秒，且这不是「打字慢」，是「首字就没出来」**：
   所有 token 事件在同一毫秒内批量到达——现在的流式是**假流式**（终答生成完毕后分片推送），
   用户从发送问题到看见第一个字，等的是**整条链路的全部时间**。
2. **94% 的等待来自两次串行远程 LLM 调用**（规划步 ~3.2s + 作答步阻塞生成 ~27s），
   检索环节只占 ~2s，数据库/审计/限流等本地开销均可忽略。
3. 两个收益最大的优化方向：
   - **改真流式**（终答 LLM 的 token 直接透传 SSE）：TTFT 33.9s → 预计 **6~9s**，不动任何模型配置；
   - **关闭思考模式**（`LLM_DISABLE_THINKING=true`）：实测端到端 TTFT 33.9s → **16.1s**（-53%），
     但需先回归验证作答质量与工具调用行为，二者叠加后 TTFT 预计可压到 **4~6s**。

## 二、链路解剖：首 token 前到底发生了什么

```
用户发送问题
  │
  ├─ ① prepare：SQLite 落库 user 消息 + 组装系统提示/多轮历史      ~0.3-1.4s
  │
  ├─ ② reason（第 1 次 LLM 远程调用，阻塞式）：
  │      bind_tools 决定调用哪些工具，返回 tool_calls 才推 plan 事件  ~3.2s（首事件 3.1-4.6s）
  │
  ├─ ③ act：search_knowledge → HTTP 调 L2 /search（双路混合检索）    ~1.3-2.0s
  │      （实测一次规划并发发出 2 个检索调用，同毫秒完成，不额外加时延）
  │
  ├─ ④ reason（第 2 次 LLM 远程调用，阻塞式，思考模式开）：
  │      读入系统提示 + 检索证据块，生成 1000+ 字终答 ★最大头        ~9.5-27.8s
  │
  ├─ ⑤ finalize：输出审计（规则）+ SQLite 落库助手回答 + 组装 meta    <0.1s
  │
  └─ ⑤' meta + token×N + done 在同一毫秒批量冲出 —— token 只是把
        已定稿文本按 48 字分片（graph._chunks），打印机效果纯属展示层
```

关键代码事实（`src/agent/graph.py` `aask_stream`）：走 LangGraph
`stream_mode="updates"`（按节点完成推送），`token` 事件只在 `finalize` 节点的
update 落地后由 `_chunks(delta["answer"])` 分片产出——**图内没有任何一处把 LLM
的增量 token 接到 SSE 上**，所以 TTFT 恒等于全链路总时长（实测 done 与首个
token 仅差 1-2ms 可证）。

## 三、实测数据

### 3.1 组件隔离计时（单进程直调，不经 HTTP 服务层）

| 环节 | 中位 | 最小 | 最大 | 说明 |
|---|---:|---:|---:|---|
| LLM 规划步（bind_tools + 阻塞 ainvoke） | 3226ms | 2698 | 3657 | 短 prompt，3 次均正确返回 tool_calls |
| L2 `/search` HTTP（双路混合检索全程） | 1491ms | — | — | embedding 已在 GPU（RTX 4060）+ 查询向量 LRU 缓存 |
| LLM 作答步·流式·**思考开（现状）** | TTFT 2851ms | 1898 | 9495 | 短 prompt；总时长中位 5806ms，chunk ~290 |
| LLM 作答步·流式·**思考关** | TTFT 1496ms | 1492 | 1632 | 总时长中位 4306ms；TTFT -48%，且抖动几乎消失（9495→1632） |

### 3.2 端到端 SSE 事件时序（`POST /chat/stream`，3 轮中位）

| 场景 | 首事件(plan/meta) | **TTFT(token)** | 总时长 | token 到达形态 |
|---|---:|---:|---:|---|
| 知识库问答（思考开，现状） | 4610ms | **33916ms** | 33918ms | 同一毫秒批量到达（假流式实锤） |
| 知识库问答（思考关） | 3113ms | **16078ms** | 16079ms | 同上 |
| 闲聊（思考开） | 5030ms | 5030ms | 5031ms | 同上 |
| 闲聊（思考关） | 3435ms | 4456ms | 4457ms | 同上 |
| 多轮会话 KB 问答（思考开，同 session 第 2 问） | 3441ms | **37435ms** | 37437ms | 模型二次 replan + 再检索，链路多跑一轮 |

### 3.3 耗时占比拆解（现状：KB 单轮，TTFT 33.9s）

| 分段 | 耗时 | 占比 |
|---|---:|---:|
| prepare + 规划步 LLM（→plan 事件 @4.6s） | ~4.6s | 14% |
| act 检索（→step 事件 @6.6s） | ~2.0s | 6% |
| **作答步 LLM 阻塞生成 + finalize（→meta/token @33.9s）** | **~27.3s** | **80%** |
| finalize 之后的推送 | ~0.002s | ≈0% |

两次远程 LLM 合计 ≈31.9s，**占 TTFT 的 94%**。作答步比 3.1 节组件实测（5.8s）慢很多，
原因是真实链路里 prompt 带着长系统提示 + 5 条证据块（content + original_text 逐字），
prefill 更大、终答更长、思考链也更长。

## 四、瓶颈与优化建议（按 ROI 排序）

### P0-1 假流式 → 真流式：把终答 LLM 的 token 直接接到 SSE 上

- **问题**：架构性最大头。前端体验为「白屏 34 秒后瞬间刷完全文」，`done.elapsed_ms`
  与 TTFT 相等即证据。
- **方案**：`aask_stream` 改用 `astream_events`（或 `stream_mode="messages"`），当 reason
  节点产出的 AIMessage 无 tool_calls（即终答轮）时，把增量 token 实时下发；
  `meta` 事件（intent/answer_type/sources）改在轮次开始前按已确定的工具轨迹**先行推送**。
- **与防幻觉护栏的冲突及处理**：现设计「先审计、后上屏」，流式意味着文本先上屏。
  建议两段式：① token 流先以「草稿」形态渲染；② 流尾跑 `audit_answer` 规则检测，
  命中拦截时下发既有 `security` 事件，前端整气泡替换为安全话术（规则审计本身 <10ms，
  用户几乎无感）。拒答/行情逐字/固定话术分支保持现状整段推送，**三条防幻觉不变量不回退**。
- **预期收益**：TTFT 33.9s → plan(3.2)+act(1.5)+终答模型 TTFT(2.9~5s) ≈ **7.6~9.7s**；
  叠加 P0-2 后 ≈ **5~6s**。改动集中在 graph 出口与前端渲染策略，不动模型与检索。

### P0-2 关闭思考模式（LLM_DISABLE_THINKING=true）

- **实测**：端到端 TTFT 33.9s → 16.1s（-53%）；组件级 TTFT 2851→1496ms，
  最坏抖动 9495→1632ms（长尾基本抹平）；作答步总时长 -26%。
- **前提**：`config.py` 注释既有结论「GLM-5.1 强制思考会忽略该参数」，本次 deepseek-v4-flash
  实测**参数生效**。但金融场景不能只看速度：需先跑一轮检索/e2e 回归（`tests/phase4_e2e.py`
  及既评评测集）确认作答质量、引用规范与工具调用率无退化，再切默认值。
- **风险留痕**：本次测试中发现一次「关闭思考后未调工具直接作答」的样本，经复核为探针
  经 PowerShell 管道传参中文乱码导致的无效样本（模型把乱码当无效输入），**正式复测两轮
  均正常触发工具调用**；但该现象提示「关思考可能改变工具调用倾向」，回归时应重点盯这一点。

### P1-1 收敛 Agent Loop 空转（多轮场景 TTFT 37.4s）

- **问题**：多轮会话中模型拿到首轮检索结果后仍 replan 再查一轮（事件时序：
  plan→step→plan→step→meta），每多一轮 = +3~8s LLM + 1.5s 检索。
- **方案**：① 系统提示强化「search_knowledge 返回有效切片后必须直接作答，不得重复检索」；
  ② 或对同一工具的同类查询做去重（act 层拦截与上轮 args 基本一致的重复调用）；
  ③ `AGENT_MAX_STEPS=5` 偏松，kb_qa 场景 2 步足够，可考虑按意图域收窄预算。
- **预期收益**：多轮 TTFT 37.4s → ~17s（对齐单轮），与真流式叠加后 → ~6s。

### P2-1 L2 检索链路瘦身（1.5s，占 6%）

- embedding 已上 GPU 且有查询 LRU 缓存，**大头不在这**；可抠的点：
  ① `milvus_store.search`/`get_similarity_by_ids` 每次都 `load_collection`，collection
  已加载时可跳过；② 工具层每次 `httpx.AsyncClient(...)` 新建连接，改为进程级共享
  client（连接池复用，省 TCP 握手 ~百毫秒）；③ MilvusLite gRPC 心跳限流是已知配置，
  本机开销可接受。
- **更划算的其实是「喂给 LLM 的证据瘦身」**：top_k 5→3、`original_text` 与 `content`
  截断（证据块是终答轮 prefill 膨胀的直接原因，对 P0-1 的「终答 TTFT」有放大效应）。

### P2-2 首字节体验兜底（低成本过渡方案）

- 真流式落地前的过渡：`prepare` 完成即推一条零延迟 `ack` 事件（现在首事件要等
  3.1~4.6s 的规划步跑完），前端 pending 骨架从「请求发出」就已有进度条，加上
  plan/step 时间线已能先行渲染，可显著缓解「卡死感」。此项不减少 TTFT 本身。

### 可忽略项（实测排除）

- finalize 输出审计、SQLite 落库、Redis 限流（本机 3 次往返）、SSE 编码：均在数十毫秒内，
  不构成瓶颈；前端 rAF 合帧与只读流解析无积压问题。

## 五、优化叠加后的预期

| 状态 | KB 单轮 TTFT | 闲聊 TTFT | 多轮 KB TTFT |
|---|---:|---:|---:|
| 现状（思考开 + 假流式） | 33.9s | 5.0s | 37.4s |
| 仅关思考 | 16.1s（实测） | 4.5s（实测） | ~19s |
| 仅真流式 | ~7.6-9.7s | ~1.5-2s | ~10s |
| 真流式 + 关思考 + Loop 收敛 | **~4-6s** | **~1.5-2s** | **~6s** |

**建议实施顺序**：P0-1 真流式（纯代码改造、零质量风险）→ P0-2 关思考（需质量回归）→
P1-1 Loop 收敛 → P2 项按需跟进。

## 六、落地实施记录（2026-10-06，P0-1/P1/P2 全落地；P0-2 经真网回归后回退）

### 6.1 代码变更

| 项 | 落地点 | 说明 |
|---|---|---|
| P0-1 真流式 | `graph.reason` 改 `bound.astream` + `get_stream_writer` 透传增量；`aask_stream` 改 `stream_mode=["updates","custom"]` | 「草稿-定稿对齐」协议：工具规划轮草稿到达时 `reset(tool_round)` 回收；定稿与草稿不一致（拒答/行情逐字/审计拦截/预算追加尾注）时 `reset(finalized)` 后重推权威分片；一致则零重推。新增应急开关 `AGENT_STREAM_TOKENS`（false=回退旧形态，S4 断言验证）；前端 `useChat` 新增 onReset 清空草稿。三条防幻觉不变量不回退 |
| P0-2 关思考 | `.env` `LLM_DISABLE_THINKING`：曾切 `true`，现行模型 0731 版回归后**回退 `false`** | 旧版 deepseek-v4-flash 回归正常（6.3）；0731 版关思考后 bind_tools 调用率衰减（8 样本仅 1 次真实 tool_calls，余者文字模拟检索，幻觉风险）→ 按「质量优先」纪律回退；真流式已覆盖关思考的主要 TTFT 收益，见 6.4 |
| P1 Loop 收敛 | `graph.act` 同参拦截 + `prompts` 能力段强化 + `route_observe` 行情成功直达 finalize | 同一轮会话内同工具同参数成功调用不再重复执行（回注提示型 ToolMessage 保消息配对，失败仍允许重试）；`AGENT_MAX_STEPS` 保持 5 未收窄（拦截+提示词已覆盖实测空转模式） |
| P2 细节 | `_render_kb_block` 证据瘦身（original_text 被 content 包含时不重复渲染，实测全库 21105 条 valid 切片包含率 100%，证据 prefill 近乎砍半）；`tools.py` 进程级共享 `httpx.AsyncClient`（lifespan 关停释放）；`milvus_client._ensure_loaded()` 读取路径 load 一次化 | |

### 6.2 离线协议断言（假 LLM，零出网，临时脚本已按规范清理）

S1 工具轮草稿作废 + 终答定稿=草稿零重推✓；S2 无证据拒答经 `finalized reset` 整气泡替换为固定话术✓；
S3 同参重复调用 `run_tool` 实际仅执行 1 次且轨迹标记 `skipped_duplicate`✓；
S4 关开关回退旧形态（meta 先于 token）✓；S5 行情成功直达 finalize（无补答轮，quoted 逐字文本经
reset 后整段推送）✓；回声护栏另有 S6a（单轮回声→纠偏）/S6b（双轮回声→拒答不外泄）/S7（判定函数）
断言（试验期验证，真网兜底证据见 6.4）。

### 6.3 真网 e2e 回归实测（`POST /chat/stream`，思考关 + 真流式，单次）

| 场景 | TTFT | 总时长 | 真流式证据 | 质量断言 |
|---|---:|---:|---|---|
| T1 KB 问答「公募基金收取哪些费用」 | **6.8s**（现状 33.9s，**-80%**） | 13.2s | meta 前草稿 token 660 个，分批到达 | generated ✓，溯源 5 条 ✓，引用编号 ✓ |
| T2 同会话换问法追问 | **2.4s**（多轮现状 37.4s） | 14.9s | meta 前 849 token；规划轮草稿经 `reset(tool_round)` 回收 | generated ✓，仅 1 个规划轮、无同参重复调用（P1 收敛生效） |
| T5 寒暄 | 2.2s | 3.3s | meta 前 115 token | chatted ✓ |

工具调用率：关思考后两轮 KB 问答均正常触发 `search_knowledge`，未复现「直接作答不调工具」样本（P0-2
重点盯防项通过）。（注：本节为旧版 deepseek-v4-flash 的历史基线，该组合后续在 0731 版上不复现，已回退，见 6.4）

### 6.4 现行模型终态回归（2026-10-06，deepseek-v4-flash-0731 + 思考开 + 精简提示词）

端点内模型升级为 **deepseek-v4-flash-0731**；中途试验的其它模型配置已清理，最终回归在本组合完成。
`POST /chat/stream` 五项均 PASS（单次）：

| 场景 | TTFT | 总时长 | 质量断言 |
|---|---:|---:|---|
| T1 KB 问答 | 4.6s | 12.8s | generated✓、溯源 10 条✓、引用编号✓、草稿先于 meta（真流式）✓、1 规划轮 |
| T2 同会话追问 | 1.1s | 11.6s | 终态无内部标记✓（草稿回声被护栏纠偏，见下） |
| T3 严格库外（L2 召回 0.29~0.34，全 <0.4 阈值） | 3.4s | **90.0s** | **refused✓、拒答话术逐字等于固定文本✓**、4 规划轮（有界长尾，见下） |
| T4 实时行情 | 2.1s | 2.1s | quoted✓、逐字 markdown 含来源/免责✓（直达终态无补答轮） |
| T5 寒暄 | 1.2s | 3.3s | chatted✓ |

**P0-2 回退依据**：0731 版 `thinking=disabled` 下 bind_tools 8 样本仅 1 次产出真实 tool_calls，
其余以「（正在调用实时行情工具……）」类文字模拟检索并自行续写结果（幻觉风险），另见思考开时偶发
tool_call 标记以纯文本漏出（parsed=false）→ `LLM_DISABLE_THINKING` 回退 `false`；TTFT 收益已由
真流式覆盖（上表实测）。

**上下文脚手架回声（跨模型存在，代码层护栏兜底）**：0731 版多轮追问同样会把注入的历史工具块
（`【上文工具/检索结果】`/`〔…已精简〕`）写进终答轮草稿。5 次追问实测：草稿回声 2/5 次触发，全部被
`finalize` 的 `_is_context_echo` 护栏拦截（纠偏重合成），**终态泄漏 0/5**、作废文本由 `reset(finalized)`
整气泡替换。提示词保持精简（不为单一模型加复述禁止条），防护只留在代码层。

**提示词精简后的多轮重查率**：移除试验期加入的「追问必须重查」能力条后，5 样本实测 3/5 重新调用
`search_knowledge`（generated）、2/5 凭历史直答（chatted，内容仍为上轮有据检索的汇总、无编造）；
重查率较加提示词时下降，属速度/严谨与 token 成本的权衡，直答样本经逐一审查可接受。

**残余发现（与既有模型一致，非回归）**：T3 严格库外总时长 90.0s——对无命中问题用不同 query 反复
replan（非 P1 拦截的同参重复），墙钟（60s）+步数（5）预算兑底（observe 步间检查、在途步跑完可超一个
步长）；TTFT 不受影响（草稿 3.4s 上屏），属有界长尾。收敛选项（kb_qa 收窄 `AGENT_MAX_STEPS` 或 route
前墙钟预判）有损合法多路改写重查，未实施（质量优先）。

### 6.5 实测 vs 预期（终态）

真流式（思考开）下多场景 TTFT 优于预期区间：KB 单轮 TTFT 1.1~4.6s（预期 4~6s），多轮追问
1.1~3.9s（预期 ~6s），行情直达总时长 2.1s，闲聊 1.2~3.3s；原假流式下「token 与 done 同毫秒」
的情形已消除（meta 前均有分批到达的草稿 token）。残余：T3 库外拒答 90s 有界长尾（属总时长、
非 TTFT，预算已兑底）与 6.4 记录的多轮直答采样差异；P0-2 关思考在现行模型上不可用（6.4 回退依据）。

## 七、终态优化（2026-10-07，现行模型 DeepSeek-V4.1-Flash：默认关思考 + prepare 写全异步化 + 上下文瘦身）

模型升级为 **DeepSeek-V4.1-Flash**（同端点换版本），在其上重开 P0-2「关思考」并按三项 prepare/上下文
优化压缩 TTFT。真网端到端 `POST /chat/stream` 实测见 7.2。

### 7.1 代码变更

| 项 | 落地点 | 说明 |
|---|---|---|
| 默认关思考 | `config.py` `LLM_DISABLE_THINKING` 缺省由 `false` 改 **`true`**；`.env` 同步 `true` | `_build_extra_body` 注入 `thinking:{type:disabled}`，规划步与作答步两次 LLM 均不再产思考链。**工具调用率回归**：V4.1 关思考下 KB/行情 5/5 样本正常触发 `search_knowledge`/`search_realtime_quote`（`answer_type=generated/quoted`、溯源完整），**未复现** 6.4 中 0731 版「关思考→文字模拟检索」的退化 → 支持默认关思考上线（换模型仍须重跑该回归） |
| prepare 写 SQLite 全异步化 | 新增 `memory/store/write_queue.py`（单线程 FIFO 后台写队列）；`memory_manager` 加 `begin_turn_user`/`remember_*_async`/`record_security_event_async`；`graph.prepare` 改 `async def` | user 落库、act 的 tool 落库、finalize 的助手回答 + 安全事件落库**全部摘出关键路径**：提交队列即返回（写序由单线程 FIFO 保证，seq 单调无碰撞），prepare 不再串磁盘 I/O；助手 `message_id` 预生成后立即可回 `meta`，不等落库。为正确性，prepare 入口先 `drain()` 排空上一轮写入再拿号，摘要落库改走同一队列（`summarizer._persist_compression`）避免与预分配 seq 碰撞 |
| 缩小 token 限制窗口 | `memory/config.py` `CONTEXT_MAX_TOKENS`：32000 → **16000**（工作窗口，远小于真实模型窗口，仅用于更早触发历史压缩→作答步 prefill 更小） | 单轮无历史不受影响；多轮历史更早压缩，终答轮 prefill 随之收窄 |
| 提早开始摘要 | ① `TIER_WARNING` 0.70→**0.55**（`TIER_GLOBAL` 0.82→0.75、`TIER_TRIM` 0.92→0.90）：局部摘要更早介入；② 新增**后台主动压缩**：`graph._schedule_proactive_compact` 在每轮响应结束后起守护线程跑 `memory_manager.compact_session`，按当前体量提前摘要落库 | 把可能的**阻塞式摘要 LLM 调用**从下一轮 prepare 的关键路径挪到轮后后台（用户阅读窗口内完成），下一轮 prepare 直接命中 direct 档；direct 档无副作用（不额外调 LLM）。开关 `MEMORY_PROACTIVE_COMPACT`（默认 true） |

### 7.2 真网 e2e 实测（`POST /chat/stream`，V4.1-Flash + 默认关思考 + 上述三项优化）

口径同 §三：TTFT = 发起 → 首个 `token` 事件；每场景多轮取中位；`draft_before_meta>0` 即真流式坐实。

| 场景 | 首事件 | **TTFT(token)** | 总时长 | 到达形态/质量 |
|---|---:|---:|---:|---|
| KB 单轮「公募基金收取哪些费用？禁止哪些行为？」(×3) | 1769ms | **1769ms** | 11052ms | 草稿先于 meta（350~420 token 分批到），generated、溯源正常、规划 1~2 轮 |
| 闲聊「你好，你能做什么？」(×3) | 1669ms | **1669ms** | 2517ms | chatted，0 规划轮 |
| 多轮同会话第 2 问（×2） | 1722ms | **1722ms** | 4150ms | chatted（凭历史直答、0 规划轮，见 7.3）；Q1 预热 TTFT 1511ms/总 9447ms |
| 实时行情「600519 最新价」(×2) | 1634ms | **2190ms** | 1874ms | quoted 逐字、1 规划轮（行情工具直达 finalize） |

### 7.3 实测对照与观察

- **对比本报告历史基线**（同一 KB 问句）：假流式·思考开 33916ms（§3.2）→ 真流式·思考开 4600ms（§6.4）
  → **真流式·关思考 + prepare 异步化 + 上下文瘦身 1769ms**，TTFT 再降 **~62%**、较原始形态降 **~95%**。
- **两次 LLM 仍是主开销**：关思考后规划步 TTFT 首事件 1.7s 内即到，作答步 prefill 随窗口收窄而缩短；
  prepare/落库等本地开销已从关键路径摘出（探针实测单进程闲聊整轮 2.97s，与端到端闲聊 2.5s 量级一致，
  说明磁盘写不再是可观测瓶颈）。
- **多轮追问走「凭历史直答」而非重查**（chatted、0 规划轮）：与 §6.4 记录的采样差异一致——提示词精简后
  部分追问模型选择汇总上轮有据检索直答、内容无编造；此项属速度/严谨与 token 成本的权衡，非本次改动引入的
  回归。后台主动压缩与更紧窗口可能轻微强化该倾向（历史更早收敛为要点摘要），如需强约束重查可在追问域收窄
  预算或补提示词，本次按「质量优先、不过度收紧」未强行改判。
- **关思考工具调用率**：本轮 5/5 工具型样本（3 KB + 2 行情）均正常产出真实 `tool_calls`，行情逐字、KB 溯源不变量
  未回退；这是把 P0-2 从「0731 回退」重新定为「V4.1 默认关」的关键依据。

### 7.4 实施顺序与回退

落地顺序：默认关思考（质量已回归）→ prepare 写全异步化 → 上下文窗口/摘要阈值收紧 → 后台主动压缩。
应急回退：`LLM_DISABLE_THINKING=false` 回思考开；`MEMORY_PROACTIVE_COMPACT=false` 关后台压缩；
`CONTEXT_MAX_TOKENS`/`CONTEXT_TIER_*` 按 `.env` 还原；写队列与 `prepare async` 为纯 I/O 调度改造，语义与同步串行等价
（探针已验 seq 单调/无碰撞/跨轮可读），无需独立开关。

## 附录：测量方法与复现

- 探针为单进程分步计时脚本（临时脚本已按规范清理）：组件层直调 `llm.bind_tools().ainvoke`、
  `model.astream`（分别测思考开/关的 TTFT 与总时长）、HTTP 调 L2 `/search`；
  端到端层以 httpx 流式消费 `/chat/stream`，逐事件记录到达毫秒时刻。
- 样句：`公募基金收取哪些费用？禁止哪些行为？`（kb）、`你好，你能做什么？`（闲聊）；
  每场景 3 轮，报告取中位/最小/最大。
- 局限：单台开发机 + 公网 MaaS 端点，LLM 时延存在网络抖动（现状 TTFT 21.9~34.4s 区间即
  思考链长度差异所致）；样本 3 轮偏少，结论以「量级与占比」为准，绝对值仅供基线。
- 复现：启动 L2/L3 后，用等价探针脚本即可（注意 Windows 下勿用 PowerShell 管道传中文参数，
  会乱码污染样句，应直接运行 UTF-8 脚本文件）。
