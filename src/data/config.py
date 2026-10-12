"""L1 数据层专有配置：向量库 Collection、Embedding 模型与检索参数"""
from src.config import PROJECT_ROOT

MILVUS_COLLECTION = "l1_core_knowledge"
EMBEDDING_MODEL_NAME = "BAAI/bge-base-zh-v1.5"
# 本地部署的模型目录（由 HF 缓存经 SentenceTransformer.save() 导出的标准目录）：
# 存在则优先从本地路径加载，完全不依赖 HuggingFace 缓存位置与网络
EMBEDDING_MODEL_DIR = PROJECT_ROOT / "models" / "bge-base-zh-v1.5"
EMBEDDING_MODEL_PATH = str(EMBEDDING_MODEL_DIR) if EMBEDDING_MODEL_DIR.exists() else EMBEDDING_MODEL_NAME
VECTOR_DIM = 768  # bge-base-zh-v1.5 固定输出维度
TOP_K = 3         # 默认返回切片数

# 双路混合检索（向量语义召回 + SQLite FTS5 关键词召回）
CANDIDATE_TOP_K = 10  # 每路召回的候选切片数，两路去重合并后再截取 TOP_K
# 权重经 14 条评测集网格搜索校准：两路得分先各自 min-max 归一化再加权，
# 关键词路作排序辅助信号、不宜主导（孪生规则文本关键词得分易达 1.0），故降权至 0.2。
VECTOR_WEIGHT = 0.8   # 综合得分中归一化向量相似度的权重
KEYWORD_WEIGHT = 0.2  # 综合得分中归一化关键词得分的权重

# 轻量重排序（Reranker）：双路召回得到候选集后，用交叉编码器对候选做二次精排
# 模型体量小（bge-reranker-base，约 1.1GB），本地可 CPU 运行；已在 HF 缓存则离线加载
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"
# 本地部署目录优先（models/bge-reranker-base），无本地目录时回退 HF 模型名（命中缓存/镜像）
RERANKER_MODEL_DIR = PROJECT_ROOT / "models" / "bge-reranker-base"
RERANKER_MODEL_PATH = str(RERANKER_MODEL_DIR) if RERANKER_MODEL_DIR.exists() else RERANKER_MODEL_NAME
# 模型已下载至本地 models/bge-reranker-base（离线加载不联网）。曾以 14 条旧评测集测得
# 精排收益为 0 而默认关闭；现网疑难查询（跨实体错排 P-02 等）需交叉编码器判别，
# 以 33 条 v2 评测集开启态复测（见 docs/检索指标测试报告.md 第十一节）后启用。
RERANK_ENABLED = True     # 开启即走交叉编码器精排；模型缺失/加载失败自动降级回双路加权排序
RERANK_CANDIDATE_K = 20   # 参与精排的候选上限（取加权排序后的前 K 条送交叉编码器重排）

# 近重复多样性折叠（P1 检索层策略）：跨文档逐字/近似孪生切片（如多份公司章程示例的
# 同一条决议条款）会挤占有限的 TOP_K 头部槽位，把真正不同的内容压到 rank4+。
# 加权排序后、截断前做贪心折叠：与已保留切片文本近重复（字符 2-gram Jaccard 达阈值）
# 的候选被跳过，让不同内容顶进头部。零模型、零额外时延、只影响头部排布不改召回。
# 实测（33 条评测集）：两条 top3 未命中均非逐字 duplicate 占位所致（章程多版本为等价
# 条款非逐字相同、华夏基金 vs 华夏理财为跨实体错排），折叠对其无效且使 R-10 头部序微降
# （MRR 0.870→0.865），故默认关闭；作为多版本逐字副本场景的可选防御开关保留。
DEDUP_ENABLED = False           # 关闭即回到不折叠的原排序
DEDUP_JACCARD = 0.82            # 近重复判定阈值（越高越保守，只折叠几乎逐字相同的副本）
DEDUP_POOL_K = 20               # 参与折叠比较的候选池上限（仅对排序前 K 条做两两比对）
