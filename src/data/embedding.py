"""BGE 中文向量服务：查询侧加检索指令前缀，文档侧不加；有 CUDA 独显时模型自动上 GPU

查询向量按原文本做有界 LRU 缓存（P0 检索时延优化）：一次混合检索内 `milvus_store.search`
与 `get_similarity_by_ids` 会对同一查询各调一次 `encode_query`，缓存后同查询只做一次前向编码，
且会话内重复问法零成本命中；查询文本短、量级小，缓存上限内内存可忽略。
"""
from collections import OrderedDict

import torch
from sentence_transformers import SentenceTransformer

from src.data.config import EMBEDDING_MODEL_PATH

# 查询向量缓存上限（金融问答查询文本短且重复率有限，512 条 768 维 float 列表约 <2MB）
_QUERY_CACHE_MAX = 512


class EmbeddingService:
    _instance = None
    _model = None
    _query_cache = None
    # BGE 中文检索固定指令前缀（仅查询用，文档不需要加）
    QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._query_cache = OrderedDict()
            # 本地部署目录优先（models/bge-base-zh-v1.5），无本地目录时回退 HF 模型名
            # GPU 自适应：驱动/CUDA wheel 就绪时自动上卡，否则回退 CPU——同一代码
            # 单机两用，需要强制 CPU 时设 CUDA_VISIBLE_DEVICES="" 即可。
            # 注意：CPU 与 GPU 浮点结果有极微差异，切卡后应全量重嵌一次统一基线，
            # 避免混合两种设备产出的向量（sync_from_sqlite 全删重插正好承担此动作）
            use_gpu = torch.cuda.is_available()
            cls._model = SentenceTransformer(
                EMBEDDING_MODEL_PATH, device="cuda" if use_gpu else None
            )
            cls._batch_size = 256 if use_gpu else 32
        return cls._instance

    def encode_query(self, text: str) -> list[float]:
        """生成用户查询的向量，必须加检索指令前缀；同文本命中缓存则跳过前向编码"""
        cached = self._query_cache.get(text)
        if cached is not None:
            self._query_cache.move_to_end(text)  # 命中即提到最近使用端
            return cached
        full_text = self.QUERY_PREFIX + text
        vector = self._model.encode(full_text, normalize_embeddings=True)
        result = vector.tolist()
        self._query_cache[text] = result
        if len(self._query_cache) > _QUERY_CACHE_MAX:
            self._query_cache.popitem(last=False)  # 逐出最久未用
        return result

    def encode_doc(self, text: str) -> list[float]:
        """生成单条知识库文档的向量，不加前缀"""
        vector = self._model.encode(text, normalize_embeddings=True)
        return vector.tolist()

    def encode_doc_batch(self, texts: list[str]) -> list[list[float]]:
        """批量生成文档向量，不加前缀（GPU 时批次提到 256 喂饱显存）"""
        vectors = self._model.encode(texts, normalize_embeddings=True,
                                     batch_size=self._batch_size)
        return vectors.tolist()


# 全局单例
embedding_service = EmbeddingService()
