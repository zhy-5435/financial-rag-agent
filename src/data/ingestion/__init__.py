"""L1 自动摄取子包：LlamaIndex 限域用作「文档解析 + 结构化感知 + 切块」引擎

数据流定位（见 docs/LlamaIndex引入规划.md）：
    data/incoming/ 新文档 → readers(格式路由) → structuring(章节/条款) →
    chunking(切块) → quality(三检门控) → loaders(映射 knowledge_chunk 行) →
    SQLite 权威主库（Phase 4 接入）→ Milvus 向量同步 → 双路混合检索（零改动）

Phase 0 交付：`python -m src.data.ingestion --check` 空跑，仅打印解析与切块结果。
"""
