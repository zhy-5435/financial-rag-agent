"""L1 数据层：知识库构建、存储与向量化

数据流：投放目录原始文档 → 自动摄取（解析/切块/质控/版本化）→ SQLite 主库（doc_version + knowledge_chunk）
→ Milvus Lite 向量索引；检索时：query → Milvus COSINE 匹配 → knowledge_id 回 SQLite 补全切片详情
"""
