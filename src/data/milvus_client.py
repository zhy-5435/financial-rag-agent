"""Milvus Lite 向量索引：全量同步（幂等回退）/ 增量 delta 维护 / 带生效期过滤的语义检索"""
import hashlib
import json
import time

import pandas as pd
from pymilvus import MilvusClient

from src.config import MILVUS_DB_PATH
from src.data.config import MILVUS_COLLECTION, TOP_K, VECTOR_DIM
from src.data.embedding import embedding_service
from src.data.sqlite_client import sqlite_client


# 长期有效（expire_date 为 NULL）在时间轴上的落点：与 SQL 路「IS NULL 即不限」口径对齐，
# 取足够远的日期而不取 int64 上界，避免后续任何日期加法溢出
OPEN_END_DATE = "2099-12-31"
_OPEN_END_TS = int(pd.to_datetime(OPEN_END_DATE).timestamp())


def _is_blank(value) -> bool:
    """生效期空值判定：None / NaN / NaT / 空串与纯空白均视为未填写
    （列全为空时 pandas 会把 TEXT 列推成 float64，读到的就是 NaN 而非 None）"""
    if value is None or (not isinstance(value, str) and bool(pd.isna(value))):
        return True
    return not str(value).strip()


def _date_str_to_ts(date_str, *, empty_as: str = "start") -> int:
    """日期字符串转 Unix 时间戳，用于检索过滤。

    两列在表结构中允许 NULL（README：expire_date 为 NULL 表示长期有效），而向量路的
    effective_ts / expire_ts 必须是可比较整数，故空值在写入侧归到时间轴端点：
    empty_as='start' → 0（立即生效），'end' → OPEN_END_DATE（永不过期）。
    缺此兜底时 NULL 会在向量同步阶段对 None 调 .timestamp() 直接抛 AttributeError。
    """
    if _is_blank(date_str):
        return _OPEN_END_TS if empty_as == "end" else 0
    return int(pd.to_datetime(date_str).timestamp())


def kid_to_num_id(knowledge_id: str) -> int:
    """knowledge_id → 向量库数值主键：取 sha1 前 15 位十六进制（60 bit，正整数安全）。

    确定性映射使增量插入幂等：同一 knowledge_id 重插即原位覆盖（auto_id=False 的
    insert 为 upsert 语义），中断后补跑不会产生重复向量；与旧全量同步的序号 id
    空间兼容（全量重建会先清空集合，两套 id 不共存）。"""
    return int(hashlib.sha1(knowledge_id.encode("utf-8")).hexdigest()[:15], 16)


class MilvusStore:
    def __init__(self):
        self.client = MilvusClient(str(MILVUS_DB_PATH))
        self.collection_name = MILVUS_COLLECTION
        self._collection_loaded = False
        self._init_collection()

    def _ensure_loaded(self) -> None:
        """读取路径的 load_collection 幂等但每次都有 gRPC 往返成本（P2 时延优化）：
        进程内首次检索加载后记账，后续直接跳过；写入/同步路径仍显式 load，
        兼容外部释放后重载。"""
        if self._collection_loaded:
            return
        self.client.load_collection(self.collection_name)
        self._collection_loaded = True

    def _init_collection(self):
        """初始化集合，幂等操作。
        静态 schema 仅含 id / vector；knowledge_id 等业务字段为动态字段（随行存储），
        旧版本遗留数据行由 sync_from_sqlite 的全删重插机制统一清理。
        """
        if not self.client.has_collection(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                dimension=VECTOR_DIM,
                metric_type="COSINE",  # 余弦相似度，返回值越大越相似
                auto_id=False,
                description="L1 核心权威知识库向量索引",
            )
            print(f"Milvus 集合 {self.collection_name} 创建完成")
        else:
            print(f"Milvus 集合 {self.collection_name} 已存在")

    def sync_from_sqlite(self):
        """全量同步 SQLite 有效切片到 Milvus：先清空集合再重新插入，保证双库强一致。

        注：无有效切片时也必须清空集合——主库被重建为空（如全量重建的第一步）后，
        残留向量仍会参与召回并回查不到详情，属脏索引。"""
        df = sqlite_client.get_valid_knowledge_with_meta()
        self.client.load_collection(self.collection_name)
        self._collection_loaded = True
        # 清空现有向量，防止重复构建时累积旧数据（空库场景下即为唯一的清理动作）
        self.client.delete(collection_name=self.collection_name, filter="id >= 0")
        if df.empty:
            print("SQLite 中无有效知识切片，已清空向量集合")
            return

        contents = df["content"].tolist()
        vectors = embedding_service.encode_doc_batch(contents)

        data = []
        for idx, (_, row) in enumerate(df.iterrows()):
            data.append({
                "id": idx + 1,
                "vector": vectors[idx],
                "knowledge_id": row["knowledge_id"],
                "doc_version_id": row["doc_version_id"],
                "effective_ts": _date_str_to_ts(row["effective_date"]),
                "expire_ts": _date_str_to_ts(row["expire_date"], empty_as="end"),
                "status": row["status"],
            })

        self.client.insert(collection_name=self.collection_name, data=data)
        print(f"Milvus 同步完成，共插入 {len(data)} 条有效向量")

    def apply_delta(self, delete_ids: list[str], insert_df: pd.DataFrame) -> tuple[int, int]:
        """增量同步（Phase 4 P0）：只删退出版本的向量、只嵌新入库切片，不动其余存量。

        替代每轮全量重嵌（O(全库)→O(本批)），是大批量文档持续入库的性能基础。
        一致性保障：插入 id 由 knowledge_id 确定性哈希（重跑幂等）；字段与全量路同构。
        存量与主库长期漂移（如同步中断）时的修复通道：python -m src.data.pipeline 全量重建，
        或摄取命令加 --full-sync。返回 (删除数, 插入数)。
        """
        n_del = n_ins = 0
        self.client.load_collection(self.collection_name)
        self._collection_loaded = True
        insert_kids = insert_df["knowledge_id"].tolist() if not insert_df.empty else []
        # delete-then-insert 实现 upsert 语义：Milvus Lite 的 insert 不保证同主键覆盖，
        # 新批 id 也先删一遍，上批中断于插入阶段的残留不会变重复向量（重跑幂等）
        to_delete = list(dict.fromkeys(delete_ids + insert_kids))
        if to_delete:
            # knowledge_id 为纯 ASCII（见 loaders 命名规则），表达式无转义风险
            self.client.delete(collection_name=self.collection_name,
                               filter=f'knowledge_id in {json.dumps(to_delete, ensure_ascii=False)}')
            n_del = len(delete_ids)
        if not insert_df.empty:
            contents = insert_df["content"].tolist()
            vectors = embedding_service.encode_doc_batch(contents)
            data = []
            for idx, (_, row) in enumerate(insert_df.iterrows()):
                data.append({
                    "id": kid_to_num_id(row["knowledge_id"]),
                    "vector": vectors[idx],
                    "knowledge_id": row["knowledge_id"],
                    "doc_version_id": row["doc_version_id"],
                    "effective_ts": _date_str_to_ts(row["effective_date"]),
                    "expire_ts": _date_str_to_ts(row["expire_date"], empty_as="end"),
                    "status": row["status"],
                })
            self.client.insert(collection_name=self.collection_name, data=data)
            n_ins = len(data)
        if n_del or n_ins:
            print(f"Milvus 增量同步：删除 {n_del} 条，新增 {n_ins} 条（其余存量未触碰）")
        return n_del, n_ins

    def count_rows(self) -> int:
        """集合内当前向量总数，供增量链路主库对账（Lite 无独立 count API，拉全量主键计数；
        十万级存量下每批一次仍有成本，仅用于校验/演练，不参与常规同步路径）"""
        self.client.load_collection(self.collection_name)
        rows = self.client.query(collection_name=self.collection_name, filter="id >= 0",
                                 output_fields=["id"])
        return len(rows)

    def search(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """带过滤的向量检索：仅返回 status=valid 且所属文档在生效时间范围内的切片。"""
        self._ensure_loaded()

        current_ts = int(time.time())
        query_vector = embedding_service.encode_query(query)
        filter_expr = (
            f'status == "valid" '
            f"and effective_ts <= {current_ts} "
            f"and expire_ts >= {current_ts}"
        )

        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            filter=filter_expr,
            limit=top_k,
            output_fields=["knowledge_id"],
        )

        hits = results[0]
        if not hits:
            return []

        kid_list = [hit["entity"]["knowledge_id"] for hit in hits]
        df_detail = sqlite_client.get_knowledge_detail_by_ids(kid_list)

        result = []
        for hit in hits:
            kid = hit["entity"]["knowledge_id"]
            matched = df_detail[df_detail["knowledge_id"] == kid]
            if matched.empty:
                continue
            detail = matched.iloc[0]
            # COSINE 度量下 Milvus 返回的 distance 即余弦相似度，范围 [-1,1]，越大越相似
            result.append({
                "knowledge_id": kid,
                "doc_version_id": detail["doc_version_id"],
                "chunk_type": detail["chunk_type"],
                "content": detail["content"],
                "original_text": detail["original_text"],
                "clause_position": None if pd.isna(detail["clause_position"]) else detail["clause_position"],
                "heading_path": detail["heading_path"],
                "keywords": self._parse_keywords(detail["keywords"]),
                "doc_name": detail["doc_name"],
                "doc_type": detail["doc_type"],
                "version": detail["version"],
                "publish_dept": detail["publish_dept"],
                "effective_date": detail["effective_date"],
                "similarity": round(float(hit["distance"]), 4),
            })

        result.sort(key=lambda x: x["similarity"], reverse=True)
        return result

    def get_similarity_by_ids(self, query: str, knowledge_ids: list[str]) -> dict[str, float]:
        """对指定切片补算与 query 的余弦相似度，供混合检索补齐仅关键词路命中的候选。

        返回 {knowledge_id: 相似度}；向量入库前已归一化，本地点积即余弦相似度。
        """
        if not knowledge_ids:
            return {}
        self._ensure_loaded()
        query_vector = embedding_service.encode_query(query)
        # ensure_ascii=False：默认 json.dumps 会把中文 id 转义成 \uXXXX，Milvus 表达式解析器
        # 不支持 \u 转义（报 unknown escape sequence）；向量库中的 join 键应保持 ASCII（见 loaders），
        # 此处同时做防御，避免存量/外部非 ASCII id 直接炸掉混合检索链路
        rows = self.client.query(
            collection_name=self.collection_name,
            filter=f'knowledge_id in {json.dumps(knowledge_ids, ensure_ascii=False)}',
            output_fields=["knowledge_id", "vector"],
        )
        similarities = {}
        for row in rows:
            vector = row["vector"]
            similarities[row["knowledge_id"]] = round(
                sum(a * b for a, b in zip(query_vector, vector)), 4
            )
        return similarities

    @staticmethod
    def _parse_keywords(raw) -> list:
        """keywords 在 SQLite 中为 JSON 字符串，解析为列表便于上层直接使用"""
        if not isinstance(raw, str) or not raw.strip():
            return []
        try:
            value = json.loads(raw)
            return value if isinstance(value, list) else []
        except json.JSONDecodeError:
            return []


# 全局实例
milvus_store = MilvusStore()
