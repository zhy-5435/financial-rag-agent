"""落库映射：ChunkDraft → doc_version / knowledge_chunk 标准行 + 自动通道行构造

to_db_rows/build_doc_version_row 为纯函数（无副作用，--check 预览与 runner 写库共用）；
实际事务写入经 sqlite_client.write_new_version()，在 runner.py 编排。"""
import hashlib
import json
from datetime import date
from pathlib import Path

from src.data.ingestion.schemas import ChunkDraft, RawDocument

# knowledge_chunk.chunk_type 合法枚举镜像（与 sqlite_client.CHUNK_TYPE_VALUES 同集，
# 写库前自检，防启发式分类产出表 CHECK 之外的值）
_VALID_CHUNK_TYPES = {
    "clause", "table", "exception", "principle", "definition",
    "product_element", "fee", "risk", "process", "other",
}


def to_db_rows(drafts: list[ChunkDraft], doc_version_id: str, id_seq_start: int = 1) -> list[dict]:
    """候选切片 → knowledge_chunk 行字典（列序与 CHUNK_COLUMNS 对齐）。

    knowledge_id = AUTO- + doc_version_id 的 sha1 短哈希 + 局部序号：
    ① 与库内存量 L1- 前缀切片（历史人工通道遗留）不碰撞，AUTO- 前缀可整体圈定回退；
    ② id 保持纯 ASCII —— knowledge_id 同时是 Milvus 动态字段与 filter 表达式 join 键，
       中文 id 会撞 Milvus 不支持的 \\u 转义（端到端演练实测），文档名中文留在
       doc_version_id/文档表侧，不入侵 id 命名空间；
    ③ 同一版本内序号稳定有序，升版即换哈希段，新旧版切片可按前缀整体圈定。"""
    id_prefix = "AUTO-" + hashlib.sha1(doc_version_id.encode("utf-8")).hexdigest()[:12]
    rows = []
    for i, d in enumerate(drafts, start=id_seq_start):
        ctype = d.chunk_type if d.chunk_type in _VALID_CHUNK_TYPES else "other"
        rows.append({
            "knowledge_id": f"{id_prefix}-{i:04d}",
            "doc_version_id": doc_version_id,
            "clause_position": d.clause_position,
            "heading_path": d.heading_path,
            "chunk_type": ctype,
            "content": d.content,
            "original_text": d.original_text,
            # keywords 由 keywords.py 自动抽取（LLM 主路/规则兜底）填入 draft；
            # ensure_ascii=False 保持中文可读（该列不进 Milvus 表达式，仅 FTS body 拼接）；空列表落 NULL
            "keywords": json.dumps(d.keywords, ensure_ascii=False) if d.keywords else None,
            "status": d.suggested_status,
        })
    return rows


# 投放目录名提示 → doc_version.doc_type（与表 CHECK 口径对齐，未命中置未分类）
_DOC_TYPE_DIR_HINTS = (("监管", "监管规则"), ("内部制度", "内部制度"), ("产品说明书", "产品说明书"))

# 自动摄取无发布元数据，生效期取当日、失效期显式置远期占位（而非留 NULL）：
# 检索链路已支持两列为 NULL（见 milvus_client._date_str_to_ts 兜底），此处仍写占位值，
# 是为了让存量数据的日期语义在库内可见、便于人工补录真实失效日时直接覆盖
AUTO_EXPIRE_DATE = "2099-12-31"


def infer_doc_type(path: Path) -> str:
    """按投放目录路径中的提示词推断文档类型（目录分类即路由信号，同要素通道口径）"""
    parts = "".join(p.name for p in path.parents)
    for hint, doc_type in _DOC_TYPE_DIR_HINTS:
        if hint in parts:
            return doc_type
    return "未分类"


def build_doc_version_row(doc: RawDocument, doc_version_id: str, doc_id: str, version_n: int,
                          status: str, ingest_day: date | None = None) -> dict:
    """文档级行：发布元数据从缺（自动通道无人工登记），publish_date 留空供人工补录；
    status=valid 正常入检索，draft 为整篇隔离态（检索双重过滤天然不召回，人工复核后改 valid）"""
    today = (ingest_day or date.today()).isoformat()
    return {
        "doc_version_id": doc_version_id,
        "doc_id": doc_id,
        "doc_name": doc.title,
        "version": f"V{version_n}",
        "doc_type": infer_doc_type(doc.source_path),
        "publish_dept": None,
        "publish_date": None,
        "effective_date": today,
        "expire_date": AUTO_EXPIRE_DATE,
        "status": status,
        "file_path": str(doc.source_path),
        # created_at/updated_at 不列入行字典：写入侧按行键拼列，留给表 DEFAULT CURRENT_TIMESTAMP 填充
    }


def db_row_from_doc(doc: RawDocument) -> dict:  # 预留：未来若需从文档正文抽取发布元数据
    raise NotImplementedError("发布元数据自动抽取不在 Phase 4 范围")
