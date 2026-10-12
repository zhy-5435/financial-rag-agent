"""增量检测与版本标识：投放目录快照 → 与 ingest_state 表比对 → added/changed/removed/unchanged

摄取是本库唯一的数据入口，ingest_state 因此是全库唯一的「已入库」凭据：只清业务表而留着
本表，重跑会全部判 unchanged 而零写库（全量重建走 src.data.pipeline，会连本表一起清）。
版本递增与 superseded 的库操作在 runner.py 编排，本模块只做纯计算：内容指纹、变更分类、doc_id 派生。
"""
import hashlib
from pathlib import Path

from src.data.ingestion.readers import is_supported
from src.data.ingestion.schemas import FileSnapshot


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def snapshot_dir(directory: Path) -> list[FileSnapshot]:
    """扫描目录内受支持文件，生成 (路径, mtime, sha256) 快照条目列表"""
    return [
        FileSnapshot(path=p, mtime=p.stat().st_mtime, sha256=file_sha256(p))
        for p in sorted(directory.rglob("*"))
        if p.is_file() and is_supported(p)
    ]


def diff_state(new: list[FileSnapshot], state: dict[str, str]) -> dict[str, list[FileSnapshot]]:
    """当前快照与已摄取状态（{路径: sha256}）比对，按分类返回快照条目。

    以 sha256 而非 mtime 判定变更：重新投放未修改的文件（仅时间戳变化）不误摄取。
    """
    result: dict[str, list[FileSnapshot]] = {"added": [], "changed": [], "removed": [], "unchanged": []}
    new_map = {str(s.path): s for s in new}
    for key, snap in new_map.items():
        if key not in state:
            result["added"].append(snap)
        elif state[key] != snap.sha256:
            result["changed"].append(snap)
        else:
            result["unchanged"].append(snap)
    for key in state:
        if key not in new_map:
            result["removed"].append(key)  # removed 只有路径可用（文件已不在）
    return result


def derive_doc_id(path: Path) -> str:
    """投放文件 → 文档实体标识：AUTO- 前缀 + 去空白规范化的文件名（不含扩展名）。

    同一 doc_id 多次摄取即版本递增（V1→V2→…）；重命名文件视为新文档（旧文档
    走 removed → expired 退出）。以路径而非内容做实体锚定：内容变化正是升版信号。
    """
    return "AUTO-" + "".join(str(path.stem).split())


def doc_version_id(doc_id: str, version_n: int) -> str:
    """文档实体 + 版本号 → doc_version 主键，如 AUTO-深圳证券交易所交易规则:V2"""
    return f"{doc_id}:V{version_n}"
