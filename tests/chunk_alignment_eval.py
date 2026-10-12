"""Phase 1 切块对齐评测：自动「节级」切块 vs 人工金标切片（只读，不写任何库）

粒度决策：自动通道以「节」为基本 Chunk（人工金标以「条」为粒度，二者有意不同），
故对齐指标重定义为——
    覆盖率       每个人工金标切片的 original_text 是否被完整包含于某一个自动 chunk
                 （金标不被拦腰切断即视为边界合格，门槛 ≥80%，实测预期接近 100%）
    章节路径命中率 覆盖该金标切片的自动 chunk，其 heading_path 是否包含金标
                 heading_path 的 章/节 层级要素（门槛 ≥90%）
    结构统计     原文行首条款行数 vs 自动识别条款数、chunk 数、条级回退数

启动方式（项目根目录）：
    python -X utf8 tests/chunk_alignment_eval.py
"""
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RAW_DATA_DIR  # noqa: E402
from src.data.ingestion import quality  # noqa: E402
from src.data.ingestion.chunking import chunk_document  # noqa: E402
from src.data.ingestion.readers import read_document  # noqa: E402

# 金标文档范围：4 份监管规则
GOLD_DOC_IDS = ("REG-001:V2026", "REG-002:V2026", "REG-003:V2020", "REG-004:V2020")

_COVERAGE_GATE = 0.80
_PATH_GATE = 0.90

# 原文行首条款（与 structuring 判定一致）：多级点号编号 或 第X条
_RE_SOURCE_CLAUSE = re.compile(
    r"^(?:\d+(?:\.\d+)+[\u2002\u3000\s、.]|第[一二三四五六七八九十百千零〇\d]+条[\u2002\u3000\s、.])"
)
# 归一化：去除全部空白字符（含全角/窄空格），消除排版差异对包含判断的干扰
_RE_WS = re.compile(r"[\s\u3000\u2002]+")


def _norm(text: str) -> str:
    return _RE_WS.sub("", text or "")


# 层级标题中的编号前缀（章/节/条中文编号与点号数字），提取裸标题用于可比性归一
_RE_LEVEL_NUM = re.compile(r"^(?:第[一二三四五六七八九十百千零〇\d]+[章节编篇部条]|[\d.]+|[（(]\d+[)）])[\s\u3000\u2002、.]*")


def _path_titles(heading_path: str, strip_leaf: bool) -> set[str]:
    """heading_path 拆级→剥编号取裸题→（末级为条款号时）剔末级→归一化集合。

    金标与自动对同一层级的写法不同（金标「第三章 > 3.8 融资融券交易」，自动「第三章 证券
    买卖 > 第八节 融资融券交易」；金标裸题「总则」vs 自动「第一章 总则」），剥掉编号后按
    标题要素集合比较才可对；无裸题的层级（如纯「第一条」）归一为空集，作通配处理。
    strip_leaf 仅在该 chunk 末级是条款号（clause_position 非空）时剔末级；节级块末级是节名，保留。
    """
    parts = [p.strip() for p in (heading_path or "").split(">") if p.strip()]
    if strip_leaf and len(parts) > 1:
        parts = parts[:-1]
    titles = set()
    for p in parts:
        t = _norm(_RE_LEVEL_NUM.sub("", p))
        if t:
            titles.add(t)
    return titles


def load_gold(conn, doc_version_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT knowledge_id, clause_position, heading_path, original_text, file_path"
        " FROM knowledge_chunk k JOIN doc_version d USING (doc_version_id)"
        " WHERE k.doc_version_id = ? AND k.status = 'valid' ORDER BY k.knowledge_id",
        (doc_version_id,),
    ).fetchall()
    keys = ("knowledge_id", "clause_position", "heading_path", "original_text", "file_path")
    return [dict(zip(keys, r)) for r in rows]


def eval_doc(conn, doc_version_id: str) -> dict:
    gold = load_gold(conn, doc_version_id)
    if not gold:
        return {"doc": doc_version_id, "skip": "无金标"}
    path = RAW_DATA_DIR / gold[0]["file_path"].lstrip("/")
    doc = read_document(path)
    drafts = quality.annotate(chunk_document(doc))

    auto_norms = [(_norm(d.original_text), d) for d in drafts]
    n_cover = n_path = 0
    misses, path_misses = [], []
    for g in gold:
        gn = _norm(g["original_text"])
        hit = next((d for an, d in auto_norms if gn and gn in an), None)
        if hit is None:
            misses.append(g["knowledge_id"])
            continue
        n_cover += 1
        g_elems = _path_titles(g["heading_path"], strip_leaf=bool(g["clause_position"]))
        a_elems = _path_titles(hit.heading_path, strip_leaf=bool(hit.clause_position))
        # 标题要素双向子集即命中（空集作通配：自动更粗/更细都不罚，金标空集时自动任意标题∈）
        if g_elems <= a_elems or a_elems <= g_elems:
            n_path += 1
        else:
            path_misses.append(g["knowledge_id"])

    # 结构统计：原文行首条款 vs 自动识别（回退条级 chunk 的 clause_position + 被合并条款号）
    src_clauses = sum(1 for ln in doc.text.splitlines() if _RE_SOURCE_CLAUSE.match(ln.strip()))
    auto_clauses = {d.clause_position for d in drafts if d.clause_position}
    merged_clause_ids = sum(
        len(re.findall(r"(?:^|\n)(?:\d+(?:\.\d+)+|第[一二三四五六七八九十百千零〇\d]+条)", d.original_text))
        for d in drafts
    )
    n_review = sum(1 for d in drafts if d.issues)
    return {
        "doc": doc_version_id,
        "title": doc.title,
        "gold_n": len(gold),
        "coverage": n_cover / len(gold),
        "path_hit": n_path / n_cover if n_cover else 0.0,
        "src_clauses": src_clauses,
        "auto_explicit_clauses": len(auto_clauses),
        "auto_text_clause_ids": merged_clause_ids,
        "chunks": len(drafts),
        "need_review": n_review,
        "misses": misses[:8],
        "path_misses": path_misses[:8],
    }


def main():
    conn = sqlite3.connect("file:db/sqlite/l1_core.db?mode=ro", uri=True)
    print("=" * 66)
    print("Phase 1 切块对齐评测（自动节级 chunk vs 人工条级金标）")
    print("=" * 66)
    results = [eval_doc(conn, vid) for vid in GOLD_DOC_IDS]
    conn.close()

    total_gold = total_cover = total_path = 0
    for r in results:
        if r.get("skip"):
            print(f"{r['doc']}: {r['skip']}")
            continue
        total_gold += r["gold_n"]
        total_cover += round(r["coverage"] * r["gold_n"])
        total_path += round(r["path_hit"] * r["gold_n"] * r["coverage"])
        print(f"\n[{r['title']}]")
        print(f"  金标 {r['gold_n']} 条 | 自动 chunk {r['chunks']} 片（need_review {r['need_review']}）")
        print(f"  覆盖率 {r['coverage']:.1%}（门槛 {_COVERAGE_GATE:.0%}）"
              f" | 章节路径命中率 {r['path_hit']:.1%}（门槛 {_PATH_GATE:.0%}）")
        print(f"  结构统计：原文行首条款 {r['src_clauses']} 行，自动显式条款号 {r['auto_explicit_clauses']} 个，"
              f"chunk 内含条款编号 {r['auto_text_clause_ids']} 处")
        if r["misses"]:
            print(f"  未覆盖金标样例：{r['misses']}")
        if r["path_misses"]:
            print(f"  路径未命中样例：{r['path_misses']}")

    cov = total_cover / total_gold if total_gold else 0
    path = total_path / total_cover if total_cover else 0
    print("\n" + "=" * 66)
    verdict = "PASS" if cov >= _COVERAGE_GATE and path >= _PATH_GATE else "FAIL"
    print(f"总计：金标 {total_gold} 条 | 覆盖率 {cov:.1%} | 章节路径命中率 {path:.1%} → 验收 {verdict}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
