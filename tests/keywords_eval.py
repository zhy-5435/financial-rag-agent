"""keywords 自动抽取常驻评测：规则路确定性断言 + LLM 路实测与降级模拟 + 落库/FTS 链路接线

复跑：python -X utf8 tests/keywords_eval.py（约 10 秒；含 1 次真实 LLM 批量调用，
端点不可用时该组断言自动验证「降级到规则路」兜底而非 FAIL——入库永不依赖网络是设计不变量）。
"""
import io
import json
import re
import sys
import tempfile
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.ingestion import keywords as kw
from src.data.ingestion.__main__ import main as cli_main
from src.data.ingestion.config import KEYWORDS_TOP_N
from src.data.ingestion.loaders import to_db_rows
from src.data.ingestion.schemas import ChunkDraft
from src.data.sqlite_client import sqlite_client

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" | {detail}" if detail else ""))


def _draft(content: str, heading: str = "测试文档 > 交易限制") -> ChunkDraft:
    return ChunkDraft(source_path=Path("t.txt"), heading_path=heading, clause_position=None,
                      chunk_type="clause", content=content, original_text=content)


# ---------- A. 规则路（确定性、离线） ----------
def eval_rule() -> None:
    print("\nA 规则路抽取（确定性断言）")
    text = ("科创板单一账户单日累计买入股票不得超过20万股，申报价格上限为前收盘价的20%，"
            "深圳证券交易所科创板实行T+1交收制度，交易参与人应当遵守相关风控指标。")
    words = kw.extract_rule(text, "深圳证券交易所交易规则 > 第二章 交易时间 > 2.4 交易限制")
    joined = " ".join(words)
    check("A1 数值/字母实体保留（20%、T+1）", "20%" in words and "T+1" in words, str(words))
    check("A2 章节主题词剥离编号前缀", "交易时间" in words and "交易限制" in words
          and not any(w.startswith("第") for w in words))
    check("A3 停用字开头的噪声词被过滤", not any(re.match(r"^[的了和与是在为对按据]", w) or
          "应当" in w or "根据" in w for w in words), str(words))
    check("A4 数量与长度上限", len(words) <= KEYWORDS_TOP_N and all(2 <= len(w) <= 16 for w in words))
    check("A5 确定性（两次调用结果一致）", kw.extract_rule(text, "同一 > 路径") ==
          kw.extract_rule(text, "同一 > 路径"))

    # 滑窗交叠去噪：选了「深圳证券交易所」类长词后不应再收「圳证券交易所」类变体
    real_doc = Path("data/raw/正式版文档/监管规则类/深圳证券交易所交易规则（2026年修订）.txt")
    if real_doc.exists():
        body = real_doc.read_text(encoding="utf-8", errors="replace")[:1200]
        grams = kw._ngram_terms(body)
        bad = [(a, b) for i, a in enumerate(grams) for b in grams[i + 1:]
               if len(set(b) & set(a)) / max(len(set(b)), 1) >= 0.6]
        check("A6 真实规则段落无滑窗交叠变体共存", not bad, str(bad or grams))
    else:
        check("A6 真实规则段落无滑窗交叠变体共存", True, "样例缺失，跳过（视为通过）")

    cleaned = kw._clean_words(["短", "超长的十七个字符字符字符字符字符字", 3, None, "合法词", "合法词", ""])
    check("A7 _clean_words 剔除非字符串/超短超长/重复", cleaned == ["合法词"], str(cleaned))


# ---------- B. LLM 路：真实调用 + 降级兜底 ----------
def eval_llm() -> None:
    print("\nB LLM 路（实测 + 降级设计不变量）")
    drafts = [
        _draft("上海证券交易所科创板股票实行竞价交易，日内有效申报的价格涨跌幅限制比例为20%。"),
        _draft("融资融券的标的证券由交易所确定并公布，投资者信用交易需缴纳保证金，维持担保比例不得低于130%。"),
        _draft("基金管理人办理基金份额的申购与赎回，开放日为上海证券交易所正常交易日，赎回费率按持有期限分档。"),
        _draft("投资者适当性管理要求经营机构了解客户的产品风险等级，将适当的产品销售给适合的投资者。"),
        _draft("债券现券交易可以采用现券买卖或者质押式回购方式，报价方式包括点击成交和询价成交。"),
    ]
    note = kw.assign_keywords(drafts, "llm")
    all_words = [w for d in drafts for w in d.keywords]
    endpoint_ok = "退规则" not in note and "不可用" not in note
    if endpoint_ok:
        check("B1 LLM 实测：全部切片拿到关键词", all(d.keywords for d in drafts), note)
        check("B2 LLM 实测：数量在上限内且无整句", all(1 <= len(d.keywords) <= KEYWORDS_TOP_N
              and not any("。" in w or len(w) > 16 for w in d.keywords) for d in drafts),
              json.dumps([d.keywords for d in drafts], ensure_ascii=False))
        extra = [w for d in drafts for w in d.keywords if w not in d.content]
        check("B3 LLM 增量价值：存在正文未出现的补充词（同义/简称）", len(extra) > 0,
              f"补充词示例：{extra[:5]}")
    else:
        check("B1 端点不可用时整批降级规则路（入库不依赖网络）", all(d.keywords for d in drafts),
              f"{note}（本次运行端点不可用，降级链路已验证）")
    # 模拟单批失败：注入必抛异常的假客户端，llm_extract 必须兜底非空且报告回退批数
    real_get_llm = kw._get_llm
    try:
        kw._get_llm = lambda: (_ for _ in ()).throw(RuntimeError("模拟断网"))
        res, degraded = kw.llm_extract(["科创板涨跌幅限制为20%，实行注册制。" * 3])
        check("B4 模拟断网：整批回退规则路且结果非空", degraded == 1 and bool(res[0]), str(res))
        note2 = kw.assign_keywords([_draft("科创板实行注册制，涨跌幅限制为20%。")], "llm")
        check("B5 建连失败 assign 返回降级说明", note2.startswith("rule(llm不可用"), note2)
    finally:
        kw._get_llm = real_get_llm
    # 容错解析：围栏 JSON / 坏输出 / 编号缺失
    fenced = '```json\n{"0": ["科创板", "20%"], "1": []}\n```'
    parsed = kw._parse_reply(fenced, 2)
    check("B6 围栏 JSON 解析与空数组编号剔除", parsed == {0: ["科创板", "20%"]}, str(parsed))
    check("B7 坏输出解析为空 dict（由上层退规则）", kw._parse_reply("抱歉我不知道", 2) == {})


# ---------- C. 落库映射与 FTS 链路 ----------
def eval_pipeline() -> None:
    print("\nC 写库映射与混合召回链路")
    d = _draft("科创板涨跌幅限制为20%。")
    d.keywords = ["科创板", "涨跌幅", "20%"]
    rows = to_db_rows([d, replace(d, keywords=[])], "AUTO-测试:V1")
    check("C1 keywords 序列化为 ensure_ascii=False JSON",
          rows[0]["keywords"] == json.dumps(["科创板", "涨跌幅", "20%"], ensure_ascii=False)
          and "\\u" not in rows[0]["keywords"], rows[0]["keywords"])
    check("C2 空关键词落 NULL（保持旧行为）", rows[1]["keywords"] is None)

    body = sqlite_client._fts_body("申报价格上限为前收盘价的20%", rows[0]["keywords"])
    # FTS body 为单字空格切分串，按 token 序验 keywords 已拼入
    check("C3 keywords 拼入 FTS 索引 body", "科 创 板" in body and "涨 跌 幅" in body, body[:80])


# ---------- D. CLI --check 接线（空跑零联网） ----------
def eval_cli() -> None:
    print("\nD --check 接线")
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "测试细则.txt"
        f.write_text("第一章 总则\n1.1 为规范科创板市场交易行为，涨跌幅限制为20%，投资者应当遵守。\n"
                     "1.2 上海证券交易所对本所证券交易实行公开、公平、公正的原则监管。\n", encoding="utf-8")
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cli_main(["--check", str(f), "--preview", "1"])
        out = buf.getvalue()
        check("D1 --check 空跑成功且打印抽取说明", rc == 0 and "keywords 自动抽取：rule" in out,
              [ln for ln in out.splitlines() if "keywords" in ln or "kw=" in ln][:2])
        check("D2 预览行含 kw= 字段", "kw=" in out)


if __name__ == "__main__":
    print("=" * 60)
    print("keywords 自动抽取评测")
    print("=" * 60)
    eval_rule()
    eval_llm()
    eval_pipeline()
    eval_cli()
    fails = [name for name, ok, _ in RESULTS if not ok]
    print("\n" + "=" * 60)
    print(f"共 {len(RESULTS)} 项断言，FAIL {len(fails)} 项" + (f"：{fails}" if fails else " ✓"))
    sys.exit(1 if fails else 0)
