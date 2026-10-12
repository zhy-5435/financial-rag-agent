"""构建 Ragas 离线评估测试集：真实系统作答 + 人工核对的 ground_truth

样本事实依据逐条核对自 data/raw/正式版文档 原文（条款编号见 SAMPLES 的 source_ref 字段）。
作答与检索证据由 L3 POST /chat 现场采集（需 L2 :8000 与 L3 :8001 已启动），
落盘为冻结测试集 data/eval/ragas_testset.json，供评估命令直接读取，保证结果可复现。

用法（项目根目录执行）：
    .venv1\\Scripts\\python.exe tests\\build_ragas_testset.py

产物：data/eval/ragas_testset.json
"""
import datetime as dt
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = PROJECT_ROOT / "data" / "eval" / "ragas_testset.json"
L3_BASE = "http://127.0.0.1:8001"

# 单条证据注入长度上限（字符）：与 ragas 提示词预算平衡，过长会拖慢 judge 且稀释要点
CONTEXT_MAX_CHARS = 1200

# 评测样本：问题 + 标准答案（人工核对原文）+ 出处（仅供人工复核，不参与打分）
SAMPLES: list[dict] = [
    {
        "id": "rs-01",
        "category": "内部制度类/差旅报销管理办法",
        "question": "出差住宿费标准是多少",
        "source_ref": "第八条",
        "ground_truth": (
            "住宿费按单间/天计算，同性两人出差原则上合住并按一间标准报销："
            "一类城市员工350元、部门负责人450元、副总及以上600元；"
            "二类城市员工300元、部门负责人400元、副总及以上550元；"
            "三类城市员工250元、部门负责人350元、副总及以上500元。"
        ),
    },
    {
        "id": "rs-02",
        "category": "内部制度类/差旅报销管理办法",
        "question": "出差的市内交通费和伙食补助标准是什么",
        "source_ref": "第九条",
        "ground_truth": (
            "市内交通与伙食补助实行定额包干、无需提供发票："
            "市内交通一类80元/天、二类60元/天、三类40元/天；"
            "伙食补助一类100元/天、二类80元/天、三类60元/天。"
        ),
    },
    {
        "id": "rs-03",
        "category": "内部制度类/差旅报销管理办法",
        "question": "员工出差需要经过什么审批流程",
        "source_ref": "第五条、第六条",
        "ground_truth": (
            "员工出差须提前3个工作日在OA提交《差旅申请单》，由部门负责人审批；"
            "部门负责人出差由分管副总审批；副总及以上出差由总经理审批；"
            "紧急公务无法提前审批的，须在出差后3个工作日内补全审批手续，逾期不予报销。"
        ),
    },
    {
        "id": "rs-04",
        "category": "内部制度类/考勤与休假管理制度",
        "question": "员工迟到早退会怎么处理",
        "source_ref": "第六条、第七条、第八条",
        "ground_truth": (
            "上班晚到30分钟内为迟到，下班提前30分钟内为早退；"
            "月累计3次以内口头警告，3次以上按事假半天处理；"
            "迟到早退超过30分钟属旷工界定情形，旷工1天扣发3天工资，"
            "连续旷工3天或年度累计旷工5天按严重违纪解除劳动合同；代打卡双方均按旷工1天处理。"
        ),
    },
    {
        "id": "rs-05",
        "category": "内部制度类/考勤与休假管理制度",
        "question": "带薪年休假有多少天",
        "source_ref": "第十条",
        "ground_truth": (
            "累计工作满1年不满10年休5天，满10年不满20年休10天，满20年休15天；"
            "当年未休完的可结转至次年3月，逾期作废。"
        ),
    },
    {
        "id": "rs-06",
        "category": "内部制度类/信息安全与数据保密管理制度",
        "question": "涉密数据可以存放在个人设备或公共云盘吗",
        "source_ref": "第十条",
        "ground_truth": (
            "不可以。涉密数据须存储在公司指定服务器，严禁存储在个人设备、公共云盘；"
            "传输涉密数据必须使用加密通道，严禁通过微信、QQ等公共渠道传输涉密文件。"
        ),
    },
    {
        "id": "rs-07",
        "category": "内部制度类/产品审批管理制度",
        "question": "新产品立项需要经过哪些审批环节",
        "source_ref": "第七条、第十条",
        "ground_truth": (
            "立项须提交《产品立项申请书》（含市场分析、盈利预测、资源需求、风险评估），"
            "经部门初审后报公司级审批，通过后方可启动研发；"
            "上线发布须提交《上线发布申请》并附测试报告、合规审核意见、运营方案与应急预案，"
            "普通产品由分管副总审批，涉及用户资金、核心数据的产品由总经理审批；"
            "一般事项3个工作日内办结，重大事项7个工作日内办结。"
        ),
    },
    {
        "id": "rs-08",
        "category": "监管规则类/上海证券交易所交易规则",
        "question": "A股主板股票的涨跌幅限制比例是多少",
        "source_ref": "3.3.13",
        "ground_truth": (
            "上交所对股票、基金交易实行价格涨跌幅限制，涨跌幅限制比例为10%，"
            "涨跌幅限制价格=前收盘价×（1±涨跌幅限制比例）；"
            "首次公开发行上市后前5个交易日、退市整理期首个交易日、退市后重新上市首个交易日等情形不实行涨跌幅限制。"
        ),
    },
    {
        "id": "rs-09",
        "category": "监管规则类/上海证券交易所交易规则",
        "question": "盘后固定价格交易的时间是怎么规定的",
        "source_ref": "3.7.2、3.7.3",
        "ground_truth": (
            "每个交易日的15:05至15:30为盘后固定价格交易时间，"
            "交易系统按时间优先顺序对收盘定价申报撮合并以当日收盘价成交；"
            "收盘定价申报的接受时间为每个交易日9:30至11:30、13:00至15:30；"
            "当日15:00仍处于停牌状态的证券不进行盘后固定价格交易。"
        ),
    },
    {
        "id": "rs-10",
        "category": "监管规则类/上海证券交易所交易规则",
        "question": "大宗交易的申报数量门槛是多少",
        "source_ref": "3.6.1",
        "ground_truth": (
            "A股单笔买卖申报数量不低于30万股，或交易金额不低于200万元人民币；"
            "B股单笔申报数量不低于30万股，或交易金额不低于20万美元；"
            "基金单笔申报数量不低于200万份，或交易金额不低于200万元；"
            "上交所可根据市场情况调整大宗交易的最低限额。"
        ),
    },
    {
        "id": "rs-11",
        "category": "监管规则类/证券期货投资者适当性管理办法",
        "question": "投资者分为哪几类，专业投资者需要符合什么条件",
        "source_ref": "第七条、第八条",
        "ground_truth": (
            "投资者分为普通投资者与专业投资者，普通投资者在信息告知、风险警示、适当性匹配等方面享有特别保护；"
            "专业投资者包括经批准的金融机构及其发行的理财产品、社保基金与企业年金等养老基金、QFII与RQFII，"
            "以及最近1年末净资产不低于2000万元、金融资产不低于1000万元且具有2年以上投资经历的法人或其他组织，"
            "金融资产不低于500万元或最近3年个人年均收入不低于50万元且具有2年以上投资经历（或2年以上金融相关工作经历）的自然人。"
        ),
    },
    {
        "id": "rs-12",
        "category": "监管规则类/证券投资基金销售机构监督管理办法",
        "question": "申请注册基金销售业务资格应当具备哪些条件",
        "source_ref": "第七条",
        "ground_truth": (
            "应当具备：财务状况良好、运作规范；有与基金销售业务相适应的营业场所、安全防范等设施，"
            "信息管理平台符合中国证监会规定；具备健全高效的业务管理和风险管理制度，"
            "反洗钱、反恐怖融资及非居民金融账户涉税信息尽职调查等制度合规，"
            "销售结算资金管理、投资者适当性管理、内部控制等制度符合证监会规定；"
            "取得基金从业资格的人员不少于20人；最近3年没有受到刑事处罚或重大行政处罚、"
            "最近1年没有因相近业务被采取重大行政监管措施等；以及中国证监会规定的其他条件。"
        ),
    },
    {
        "id": "rs-13",
        "category": "产品说明书类/易方达纯债债券型证券投资基金A",
        "question": "易方达纯债债券型基金A类的管理费和申购费是怎么收的",
        "source_ref": "费率结构",
        "ground_truth": (
            "管理费年化0.30%、托管费年化0.20%；"
            "A类份额在认购/申购时前端收取申购费，C类免申购费但计提销售服务费。"
        ),
    },
    {
        "id": "rs-14",
        "category": "产品说明书类/中邮理财邮银财富邮鸿宝30号",
        "question": "邮鸿宝30号理财产品的风险等级和费率是多少",
        "source_ref": "风险等级、费率结构",
        "ground_truth": (
            "风险等级为PR1（低风险）；申购/赎回费为0，固定管理费与托管费按产品说明书约定计提，无浮动管理费。"
        ),
    },
    {
        "id": "rs-15",
        "category": "产品说明书类/信银理财日盈象天天利199号（疑难）",
        "question": "信银理财日盈象天天利199号的管理费率是多少",
        "source_ref": "费率结构",
        "ground_truth": (
            "固定管理费年化0.30%，另有销售服务费0.30%、托管费0.03%、申购/赎回费0。"
        ),
    },
]


def call_chat(question: str, timeout: float = 120.0) -> dict:
    """POST L3 /chat 串行采集：429 按 Retry-After 等待重试，三次仍失败返回 {"__error__": ...} 不抛异常"""
    started = time.perf_counter()
    last_err = ""
    for _ in range(3):
        try:
            resp = httpx.post(f"{L3_BASE}/chat", json={"question": question}, timeout=timeout)
            if resp.status_code == 429:
                wait = min(float(resp.headers.get("Retry-After") or 5), 30)
                print(f"    [限流 429] 等待 {wait}s 后重试")
                time.sleep(wait + 1)
                last_err = "rate_limited"
                continue
            if resp.status_code == 200:
                data = resp.json()
                data["wall_ms"] = int((time.perf_counter() - started) * 1000)
                return data
            last_err = f"http_{resp.status_code}: {resp.text[:200]}"
        except Exception as e:  # noqa: BLE001 采集异常交给上层记为空作答
            last_err = f"{type(e).__name__}: {e}"
        time.sleep(2)
    return {"__error__": last_err, "wall_ms": int((time.perf_counter() - started) * 1000)}


def collect(sample: dict) -> dict:
    """调用 L3 /chat 采集真实作答与检索证据，组装为 Ragas 单样本"""
    question = sample["question"]
    print(f"  [{sample['id']}] {question}")
    data = call_chat(question)
    rec = {
        "id": sample["id"],
        "category": sample["category"],
        "question": question,
        "source_ref": sample["source_ref"],
        "ground_truth": sample["ground_truth"],
        "collected_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    if "__error__" in data:
        print(f"    ✗ 采集失败：{data['__error__']}")
        rec.update({"answer": "", "retrieved_contexts": [], "answer_type": None,
                    "collect_error": data["__error__"]})
        return rec

    contexts: list[str] = []
    for s in data.get("sources") or []:
        text = (s.get("original_text") or s.get("content") or "").strip()
        if not text:
            continue
        head = f"【{s.get('doc_name')}｜{s.get('clause_position') or '未标注位置'}】"
        contexts.append(f"{head}{text[:CONTEXT_MAX_CHARS]}")
    rec.update({
        "answer": (data.get("answer") or "").strip(),
        "retrieved_contexts": contexts,
        "answer_type": data.get("answer_type"),
        "intent": data.get("intent"),
        "n_sources": len(contexts),
    })
    print(f"    answer_type={data.get('answer_type')} 证据 {len(contexts)} 条 "
          f"回答 {len(rec['answer'])} 字")
    return rec


def main():
    only_missing = "--only-missing" in sys.argv  # 断点续采：已有样本不重复调用
    records: list[dict] = []
    existing: dict[str, dict] = {}
    if only_missing and OUT_PATH.exists():
        old = json.loads(OUT_PATH.read_text(encoding="utf-8"))
        existing = {r["id"]: r for r in old.get("samples", []) if not r.get("collect_error")}
        print(f"已有可用样本 {len(existing)} 条，仅补采缺失/失败样本")

    for sample in SAMPLES:
        if sample["id"] in existing:
            records.append(existing[sample["id"]])
            print(f"  [{sample['id']}] 沿用已采集记录")
            continue
        records.append(collect(sample))
        time.sleep(1)  # 拉开请求间隔，避开单用户 QPS 限流窗口

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "description": "Ragas 离线评估测试集：真实系统作答 + 人工核对标准答案（四指标评估输入）",
        "built_at": dt.datetime.now().isoformat(timespec="seconds"),
        "collect_channel": "L3 POST /chat（Agent Loop 真实链路）",
        "fields": {
            "question": "用户问题 → ragas user_input",
            "retrieved_contexts": "本轮检索到的知识库切片原文（含文档名与条款位置）→ ragas retrieved_contexts",
            "answer": "系统实际回答 → ragas response",
            "ground_truth": "人工核对原文所得标准答案 → ragas reference",
        },
        "samples": records,
    }
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    ok = [r for r in records if not r.get("collect_error")]
    empty_ctx = [r["id"] for r in ok if not r["retrieved_contexts"]]
    print("-" * 60)
    print(f"采集成功 {len(ok)}/{len(records)} 条，无证据样本：{empty_ctx or '无'}")
    print(f"测试集：{OUT_PATH}")


if __name__ == "__main__":
    main()
