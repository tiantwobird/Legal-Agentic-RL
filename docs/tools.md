# LegalAgent-RL｜本地法律条文工具

基于用户提供的 `labor_subset.jsonl`，实现两个**无需 API Key、无需联网、无需 GPU**的 Python Tool：

1. `search_statute(query, applicable_date, limit=5)`：按问题检索法律条文，返回稳定条文 ID、摘要、版本和来源。
2. `get_statute(article_id, applicable_date)`：读取该日期下指定条文的完整文本、历史版本元信息与核验提示。

## 数据审计结论

- 351 条、5 份法律文件、5 个文本版本；各文件只包含**一个**完整条文快照。
- 351 条 `text_sha256` 均匹配；未发现重复的 (law_id, article_number, version_id)，各法条序号连续。
- 全部 `effective_to=null`；**不是完整历史版本数据库**。如果查询日期早于唯一存储版本的 `effective_from`，只返回 `version_unavailable_for_date`，不会将较新法条套到较早案件。
- 全部 `source_checked_at` 来源日期为 2026-05-21 左右；在此之后的日期，按当前数据只能给出 `unverified_after_snapshot`，不能认定那天适用的法律版本已被核实。
- `status=effective` 是**输入数据记录**，不是本工具独立的法律效力证明。
- 数据库只涵盖所给 5 部法律；没有劳动领域所有法规、全部地方规定以及其他相关解释的完整覆盖。

## 文件结构

```text
legalagent/tools/
├── legal_tools.py            # 两个版本感知工具（完整版，评测/人工查条用）
├── tools_schema.py           # 工具 JSON Schema + dispatch() 唯一分发入口
├── agent_tools.py            # 训练用精简版 search_compact / get_compact
├── rollout_demo.py           # 单题多轮 rollout（生成→解析→执行→回填）
├── baseline_toolcalling.py   # 开放问题基线（6 题）
└── demo.py                   # 最小运行示例，不需要模型

data/
├── labor_subset.jsonl        # 原始子集（351 行）
└── legal_statutes.sqlite3    # 构建后的只读快照
```

## 运行

需要 Python 3.10+，只依赖标准库。

```bash
# 在仓库根目录执行；所有默认路径由 legalagent/paths.py 统一给出
python -m legalagent.data.build_db          # 更新数据后必须重建
python -m legalagent.tools.demo             # 不需要模型，验证两个工具
python -m unittest discover -s tests -t . -p "test_legal_tools.py"
```

## 两个工具的调用方式

```python
from legal_tools import search_statute, get_statute

hits = search_statute("严重违反规章制度 解除劳动合同", "2024-01-01", limit=3)
assert hits["ok"]
print([(r["article_id"], r["score"]) for r in hits["results"]])

article_id = hits["results"][0]["article_id"]  # 不要自行编造 ID
article = get_statute(article_id, "2024-01-01")
print(article["article"]["text"])

# 历史覆盖不足：不能假装 2013 年版本适用于 2010 年
old = get_statute("labor_contract_law:39", "2010-01-01")
assert old["error"] == "version_unavailable_for_date"
```

工具输出均为 JSON-serializable 的字典。`article_id` 使用稳定规范 `law_id:article_number`（例如 `labor_contract_law:39`）；选中的 `version_id` 另行保存。以后补充历史版本时，Agent 无须改动 `article_id`，只需传入正确的 `applicable_date`。

### `search_statute` 输出

- `results[*].article_id`：供 `get_statute` 读取
- `results[*].snippet`：字符截断的原文摘要，不是完整法条
- `results[*].version_id`、`effective_from/to`、`source_url`：溯源信息
- `results[*].score`：**内部检索相似度分数，不是法律正确性 Reward**
- `coverage`、`warnings`：历史版本缺失或查询日期晚于来源核验日期时的告警

### `get_statute` 输出

- `article.text`：原始完整法条
- `article.text_sha256`：核验正文未被篡改
- `article.source_url`：来源链接
- `temporal_status=within_snapshot_date_window`：本地版本的存储有效期与给定日期匹配，但仍需判断复杂的过渡适用规则
- `temporal_status=unverified_after_snapshot`：日期晚于源数据核验时间，该版本是否仍在该日有效未核实
- `error=version_unavailable_for_date`：没有相应版本的本地文本，不能替代为当前版本

## 检索算法说明

默认检索器是纯 Python 的字符 2-gram + 领域术语归一化 + 显式法条引用匹配（351 条规模无需向量 DB）。包括少量确定性的词项扩展，例如 `辞退 -> 解除劳动合同`，`双倍工资 -> 二倍的工资`。**这是基本词项召回器，不是语义检索器。**

例如：

| 用户问题 | 预期条文 | 测试要求 |
|---|---|---|
| 严重违反规章制度解除劳动合同 | 劳动合同法第 39 条 | Top 3 |
| 违法解除劳动合同赔偿金 | 劳动合同法第 87 条 | Top 3 |
| 没签合同、双倍工资 | 劳动合同法第 82 条 | Top 1 |
| 劳动争议仲裁时效 | 调解仲裁法第 27 条 | Top 1 |
| 劳动合同法第三十九条 | 劳动合同法第 39 条 | Top 1 |

这只是检索器的 smoke test，不构成独立的法律正确率评估；真实训练前应建立人工核验的标准引用标签、Top-k Recall 和失败类别分析。

## TRL GRPO 的集成接口

`legal_tools.py` 中两个**模块级 Python 函数**都有 type hints、Google-style docstring，可作为 TRL `GRPOTrainer` 的 `tools` 参数：

```python
from trl import GRPOTrainer
from legal_tools import search_statute, get_statute

# 这是集成片段，而不是完整可运行训练脚本。
trainer = GRPOTrainer(
    model=model,
    args=grpo_config,
    train_dataset=train_dataset,
    reward_funcs=[reward_fn],
    tools=[search_statute, get_statute],
)
```

注意：

1. `train_dataset` 的 `prompt` 中必须告诉 Agent 案件的 `applicable_date`，工具不能自行取当前日期。
2. `reward_fn` 必须在最终答案中提取所引 `article_id`，独立地验证版本和引用是否支持结论；**不能把检索排序分直接当 Reward**。
3. 当前工具是无状态只读工具，适合 `tools=[...]`；若未来要追踪每条轨迹的调用次数、证据集或状态机，改用 `environment_factory` 封装环境。
4. 不同 TRL/transformers 版本接口会调整；先在对应版本做最小 rollout 测试。
5. 测试集与检索库存放的法条可以重叠，但**法律问题、事实模板和标准答案不能泄漏到测试集**。

## Tool Calling 基线

在数据与工具之上补了「模型真的会调工具吗」这一层，用来给后续 GRPO 提供 before/after 对照。

| 新增文件 | 作用 |
| --- | --- |
| `tools_schema.py` | 两个工具的 JSON Schema（给 Qwen chat template）+ 唯一分发入口 `dispatch()`，未知工具/缺参数都返回结构化错误 |
| `rollout_demo.py` | 单题 rollout：加载模型 → 生成 → 解析 `<tool_call>`（截断的 JSON 会自动补括号）→ 执行 → 回填 `<tool_response>` → 继续生成，最多 N 轮 |
| `baseline_toolcalling.py` | 固定 6 题基线：统计 tool_call_rate、引用是否真来自工具输出（grounded）、编造 ID 数 |
| `run_demo.ps1` | 设置 `KMP_DUPLICATE_LIB_OK` / 离线 / UTF-8 后调用 `rollout_demo.py` |

### 运行

```powershell
.\scripts\run_demo.ps1 --query "员工严重违反规章制度被辞退，公司需要支付赔偿金吗？" --date 2024-01-01
.\scripts\run_demo.ps1 --scripted --few-shot                        # 注入一次真实工具调用，只验证链路
python -m legalagent.tools.baseline_toolcalling --few-shot          # 6 题基线（用 .venv-ml 的解释器）
```

`--scripted` 已验证链路通：注入 `search_statute` → 返回 `labor_law:25`、`labor_contract_law:39` → 模型基于工具输出作答并回引这两个 ID。

### 怎么测你自己的基线


```powershell
python -m legalagent.tools.baseline_toolcalling --few-shot --download
```

输出 `results/baseline_report.json`，每题记录 `tool_called` / `tool_ids` /
`cited_ids` / `grounded_citations` / `hallucinated_citations`，可直接算三个率：
`tool_call_rate`、`grounded_citation_rate`、`hallucinated_ids`。

这套奖励针对的三类失败形态（也是这套基线要暴露的东西）：

1. **不调工具直接答**：凭记忆给结论，甚至给出从未检索过的 `article_id`；
2. **编 ID**：引用的 ID 不在工具返回集合里 —— `hallucinated_citations` 就是为这个准备的；
3. **复读兜底话术**：把系统提示里「本地数据不足」原样搬出来，同样没有先调用工具。

对应关系：`tool_use` 打第 1 类，`grounding` 打第 2 类，`gold` / `answer_points` 打结论与要点。


> 研究演示用途，不能将小型子集的“未检索到”当作法律上的“不存在”。
