# 劳动法小型子集

给 RL / agent 实验用的小型、可追溯法律子集：**5 部规范、351 条条文**，每一条都来自本机 `chinalaw` 数据库（官方来源 + 核查时间 + 内容哈希），不是模型记忆。

| 文件 | 内容 |
| --- | --- |
| `labor_subset.jsonl` | 351 条记录，一行一条 JSON，UTF-8 无 BOM |
| `manifest.json` | 每部法规的元数据、revision 列表、来源哈希、条数、统计 |
| `export_subset.py` | 生成脚本（调用 `chinalaw get --format json`，不直连 SQLite） |

## 收录范围

| law_id | 正式名称 | 层级 | 条数 | effective_from |
| --- | --- | --- | ---: | --- |
| `labor_law` | 中华人民共和国劳动法 | 法律 | 107 | 2018-12-29 |
| `labor_contract_law` | 中华人民共和国劳动合同法 | 法律 | 98 | 2013-07-01 |
| `labor_contract_law_regulation` | 中华人民共和国劳动合同法实施条例 | 行政法规 | 38 | 2008-09-18 ★ |
| `labor_dispute_arbitration_law` | 中华人民共和国劳动争议调解仲裁法 | 法律 | 54 | 2008-05-01 |
| `labor_dispute_interpretation_1` | 最高法关于审理劳动争议案件适用法律问题的解释（一） | 司法解释 | 54 | 2021-01-01 |

★ 这一部是抓取入库的（`gov_xzfgk` 源），入库时 `status=unknown`、`effective_at=null`，脚本用 `OVERRIDES` 按**国务院令第535号（2008-09-18 公布，自公布之日起施行）**补齐，并在记录里写明 `effective_from_basis`。不确定就删掉 `OVERRIDES` 里那一条，导出会如实留空。

## 字段映射（对应教程里的 JSON）

| 教程字段 | 本数据集 | 来源 |
| --- | --- | --- |
| `law_id` | `labor_contract_law` 等 slug | `export_subset.py` 的 `LAWS` 映射，稳定不随库变 |
| `article_number` | `39`（int） | `article.number` |
| `version_id` | `2c909fdd…@cc22abe84867` | `selected_revision.id`（法规 id + 内容哈希前缀） |
| `effective_from` | `2013-07-01` | `revision.effective_at`，缺失时回退 `law.effective_at` |
| `effective_to` | `null` | 下一条 revision 的 `effective_at`；末版取 `law.repealed_at`；现行版本为 `null` |
| `text` | 条文正文 | 入库清洗后的官方文本 |
| `status` | `effective` | `law.status` 映射：current→effective / amended→superseded / repealed→repealed；同时保留原始值 `law_status_raw` |
| `source_url` | 官方详情页 | `law.source_url` |
| `source_hash` | 64 位 SHA256 | `revision.content_hash`（**版本级**内容哈希） |

额外加了三个便于审计的字段：

- `text_sha256`：该条正文自身的 SHA256（记录级完整性校验，防篡改/防重复）；
- `source_checked_at`：来源核查时间（本库数据为 2026-05-21）；
- `part`：条文所在编 / 章，便于按章节切分训练样本。

## 关于 `effective_to`（教程特别提醒的那点）

这里**没有**从 git 提交时间或发布新闻推断任何日期：

1. 生效日只取 `revision.effective_at`（来自全国人大 / 国务院官方记录）；
2. 失效日只取「下一条 revision 的生效日」或 `law.repealed_at`；
3. 现行有效且未被废止 → `null`，不猜。

当前 chinalaw 的内置数据对这 5 部法各自只有 **1 个 revision**（历史版本未随包分发），所以 351 条记录的 `effective_to` 全是 `null`。要做跨版本训练（例如 2008 版 vs 2013 版劳动合同法），需要先补历史版本：

```powershell
chinalaw history 劳动合同法 --format md      # 看官方版本链
chinalaw fetch 劳动合同法 --force            # 重新抓取（源站有历史则一起入库）
python -m legalagent.data.export_subset --include-history   # 按每个 revision 的 effective_at 用 --as-of 导出
```

## 已知缺口

- **《劳动争议司法解释（二）》（2025-09-01 施行）不在库里**：`court_gongbao` 搜索返回 502，`court_main` 只命中新闻发布会稿，没有正式条文。需要时用 `chinalaw fetch "…解释（二）" --source court_gongbao` 重试，或从最高法公报页手工核对后 `fetch`。
- **《劳动合同法实施条例》**只能从 `gov_xzfgk` 抓到：`flk_npc` 路径因正文托管在 `flkoss.obs-bj2.cucloud.cn`（不在来源白名单）而失败。这是上游适配器的限制，不是网络问题。
- 本子集是**现行版本快照**，不含被修改前的旧文本。

## 重新生成

```powershell
# 在仓库根目录执行
python -m legalagent.data.export_subset
```

脚本默认用仓库内的 `chinalaw-cli\.venv\Scripts\python.exe -m chinalaw`（路径由 `legalagent/paths.py` 的 `ROOT` 推出，不再写死绝对路径），也可用环境变量 `CHINALAW_PYTHON` 指定解释器。注意该 venv 的 editable 安装 `.pth` 指向初次 `pip install -e` 时的目录，仓库被移动或复制后需要重装一次才能真正指向本副本。数据源是本机库 `C:\Users\24190\.chinalaw\chinalaw.db`；换库用 `chinalaw --db <path>` 或设置 `CHINALAW_DB`。

## 抽查方法

```powershell
# 随便挑一条，和官方页面比正文
chinalaw article 劳动合同法 39 --format card
# 校验数据集里的哈希与库内文本一致
python -c "import json,hashlib;r=[json.loads(l) for l in open('labor_subset.jsonl',encoding='utf-8')][0];print(hashlib.sha256(r['text'].encode()).hexdigest()==r['text_sha256'])"
```
