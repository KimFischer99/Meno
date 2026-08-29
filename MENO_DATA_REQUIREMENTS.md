# Meno v2.0 — 真实语料与流量接入需求书

> 面向对象：拥有生产 Hermes 环境的操作者（在另一台机器上，已运行 4 个月）
> 目的：把 Gate Charter 里剩下的 5 项缺口（#13 #14 #15 #16 #17）换成可执行的数据需求
> 依据：`Meno_v2.0_Gate_Charter.md`（已审定）、`Meno_SPEC.md:1340-1372`
> 状态：需求已定；导出实现由操作者按真实日志结构完成，本地用 `benchmarks/validate_external_corpus.py` 验收

## 0. 先读这一段：一个必须先说清的限制

**4 个月的历史日志能解决 #13 #14（检索语料），但不能追溯解决 #15（校准）。**

原因是机制性的，不是配置问题。`run_real_feedback_calibration.py` 的配对逻辑是：把每条 feedback 所在的 revision `r`，与**该用户在 `r-1` 时刻已经物化的 User Token snapshot** 配对，用当时的 `preference_distribution` 作为「事前预测」，用 feedback 结果作为「事后标签」。

这是一个**前瞻性（prospective）**度量：预测必须在结果之前就已经落盘。历史日志里没有 Meno 的 snapshot —— 那时 Meno 根本没在运行。事后把日志灌进 Meno 再生成 snapshot，得到的是「用今天的模型解释昨天的结果」，那是回溯拟合，不是校准。

**所以两件事要分开安排：**

| 缺口 | 用历史日志 | 需要前瞻运行 |
|---|---|---|
| #13 #14 检索语料 | ✅ 可以 | — |
| #15 校准 | ❌ 不行 | ✅ 需要 Meno 在线跑一段时间 |
| #16 真人 pairwise | ❌ 不行 | ✅ 需要真人评估 |
| #17 社群先验 | ✅ 可作为输入 | — |

不要试图用历史日志凑 #15 的 100 条样本。若这样做了，`calibration_status` 仍会是 `uncalibrated`，而且报告会通过一个不成立的判据 —— 这正是本项目反复强调要避免的那类假证据。

---

## 1. 第一部分：检索语料（解决 #13 #14）

### 1.1 为什么现有语料不行

已实测（`benchmarks/probe_curated_viability.py`，结论 `CORPUS_UNSUITABLE_QUERY_SIGNAL`）：

```text
决定性 token 出现在 query 里的比例   0.0871   ← 与当初的 0.087 陷阱无法区分
随机基线                            0.0742
```

PersonaMem 的问题不是规模，是**问句不指名自己所问的对象**。「我最近参加了一个活动」这种问法，无法从 query 侧定位到该检索哪条历史 —— 任何只看 query + 用户历史的检索器都有同一个天花板。

### 1.2 因此语料的核心标准只有一条

> **问句必须包含足以定位答案的词。**

真实 agent 对话天然满足这一点的比例远高于合成 benchmark，因为真实用户会说「上次我们讨论的那个 Postgres 连接池配置」而不是「我最近遇到一个技术问题」。

**这条标准是可以在投入标注前先验证的**，见 §1.6。**请务必先验证再标注。**

### 1.3 需要导出什么

两个文件，JSONL 格式，UTF-8。

#### 文件 A：`sessions.jsonl` — 对话历史

每行一个会话：

```json
{
  "session_id": "<稳定匿名 ID>",
  "user_key": "<稳定匿名用户 ID>",
  "messages": [
    {
      "index": 0,
      "role": "user",
      "text": "...",
      "occurred_at": "2026-04-12T09:31:00Z"
    }
  ]
}
```

字段要求：

| 字段 | 要求 | 为什么 |
|---|---|---|
| `session_id` | 稳定、匿名、同一会话固定 | 用于分组与去重 |
| `user_key` | 稳定、匿名、跨会话可关联同一人 | Meno 的核心是跨会话用户状态；每会话换 ID 会让 preference evolution 完全不可见 |
| `index` | 会话内从 0 递增、无空洞 | 决定 event 顺序与「问题发生在第几轮」 |
| `role` | `user` / `assistant` / `system` 之一 | Meno 只从 `user` 轮抽取 claim；assistant 轮不作为用户证据（这是安全设计） |
| `text` | 原文，非摘要 | 摘要会丢掉 claim 的原始措辞，evidence 就失去意义 |
| `occurred_at` | ISO 8601 带时区（UTC 最好） | 时间衰减、stale 判定、supersede 顺序全依赖它。**不能用导出时间填充** |

**`occurred_at` 必须是消息的真实发生时间。** 如果全部填成同一个导出时刻，时间漂移相关的行为（#10 stale-active、preference evolution）就全部失效。4 个月的真实时间跨度正是这份数据最有价值的地方。

#### 文件 B：`queries.jsonl` — 评测问句与标注

每行一个待评测查询：

```json
{
  "query_id": "<稳定 ID>",
  "session_id": "<对应 sessions.jsonl>",
  "user_key": "<对应 sessions.jsonl>",
  "asked_at_index": 42,
  "query": "上次我们说的那个连接池配置，我当时定的是多少？",
  "task_type": "technical_recall",
  "relevant_message_indices": [17, 18],
  "expected_decision": "inject"
}
```

| 字段 | 要求 |
|---|---|
| `asked_at_index` | 该问句发生在会话第几轮。**只有 index < 此值的消息可作为证据** —— 防止用未来信息回答过去的问题 |
| `relevant_message_indices` | 回答此问句真正需要的消息 index 列表。这是 gold label |
| `expected_decision` | `inject`（该注入记忆）/ `abstain`（无相关记忆，不该注入）|

### 1.4 标注约束（这条最容易违反）

> **标注者只能看 query 和 `asked_at_index` 之前的用户历史。不能看答案，不能看模型输出。**

原因：如果标注者看着正确答案标「哪些消息相关」，标出来的是 oracle —— 而线上系统看不到答案。本项目已经踩过这个坑：一个读答案的 probe 给出 0.88 上限，据此做的优化实际是 −0.0102。

**必须包含 abstain 样本。** SPEC 明确警告不可只测 Recall：

> 对过度推断必须单独设置 negative benchmark，而不能只测 Recall。一个「总能给出丰富画像」的系统很可能在离线 demo 中看起来更聪明，却实际更危险。

**建议 abstain 占 20–30%** —— 即「用户历史里确实没有相关信息，正确行为是不注入」的问句。只测 Recall 会奖励一个永远输出一堆东西的系统。

### 1.5 需要多少量

| 项目 | 数量 | 依据 |
|---|---|---|
| 标注 query | **≥ 139** | 分辨 Recall@10 的 0.90 与 0.85（95% 置信）：`(1.96/0.05)² × 0.9 × 0.1` |
| 其中 abstain | 30–40 条（占 20–30%） | negative benchmark |
| 会话数 | ≥ 30，每人 ≥ 2 个会话 | 跨会话用户状态是 Meno 的核心 |
| 独立用户 | ≥ 10 | 少于此，个体差异会主导结果 |

**139 是算出来的，不是拍的。** 低于这个数无法支撑 SPEC 的 0.90 阈值 —— 与项目其他 lane 的样本分辨率规则同源。

### 1.6 先验证，再标注（**不要跳过这一步**）

标注 139 条要花不少人力。先用 **20–30 条**做预检，确认这份语料不是又一个 0.087：

```bash
# 1. 先导出小样本（20-30 条 query 即可），传到本仓库所在机器
# 2. 校验格式
PYTHONPATH=. .venv/bin/python benchmarks/validate_external_corpus.py \
  --sessions /path/to/sessions.jsonl \
  --queries /path/to/queries.jsonl \
  --output /tmp/corpus-validation.json
```

校验器会报出 **`query_signal`** 这一项。判读标准：

| query_signal | 含义 | 行动 |
|---|---|---|
| < 0.15 | 与 0.087 陷阱同级 | **停。不要标注。**换取样方式（见 §1.7） |
| 0.15 – 0.35 | 弱但可能有用 | 谨慎；先扩到 50 条再看 |
| > 0.35 | 显著优于现有语料 | 可以投入 139 条标注 |

这一步几乎零成本，能避免一次昂贵的错误。

### 1.7 如果预检不通过

不是语料没救，通常是取样偏了。可调整方向：

- **优先取「明确回指」的问句** —— 含「上次」「之前那个」「我说过的」「我们讨论的」等回指词的轮次，天然带定位信息
- **避开纯开放式闲聊轮** —— 「今天怎么样」这类无法定位任何历史
- **优先取任务型会话** —— 技术调试、项目推进类对话的问句通常指名对象；情感闲聊类通常不指名

### 1.8 一个完整的最小例子

`sessions.jsonl`（一行一会话，此处展开便于阅读）：

```json
{
  "session_id": "s-7f3a",
  "user_key": "u-2c91",
  "messages": [
    {"index": 0, "role": "user",
     "text": "帮我看下这个 API 网关的超时设置",
     "occurred_at": "2026-04-12T09:31:00Z"},
    {"index": 1, "role": "assistant",
     "text": "可以，你目前的配置是什么？",
     "occurred_at": "2026-04-12T09:31:20Z"},
    {"index": 2, "role": "user",
     "text": "我把 pgbouncer 的 max_connections 定成了 250，配合 8 个 worker",
     "occurred_at": "2026-04-12T09:33:00Z"},
    {"index": 3, "role": "assistant",
     "text": "250 对 8 个 worker 偏高，建议观察连接饱和度。",
     "occurred_at": "2026-04-12T09:33:40Z"}
  ]
}
```

`queries.jsonl`：

```json
{"query_id": "q-001", "session_id": "s-7f3a", "user_key": "u-2c91",
 "asked_at_index": 4,
 "query": "我之前给 pgbouncer 的 max_connections 定的是多少？",
 "task_type": "technical_recall",
 "relevant_message_indices": [2],
 "expected_decision": "inject"}
{"query_id": "q-002", "session_id": "s-7f3a", "user_key": "u-2c91",
 "asked_at_index": 4,
 "query": "我出差偏好哪家航空公司？",
 "task_type": "travel_preference",
 "relevant_message_indices": [],
 "expected_decision": "abstain"}
```

注意 `q-001` 为什么是好样本：问句里出现了 `pgbouncer`、`max_connections` —— 这些词让检索器**有可能**定位到 index 2。反例是「我之前配的那个参数是多少？」，那样连人都不知道该找哪条。

`q-002` 是必需的 abstain 样本：历史里没有任何航空偏好，**正确行为是不注入**。

### 1.9 校验器会检查什么

`benchmarks/validate_external_corpus.py` 会拦下这些（已实测均能捕获）：

| 检查 | 为什么重要 |
|---|---|
| `relevant_message_indices` 指向 ≥ `asked_at_index` 的消息 | 用未来信息回答过去的问题，标签无意义 |
| `inject` 却没有 relevant 索引 / `abstain` 却有 | 判定与标签自相矛盾 |
| message index 有空洞 | event 顺序不可靠 |
| 整个会话共用一个时间戳 | 说明用了导出时刻，时间衰减失效 |
| 缺 `occurred_at` | 同上 |
| `user_key` 与所属会话不一致 | 跨会话身份是 Meno 的核心 |
| 重复 `session_id` / `query_id` | 静默覆盖数据 |
| JSON 行损坏 | 报出行号，不整体崩掉 |
| 凭据/邮箱/手机号/身份证/私钥 | 隐私兜底（**报位置不报内容**，所以报告本身可安全传阅） |

两点行为值得知道：

- **结构错误在 `--sample` 模式下也会导致失败退出**。sample 模式只放宽「数量不足」，不放宽正确性。
- **`query_signal` 只统计 `inject` 样本**。abstain 样本没有证据可定位，算进去会把好语料压成坏语料。同时它只看「标注消息里有、而会话其余部分没有」的**区分性 token** —— 否则任何重复会话里常用词的问句都会虚高。



### 1.10 隐私要求（重要）

这份数据会离开生产环境，**导出前必须处理**：

- **假名化**：`user_key` / `session_id` 用稳定假名（如 `HMAC(真实ID, 本地密钥)`），映射表**留在你那台机器**，不要一起传
- **清除凭据**：正文里的 API key、token、密码、私钥、连接串必须移除或替换成占位符
- **清除直接标识**：真实姓名、手机号、邮箱、身份证、住址、银行卡
- **判断敏感话题**：涉及健康、宗教、政治、性取向、生物特征的会话，建议整段排除（Meno 对这类默认 deny，留着也用不上）
- **不要提交进仓库**：本仓库 `.gitignore` 已忽略 `artifacts/benchmarks/raw/`，把语料放这里

校验器会做一次凭据/邮箱/手机号的正则扫描并报告命中，但**它是兜底，不是保证** —— 不要依赖它替你做合规判断。

---

## 2. 第二部分：流量接通（解决 #15 #16）

### 2.1 当前状态

实测（2026-08-27，VPS 上）：

```text
14 个数据库 meno_feedback        全部为 0
~/.hermes/sessions               为空，30 天零活动
引用 meno 的 Hermes config       0 个（plugin 已注册但未接线）
```

`run_real_feedback_calibration.py` 已经写好并冻结了 gate，但**零数据，且不存在产生数据的链路**。这不是「慢但可控」，是开放式等待。

### 2.2 #15 需要什么

**必须让 Meno 在生产 Hermes 旁边真实运行一段时间**，前瞻性地积累 feedback。

接线方式（Meno 作为 sidecar，不改 Hermes core loop）：

```bash
# 在生产 Hermes 的 profile .env 中
MENO_API_URL=http://127.0.0.1:8765
MENO_API_TOKEN=<至少 32 字符的随机串>
MENO_USER_ID=<该 profile 的稳定用户标识>
```

Meno 侧需要开启物化相关的 flag（否则不会生成 snapshot，也就无法配对）：

```bash
MENO_USER_TOKEN_MATERIALIZATION_ENABLED=true
MENO_PREFERENCE_DISTRIBUTION_ENABLED=true
MENO_CONTEXT_ACTIVATION_ENABLED=true
```

**注意这几个 flag 目前默认关闭，且开启前应在非生产环境验证。** Meno 的设计保证 sidecar 挂掉不阻断 Hermes（已实测：dead port 与 hanging server 两场景，7 个 provider 入口全部非阻塞），但开启线上 flag 属于生产变更，请你自己决定时机。

冻结 gate 的要求：

| 条件 | 阈值 |
|---|---|
| paired preference outcomes 总数 | ≥ 100 |
| holdout 样本 | ≥ 30 |
| holdout 正例 / 负例 | 各 ≥ 10 |
| pairing coverage | ≥ 0.99 |
| Brier | ≤ 0.20 |
| ECE | ≤ 0.10 |
| 相对 uncalibrated extractor 的改善 | Brier 与 ECE 各 ≥ 0.02 |
| `calibration_status` | 不再是 `uncalibrated` |

**关键点：需要 ≥ 10 条负例**，即用户明确 `reject` 或 `correct` 了 Meno 记住的偏好。如果用户从不纠正，这个 gate 永远不满足 —— 所以 UI 上要让「这条记错了」足够容易点。

这需要多久取决于真实交互量，无法预估。**它是开放式的。**

### 2.3 #16 需要什么

`Utility: human pairwise preference — personalized > generic`。SPEC 自己引 Re-Centering Humans 的结论：自动评价与真人判断存在明显落差，所以**这一项组件级 oracle 永远替代不了**。

最小可行形式：

- 同一问题准备两个回答：一个注入 Meno 的 `rendered_context`，一个不注入
- 盲测（评估者不知道哪个是哪个），左右随机
- 真实用户本人评估（不是第三方标注员 —— 个性化的价值只有本人能判断）
- 记录成对偏好，统计显著性

样本量按二项检验估：要检出 65% 的偏好率显著优于 50%，约需 **80–100 对**。

### 2.4 顺带说明 #17

`community_memberships`（社会语义群 soft membership）在代码里**完全未实现**，属架构缺层。它不依赖你的数据 —— 是纯工程工作，可以独立立项推进。

但你的多用户真实语料对它有价值：SPEC 的论断是「语义群为先验、个人历史为残差」，验证它需要**多个用户**，才能看出「社群通常偏好 A，但该用户明确偏好 B 时个人残差覆盖先验」。所以 §1.5 里「独立用户 ≥ 10」这条对 #17 也有用。

---

## 3. 交付清单

拿到本文档后，建议顺序：

1. **先导出 20–30 条 query 的小样本** → 跑 `validate_external_corpus.py` → 看 `query_signal`
   - `< 0.15` 就停下调整取样，不要继续
2. 预检通过 → 补到 **≥ 139 条标注 query**（含 20–30% abstain），严格遵守「不看答案」约束
3. 语料放进 `artifacts/benchmarks/raw/`（已 gitignored），再跑一次完整校验
4. **流量接通单独安排** —— 与语料无关，且 #15 #16 无法用历史数据追溯

## 4. 一句总结

- **语料**：能解决 #13 #14。核心标准是「问句指名自己所问的对象」，且**必须先用小样本预检**。
- **流量**：#15 #16 必须前瞻性运行，历史日志无法追溯。这是开放式等待，不是排期问题。
- **不要**：不要用历史日志凑 #15 的样本；不要让标注者看答案；不要只测 Recall 而不设 abstain 样本。
