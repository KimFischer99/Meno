分析完成。我以只读方式克隆了 `feat/personamem-e2e` 分支，通读了全部源码（`src/meno` 约 2400 行）、benchmark 管线、SPEC 与测试报告，并本地跑通了测试（**27 passed, 0.56s**，确定性 TestEmbedder，无外部依赖）。总体判断：**这是一个架构方向正确、工程纪律很好的 v1.0 试验骨架，但“记忆系统”的核心智能层（抽取、合并、时间演化）还停留在占位实现，当前它更像一个“带治理的 evidence 存取层”而非真正的 user modeling 层**。你的测试报告自己给出的 No-Go 判定是诚实且准确的。

---

## 一、架构上做对的事（值得保持）

| 决策 | 证据 | 评价 |
|---|---|---|
| Postgres 为 canonical、Qdrant 为可重建派生索引 | `db.py` + `rebuild_projection()` + `meno rebuild-projection` CLI | 正确的“真值源/索引”分离，换 embedding 模型有明确 replay 协议（projection version 命名 collection） |
| Transactional outbox 异步物化 embedding | `ingest()` 同事务写 `Outbox`，`process_outbox()` 消费 | 写路径不阻塞在 Google API 上，`skip_locked` 支持 PG 多 worker |
| Evidence-backed claim + audit hash chain | `ClaimEvidence`、`_audit()` 带 `prev_hash/current_hash`、`pg_advisory_xact_lock` 防分叉 | 每个 facet 可回查来源，这是相对“又一个向量库”的真正差异点 |
| 幂等语义 | `event_id + content_hash` 冲突检测、uuid5 确定性 claim id、batch 去重 | outbox 重试后 claim id 稳定，天然幂等 |
| 隐私默认失败方向 | fail-open personalization / fail-closed privacy；sensitive claim 不送 embedding provider；删除传播到向量 | 方向正确 |
| 评测纪律 | dataset sha256、run config、secrets 只从文件/环境读且不写入结果 JSON、e2e 与 retrieval-only 在**同一批 facets** 上对比分离“检索失败”与“作答失败” | 这是整个 repo 里最专业的部分 |

---

## 二、不足与风险（按影响排序）

### A. 设计层

1. **`_audit()` 是全局写瓶颈，且污染读路径**。`pg_advisory_xact_lock(固定 id)` 把**所有用户的所有审计写入串行化**；更严重的是 `retrieve()` 对每个选中的 facet 各插一条 audit（最多 12 条/次检索）——**读热路径在做事务性写**。你 VPS 报告里 retrieval p95 1663ms 超标，这个写放大是贡献因素之一。审计应该异步化/批量采样，而不是 inline。
2. **单一 revision 计数器承担了太多语义**。ingest bump、outbox 处理 bump、feedback/consent/deletion 都 bump 同一个 `UserRevision`，导致 e2e 管线只能用“revision 稳定 30 秒”这种启发式判断 drain（`_wait_for_processing` 的 docstring 自己承认了这是 hack）。缺一个**按用户的 pending outbox 计数/drain 游标端点**。
3. **认证模型只到 sidecar 边界**。单个 `MENO_API_TOKEN`，`user_id` 由调用方任意填写——持 token 者可读写任意用户。loopback 单用户下成立，但任何多用户/多 agent 试验都需要 per-user scoped token，这会牵动所有 API 语义，**应该在下一个试验周期前决定**。
4. ** extractor 版本升级会产生平行 claim 集**。`Outbox` 唯一键是 `(event_id, processor_version)`，提升 `MENO_EXTRACTOR_VERSION` 重放旧事件会生成新 derivation key 的新 claims，而旧版本 claims 仍 active——没有跨版本 supersede/清理协议。你做“改进抽取器”试验时**必然撞上这个坑**。
5. **时间语义没有进入向量索引**。`as_of` 只在 `_admit()` 里事后过滤 claim 的 valid_from/valid_to，Qdrant payload 里没有时间字段可下推。这是 LongMemEval temporal 类问题 Recall@10 只有 0.8645 的结构性原因之一。

### B. 实现层

6. **抽取器是占位符**（`extractor.py` 82 行）：正则单模式、每事件最多产出 1 个 candidate（preference 命中即 early return）、未命中则把**整条消息原文**存成 episodic claim——没有压缩、没有跨事件合并、没有矛盾检测（自动 supersede 不存在，只有显式 feedback 才 supersede）。这直接解释了 `track_full_preference_evolution` 和 `suggest_new_ideas` 是最弱项。`Entity/ClaimEntity/ClaimEdge` 表基本是死 schema（只有 feedback 路径写过一条 supersedes 边）。
7. **half-life 衰减与证据 reinforcement 脱钩**：`_effective_confidence` 只看 `valid_from` 年龄，同一偏好被反复确认也会衰减；而 `_score` 里的 evidence 项因为每个 claim 恒只有 1 条 evidence，实际是常数 0.6——死代码路径。
8. **feedback `correct` 不幂等**：replacement claim 用 uuid4，且不校验原 claim 是否已 superseded——同一纠正请求重放两次会产出两条 active 的替换 claim。同理 `consents` 是 append-only，重复授予产生重复行（读取侧只扫 revoked，功能上无害但语义脏）。
9. **SQLite 下 `with_for_update(skip_locked)` 被静默丢弃**（我实际编译验证过，生成的 SQL 里没有 FOR UPDATE）。dev 模式若起多 worker 会双消费 outbox。PG 下没问题，但这是 dev/prod 行为分叉点。
10. **`_render()` 的细节问题**：`rendered[: max_tokens*4]` 粗暴截断可能切掉 `</user_context>` 闭合标签；`user_id` 未转义直接插入 XML 属性位。
11. **Google embedder 无熔断**：`embed_query` 最多 5 次尝试、退避上限 10s，429 风暴下单次 retrieve 可挂起约 1 分钟才走 degraded 回退；而 Hermes plugin 的 `prefetch` 默认 timeout 只有 **0.8s**——和你实测的 retrieve p95 1.66s 直接矛盾，生产配置下 prefetch 会频繁超时。这两端需要一起调。
12. `hermes_plugin` 的 `on_memory_write` 产出 `role=system` 事件，但 `extract_claims` 直接忽略 system role——这条写入路径存了 event 却永远不产生 claim，要么是遗漏要么该注明是“仅存档”。

### C. 评测层

13. `correct_support_at_k = 1.0` 这个指标是**虚的**：token 重叠 + 12 个 facets 下几乎必然非零，它不构成“正确答案被证据支持”的严格度量，报告里建议降权或删掉。
14. 本地 e2e 的 0.579 建立在 TestEmbedder 的 hashing 词袋相似度上，`track_full_preference_evolution` 0.31→0.64 的增益部分是假嵌入器与 BM25 式打分的特性——README 已声明可比性边界（好），但做下一轮试验时要警惕**在这个后端上调参过拟合**。
15. Outbox 失败 20 次后进 `failed` 状态，**没有任何 requeue/DLQ 工具**（`process-once` 只处理 pending）。

---

## 三、下一步工程试验的具体建议

按“先解锁试验能力，再提升智能”排序：

1. **先做两个小端点/机制，试验效率会翻倍**：
   - `GET /v1/users/{id}/drain`（pending outbox 计数），替代 30 秒稳定窗口的 drain 判定——e2e 检索阶段会快一个数量级；
   - audit 写入移出 retrieve 热路径（异步批量或采样），再看 p95 是否回到 900ms 门槛内。
2. **抽取器升级前，先设计 claim 版本协调**：新 `extractor_version` 重放后，如何 deactivate/merge 旧版本 claims。否则你第一个抽取器对比试验就会被重复 claim 污染，数据不可信。
3. **把 temporal 下推进 Qdrant payload**（valid_from/valid_to 进 filter），这是最便宜的 LongMemEval temporal 提分手段。
4. **做真人 pairwise / shadow traffic 试验前先解决 per-user 认证**，否则评测数据与信任模型不匹配。
5. **embedding 相关两端联动调参**：provider timeout、retry、熔断与 Hermes `prefetch` 的 0.8s 超时是一根链条，用一次 Google 429 soak 同时测两端。
6. 保持现有的评测纪律（hash、config、secrets 隔离、retrieval cache）——这是 repo 最有复用价值的资产。

## 验证情况

- 只读分析：未修改仓库任何文件（clone 在 `/tmp/meno-analysis`）。
- `pytest`：27 passed（Python 3.14，SQLite + TestEmbedder 后端）。
- 复核了 `local-personamem-e2e-full.json`（589 题，aggregate 与 README 声称一致）。
- 实际编译验证了 SQLite 方言丢弃 `FOR UPDATE SKIP LOCKED`。
- 未运行：需要 Google 凭据/LLM endpoint 的 e2e 与 VPS 管线，Qdrant 后端的集成路径（测试全部走 memory store + fake embedder）。