# PersonaMem e2e 阶段之后：修订版行动计划

> 依据 `tests/code_review.md`（只读源码 review，pytest 27 passed 复核）+ 首轮 e2e 结果综合。
> 目标：先把检索/评测管线打顺，再升级真正缺的"演化"智能层；生产路径数字（Google embedding）先于 extractor 升级定稿。

## 状态基线（已验证）
- PersonaMem 端到端 LLM-answer 评测已落地（`benchmarks/run_personamem_e2e.py` + `serve_local_dev.py`）。
- 本地全量（hashing TestEmbedder）：answer accuracy 0.579 vs 旧 scorer 0.409；track_full_preference_evolution 0.31→0.64；suggest_new_ideas 0.02→0.28。
- ⚠️ 本地增益部分建立在 hashing 词袋假嵌入器上，勿在本地后端调参过拟合；以 Google 生产路径全量为准。

## 阶段一：解锁试验能力 + 修检索管线（先做）

1. **Outbox 三件套**（原 P0，review 扩深）
   - 指数退避 + 增量分批重嵌入（不整批 rollback 后全量重试）。
   - 加熔断：embedder 现最多 5 次、退避上限 10s，429 下单次 retrieve 可挂约 1 分钟才 degraded（review #11）。
   - failed 状态补 DLQ/requeue 工具：现失败 20 次进 failed 即无路可走（review #15）。
   - **两端联动**：Hermes plugin `prefetch` 超时 0.8s 与实测 retrieve p95 1.66s 矛盾，生产下会频繁超时；provider timeout/retry/熔断 与 Hermes 侧超时是一根链条，一次 Google 429 soak 同时测两端（review #11）。
2. **`GET /v1/users/{id}/drain`**：pending outbox 计数端点，替代"revision 稳定 30s"的启发式 drain 判定（review #A-2）。e2e 检索阶段提速一个数量级。
3. **audit 移出 retrieve 热路径**：现 retrieve 对每个选中 facet 各插一条审计（最多 12 条/次），读热路径在做事务性写；与 VPS p95 1.66s 超标直接相关（review #A-1）。改为异步批量或采样。
4. **temporal 下推进 Qdrant payload**：valid_from/valid_to 进 filter（现只在 `_admit()` 事后过滤）。最便宜的 LongMemEval temporal 提分手段（review #A-5，Recall@10=0.8645 的结构性原因之一）。

## 阶段二：extractor 升级（原 P2 根因，前置必须先解）

5. **先设计 claim 版本协调协议**（review #A-4）：新 `extractor_version` 重放旧事件会生成平行 claim 集、旧 claims 仍 active，无跨版本 supersede/清理协议。不解决就开工，第一个抽取器对比试验会被重复 claim 污染，数据不可信。**这是下一阶段真正的 gate。**
6. **重写 extractor**（review #B-6）：现为占位符（正则单模式、每事件最多 1 个 candidate、未命中整条原文存成 episodic、无压缩/跨事件合并/矛盾检测，自动 supersede 不存在）。这是 track_evolution 与 suggest_new_ideas 最弱的根因，比"缺 coverage 信号"更深一层——整个"演化"语义尚未建立。
7. **half-life 与证据 reinforcement 挂钩**（review #B-7）：`_effective_confidence` 只看 valid_from 年龄，同一偏好反复确认也衰减；`_score` 的 evidence 项因每 claim 恒 1 条 evidence 是死代码常数 0.6。

## 生产复测（P1，放 extractor 升级后）

8. Google 配额重置后跑生产全量（new-vm，Google embedding，带 pacing），拿短效可比真数。
9. 检核 audit 异步化后的 retrieval p95 是否回到 900ms 门槛内。

## 不紧急但该决策/记下
- **per-user 认证**（review #A-3）：现单 `MENO_API_TOKEN`、user_id 由调用方任意填，持 token 可读写任意用户。loopback 单用户成立，任何多用户/多 agent 试验前必须决定，牵动全部 API 语义。
- **feedback `correct` 不幂等**（review #B-8）：replacement 用 uuid4、不校验原 claim 是否已 superseded，重复重放产出双 active 替换 claim；consents append-only 重复授予。
- **render 细节**（review #B-10）：`rendered[: max_tokens*4]` 粗暴截断可能切掉 `</user_context>` 闭合标签；user_id 未转义插入 XML 属性位。
- **on_memory_write role=system 事件**（review #B-12）：extract_claims 忽略 system role，写入路径存 event 却永不产 claim——应注明"仅存档"或补逻辑。
- **SQLite dev/prod 行为分叉**（review #B-9）：`with_for_update(skip_locked)` 在 SQLite 下被静默丢弃，dev 多 worker 会双消费 outbox（PG 下无问题）。
- **评测纪律保留**（review #C-13/14）：`correct_support_at_k=1.0` 是虚指标（token 重叠 + 12 facets 几乎必然非零），建议降权或删；本地 e2e 增益防过拟合。

## 执行顺序
先 1→4（阶段一），再做 5→6→7（阶段二，5 是 gate），再 8→9（生产复测）。每步保持现有评测纪律（dataset hash、config、secrets 隔离、retrieval cache）。
