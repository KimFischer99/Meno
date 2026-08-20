# Claim 版本协调协议设计（阶段二 GATE，任务 5）

> 状态：待 Trism/Anoki 审核。通过前不动任务 6/7 实现代码。
> 依据：`docs/stage2-task-brief.md` 第 5 项、`tests/code_review.md` #A-4、以及对
> `src/meno/{db,service,extractor,vector,config,cli,api}.py` 与 `benchmarks/run_personamem_e2e.py` 的实读。

## 0. 现状代码事实（设计的出发点）

1. **重放入口不存在**：`ingest`/`ingest_many` 对已存在的 `event_id` 直接返回
   `idempotent_replay=True`，**不会**新建 Outbox 行（service.py:91-103, 177-188）。
   Outbox 唯一键是 `(event_id, processor_version)`（db.py:111），schema 上允许同一事件
   挂多版本行，但目前没有任何代码路径为旧事件补插新版本的 Outbox 行。
2. **derivation_key 已含版本**：`sha256("{event_id}:{processor_version}:{candidate_index}")`
   （service.py:281-283）。bump `MENO_EXTRACTOR_VERSION` 后重放必然生成**新 claim id**
   （`_claim_id` 是 derivation_key 的 uuid5），旧版本 claims 保持 `active` —— 即 #A-4
   所说的平行 claim 集。
3. **版本戳有误标风险**：worker 建 claim 时写 `extractor_version=self.settings.extractor_version`
   （service.py:313），但 derivation_key 用的是 `row.processor_version`。版本切换窗口内
   积压的旧版本行会被新代码处理并盖上新戳，lineage 失真。
4. **已有 supersede 机制（仅 feedback 路径在用）**：旧 claim `status="superseded"` +
   `valid_to=now` + 新 claim `supersedes_id` 回指 + `ClaimEdge(relation_type="supersedes")`
   + Qdrant `delete_claim(old_id)`（service.py:561-625）。没有 `superseded_by` 正向指针，
   没有 supersede 原因字段。
5. **检索侧已有 canonical 兜底**：`retrieve` 用 Qdrant 命中 id 回 Postgres 取 claim，
   `_admit()` 再校验 `status=="active"`（service.py:451-459, 884）。所以 Qdrant 残留
   已 supersede 的点不会漏进结果——Postgres 是真值源，Qdrant 只是派生索引
   （与 code_review 第 9 行结论一致）。`rebuild_projection()` 只重放 active 非敏感
   claim，是现成的修复路径。
6. **无 schema 迁移工具**：只有 `Base.metadata.create_all()`（db.py:213），对已有库
   不会 ALTER。生产要求 Postgres、开发用 SQLite（config.py:205-208），两条方言都要兼容。
7. **评测纪律现状**：报告已记录 `questions_sha256`/`contexts_sha256`
   （run_personamem_e2e.py:171-172）；retrieval cache 只按 `run_id` + question_id 序列
   校验（:460-473）；`user_id` 与 `event_id` 都嵌入 `run_id`（:216, 225-230）——
   不同 run 天然是不同用户命名空间。报告 **未记录** extractor_version。
8. **占位 extractor 每事件至多 1 个 candidate**，`semantic_channel` 只有三种粗值
   （extractor.py），同用户同内容重复事件今天会产生多条 active 重复 claim。

## 1. 核心概念：semantic_key 与全局不变式

**定义**：`Claim.semantic_key = sha256("{user_id}:{kind}:{semantic_channel}:{normalize(value)}")`，
`normalize` = casefold + 折叠空白 + 去首尾标点（纯标准库，确定性）。它回答"（user, 语义）"
里的"语义"：同一用户、同一类别、同一通道、同一规范化取值 = 同一条事实。

- v1（占位 extractor）按上式由 service 层统一计算，extractor 不需要改接口。
- 任务 6 重写 extractor 后，允许 candidate 自带 canonical key（如实体-槽位键），
  协议不变：**键的方案可以随版本演进，不变式不变**。

**全局不变式（本协议的核心承诺）**：

> 任意时刻，每个 `(user_id, semantic_key)` 至多一条 `status='active'` 的 claim。

**强制手段**：部分唯一索引（PG 与 SQLite≥3.8 均支持 partial index）：

```sql
CREATE UNIQUE INDEX IF NOT EXISTS uq_claim_active_semantic
ON meno_claims (user_id, semantic_key) WHERE status = 'active';
```

DB 层硬保证 + service 层协调逻辑（下述）保证正常路径不撞索引；撞索引即事务失败 →
Outbox 行保持 pending → 重试时走幂等路径收敛（见 §5）。

**例外（显式决策点 D3）**：`source_type='explicit_feedback'` 的 claim 是人显式纠正，
**永不**被重放协调自动 supersede；重放若产出与其同键的 claim，跳过创建并记 audit。
理由：用户纠正必须压过机器抽取，否则 feedback 语义被版本升级悄悄推翻。

## 2. 问题一：重放语义

### 2.1 重放入口（新增，实现属任务 6 阶段）

新 CLI：`meno reprocess --extractor-version V [--user-id U] [--limit N]`：

1. 对目标事件集（默认全部，可按 user 过滤）补插 `(event_id, V)` 的 Outbox 行，
   `ON CONFLICT (event_id, processor_version) DO NOTHING` —— 重复执行幂等。
2. 同时将目标事件集上**其他版本仍为 pending** 的 Outbox 行置为 `status='cancelled'`
   （新状态值，避免版本切换窗口内旧行被新代码误处理，消除 §0.3 的误标）。
3. 不触碰 `processed/failed` 行（保留历史）。

### 2.2 worker 处理语义（对 `_process_outbox_chunk` 的协议化修改）

- worker **只处理** `processor_version == settings.extractor_version` 的行
  （select 加过滤）。这保证"处理代码版本 = 行版本 = claim 戳"三者一致；
  claim 的 `extractor_version` 改为盖 `row.processor_version`（修 §0.3）。
- 对事件 E 在版本 V 下产出的每个 candidate，按 §3 的键级协调规则落库。
- 事件级清理：E 处理完后，查"从 E 派生（ClaimEvidence 含 E）且 `extractor_version != V`
  且仍 active"的旧 claim：若其 semantic_key 不在本次产出键集里 → 以
  `reason='extractor_withdrawn'` supersede（新版 extractor 不再从该事件得出该事实）。
- 重放期间 `valid_from` 仍取 `event.occurred_at`（现状不变），时间线语义保持；
  被 supersede 的旧 claim `valid_to=now`。as_of 历史查询：旧 claim 因 `status!='active'`
  被 `_admit` 过滤，新 claim `valid_from` 等于原事件时间 —— 历史视图干净切换，无双影。

### 2.3 新旧 claim 的关系

- 同键替代：新 claim `supersedes_id` 指主前任，每个被替代者写 `superseded_by_id=新id`
  + `ClaimEdge(relation_type='supersedes')`（沿用 feedback 路径已有图式）。
- 跨版本键方案变化（任务 6 后可能出现）：旧键不被新产出覆盖 → 靠 §2.2 事件级清理
  以 `extractor_withdrawn` 收回。滚动重放期间允许短暂跨方案并存，重放完成后收敛。
- 部分重放（只重放若干事件）只影响这些事件派生的 claim，其余旧 claim 不动。

## 3. 问题二：supersede / 清理的字段与状态

### 3.1 Claim 模型新增字段（schema 变更，需 Trism 批准）

| 字段 | 类型 | 说明 |
|---|---|---|
| `semantic_key` | `String(71)`, NOT NULL DEFAULT `''` | §1 定义；`''` 仅为迁移过渡，代码路径永远写真键 |
| `superseded_by_id` | `String(64)`, FK→`meno_claims.id`, NULL | 正向指针：谁替代了我（audit/lineage O(1) 查询） |
| `superseded_reason` | `String(32)`, NULL | `version_upgrade` / `extractor_withdrawn` / `dedupe` / `feedback_correct` |

不新增 status 取值：复用现有 `active/superseded/rejected`（`revisions()` 计数、
检索过滤、Qdrant payload 过滤都已认识这三个值）。

### 3.2 键级协调规则（worker 落库时，同事务）

对版本 V 为事件 E 产出的 candidate（键 K、用户 U）：

1. `derivation_key` 已存在 → 跳过（同版本幂等重放，现状逻辑保留）。
2. 存在 active 的 `(U, K)` claim 且 `extractor_version == V`（同版本同键，如重复内容
   事件）→ **不新建 claim**，把 E 追加为现有 claim 的 `ClaimEvidence`
   （`uq_claim_event` 防重），`reason` 记 `dedupe`。这顺带为任务 7 的
   evidence reinforcement 积累真实证据数。
3. 存在 active 的 `(U, K)` 且版本更旧 → 新建 claim，旧 claim 同事务内
   `status='superseded', valid_to=now, superseded_by_id=新id, superseded_reason='version_upgrade'`，
   补 `ClaimEdge`，Qdrant 删旧点、插新点。
4. 命中 active 的 explicit_feedback 同键 claim → 跳过创建 + audit（§1 决策 D3）。
5. 事件级清理（§2.2）收回 `extractor_withdrawn`。

**事务内顺序**：先 UPDATE 旧 claim 失效并 `flush()`，再 INSERT 新 claim ——
PG 唯一索引非延迟检查，必须此序；Qdrant 操作维持现状"commit 前执行、按点幂等"。

### 3.3 feedback 路径对齐

`feedback(correct)` 现有逻辑保留，补三笔：旧 claim 写 `superseded_by_id` +
`superseded_reason='feedback_correct'`；replacement claim 计算并写 `semantic_key`
（新取值 → 新键，天然不与旧键冲突）。`reject` 不变。

### 3.4 Qdrant payload：不变

点以 claim_id 为键，supersede = 删点，新 claim = 插点。payload
（user_id/status/projection_version/valid_from_ts/valid_to_ts）不加字段、不加索引。
依据：检索正确性由 Postgres canonical 兜底（§0.5），payload 加 `semantic_key`
只有调试价值，不值得动生产 collection（§4.3）。

## 4. 问题三：迁移

### 4.1 无 Alembic 现状下的最小迁移路径

新增幂等迁移辅助（`meno migrate` CLI 或 `build_service` 启动时调用，二选一，倾向 CLI
显式执行），按方言探测后执行：

1. **加列**（两方言都支持 `ALTER TABLE ... ADD COLUMN`；存在性探测：
   SQLite 用 `PRAGMA table_info(meno_claims)`，PG 用 `information_schema.columns`）：
   - `semantic_key VARCHAR(71) NOT NULL DEFAULT ''`
   - `superseded_by_id VARCHAR(64) NULL REFERENCES meno_claims(id)`
   - `superseded_reason VARCHAR(32) NULL`
2. **回填** `semantic_key`：按 §1 公式对存量 claim 逐行计算（Python 侧，分批）。
3. **存量去重**：对 `(user_id, semantic_key)` 分组发现多条 active（历史重复确实存在，
   §0.8）→ 保留 `valid_from` 最新者，其余按 `version_upgrade` 规则 supersede
   （含 Qdrant 删点）。**必须先于索引创建**。
4. **建部分唯一索引** `uq_claim_active_semantic`（`CREATE UNIQUE INDEX IF NOT EXISTS`，
   两方言同语法）。ORM 元数据里用 `Index(..., sqlite_where=..., postgresql_where=...)`
   声明，保证新库 `create_all` 与迁移后旧库结构一致。

每步幂等、可重入；任一步失败可重跑。

### 4.2 SQLite / Postgres 差异点

- SQLite 的 `ADD COLUMN` 不能加非常量默认、不能改 nullability —— 上述三条均满足限制。
- SQLite 支持 partial index（≥3.8，2013 年起），语法与 PG 一致。
- review #B-9：SQLite 下 `with_for_update(skip_locked)` 被静默丢弃，dev 多 worker 会
  双消费 —— 部分唯一索引正是这种情况的 DB 层兜底（见 §6）。

### 4.3 生产（付费）Qdrant collection

- **零迁移**：payload schema 不变、维度不变、collection 名不变。
- 重放期间的在线影响仅是：新 claim 插点（需重新 embedding，成本 ∝ 新 claim 数，
  走现有 `vector_upsert_batch_size` 分批与退避/熔断）+ 被 supersede 旧点删除。
- 修复路径：`rebuild_projection()` 只投影 active 非敏感 claim，协调完成后跑一次即可
  让 collection 与 Postgres 完全收敛；验证查询 = DB 侧
  `GROUP BY user_id, semantic_key HAVING COUNT(*) FILTER (status='active') > 1`
  应为空 + Qdrant 点集 ⊆ active claim 集。

## 5. 问题四：幂等

### 5.1 重放幂等（逐层）

| 层 | 机制 |
|---|---|
| reprocess 入队 | `(event_id, processor_version)` 唯一 + `ON CONFLICT DO NOTHING` |
| claim 创建 | `derivation_key` 唯一（含版本），重复处理跳过（现状保留） |
| 同键同版本 | 不新建，仅 `ClaimEvidence` upsert（`uq_claim_event` 防重） |
| supersede | UPDATE 带 `WHERE status='active'` 守卫，已失效者不再动；新 claim id 是 derivation_key 的确定性 uuid5，重跑收敛到同一状态 |
| Qdrant | upsert/delete 按点 id 幂等；残留点被 `_admit` canonical 过滤，rebuild 可修复 |
| 部分唯一索引 | 并发/重试下最后的硬保证；冲突 → 事务回滚 → Outbox 保持 pending → 重试时 derivation_key 已存在或走 dedupe 分支，收敛 |

### 5.2 可重复实验（dataset hash / retrieval cache）

- dataset hash 机制不变（报告已含 questions/contexts sha256）。
- **报告绑定版本**（小改，实现属任务 6 阶段）：`/v1/health` 响应补
  `extractor_version` 字段；benchmark 报告 `config` 段记录
  `extractor_version` + `embedding_projection_version` + `policy_version`。
  没有这项，两份报告无法证明各自跑在哪个抽取器上。
- **retrieval cache 防串版**：cache payload 增加 `extractor_version` 与两个 dataset
  hash；`_load_retrieval_cache` 在 run_id 匹配之外再校验这三项，不符即弃用重跑。
  现状只按 run_id + question 序列校验，同一 run_id 换 extractor 复用 cache 会
  直接污染对照实验——这是必须堵的口子。
- 判定式可重复性：同一 extractor_version + 同一 dataset hash + 新 run_id ⇒
  event_id（uuid5 含 run_id）确定 ⇒ derivation_key 确定 ⇒ claim 集确定。

## 6. 问题五：对比实验策略

**原则：A/B 按用户命名空间隔离，绝不在同一 user 内并行两个版本。**

1. **跨 run 隔离（现有机制，直接沿用）**：`user_id`/`event_id` 嵌入 `run_id`，
   版本 A（`MENO_EXTRACTOR_VERSION=v1, run_id=expA`）与版本 B（`v2, expB`）在
   同一 DB、同一 Qdrant collection 里也是完全 disjoint 的用户、事件、claim、
   Outbox 行；向量检索按 `user_id` 过滤，互不可见。
2. **流程**：冻结 dataset（记录 hash）→ 跑 A → 跑 B（同机顺序即可）→
   对比两份报告（各自带 extractor_version 与相同 dataset hash）→ `--cleanup`
   按 user 删除。铁律：**run_id 永不跨 extractor_version 复用**（§5.2 的 cache
   校验做硬保证）。
3. **为什么不支持同 user 内 A/B**：本协议的不变式就是"每 (user, 语义) 一条 active"，
   同 user 双版本必然互相 supersede——这正是要消灭的污染，不是功能。
4. **原地升级（生产迁移故事，与 A/B 区分）**：单 user 重放旧事件 → 键级协调 +
   事件级清理把 active 集整体切到新版 → 验证 §4.3 的两条查询。先在生产 DB 快照
   副本上演练，再上生产。
5. 对照期间检索参数（max_facets、min_confidence、purpose）与 embedding 配置必须
   完全一致，只变 extractor_version；结果差异才可归因。

## 7. 并发与失败语义

- **同 user 并发**：两个 worker 同时处理同 user 的不同事件、产出同键 candidate →
  部分唯一索引保证只有一个事务成功；失败方回滚 → Outbox pending → 重试走
  dedupe/幂等分支收敛。PG 下另可用 `pg_advisory_xact_lock(hashtext(user_id))`
  降低冲突率（性能优化，非正确性依赖；SQLite 单写者天然串行）。
- **崩溃点分析**：Qdrant 操作在 DB commit 前（现状顺序）。commit 失败 → 向量里
  多了无 DB 对应的点 → retrieve 回查 DB 无此 claim → 跳过（§0.5），无害；
  commit 成功但进程随即死亡 → 旧点已删、新点已插（都在 commit 前完成），一致。
- **Outbox 失败语义不变**：协调逻辑抛错走现有退避/DLQ（20 次进 failed，
  `requeue-failed` 可救）。

## 8. 对现有代码的约束 / 风险清单（实现时必须处理）

1. **R1（必须修）**：service.py:313 claim 戳用 `settings.extractor_version` 而非
   `row.processor_version` —— 版本切换窗口 lineage 误标。本协议 §2.2 的
   "worker 只处理同版本行 + 盖行版本戳"一并解决。
2. **R2（行为变更，需批准）**：同 user 同内容重复事件从"多条 active"变为
   "一条 claim 多 evidence"。现有测试若断言 claim 计数会受影响（实现阶段先跑
   全量 pytest 定位）。
3. **R3**：feedback `correct` 本身不幂等（review #B-8：replacement 用 uuid4，
   不校验原 claim 是否已 superseded）。本协议只补字段对齐（§3.3），**不**顺手修
   B-8；但部分唯一索引上线后，重复 correct 若产生同键双 active 会撞索引报错 ——
   实现阶段需验证该路径并决定是否顺带修（倾向顺带修，理由：索引会把潜在脏数据
   变成显式 500）。
4. **R4**：`semantic_key` 用 `NOT NULL DEFAULT ''` 过渡，空键不参与唯一约束语义
   （回填保证无空键；代码路径断言非空）。若 Trism 更倾向 NULL 方案也可，但
   PG/SQLite 唯一索引都视 NULL 为互不相同，等于放弃空键行的硬保证——不推荐。
5. **R5**：任务 6 若引入多 evidence claim，事件级清理规则要扩展（现状每 claim
   恒 1 条 evidence，规则只覆盖此情形）；本文 §2.2 的规则以"claim 含被重放事件
   的 evidence"为判定，多证据情形建议届时细化为"全部证据事件均被重放才收回"。
6. **R6**：重放全量会重新 embedding 全部新 claim（点 id 全新），付费 provider
   有一次性成本；靠现有分批/退避/熔断控制节奏，必要时按 `--user-id` 分批重放。
7. **R7**：`audit_claim` 读模型建议补 `semantic_key/superseded_by_id/superseded_reason`
   展示（只读增强，非必须，随任务 6 一并）。

## 9. 待 Trism 决策点

- **D1**：新增三列 + 部分唯一索引（§3.1）——数据模型变更，需批准。
- **D2**：worker 改为只处理 `processor_version == settings.extractor_version` 的行 +
  reprocess 时 cancel 旧版本 pending 行（§2.1/§2.2）——行为变更，需批准。
- **D3**：explicit_feedback claim 免于重放协调、同键时跳过机器 claim（§1/§3.2.4）
  ——语义策略，需批准。
- **D4**：`/v1/health` 增加 `extractor_version` 输出（§5.2）——公开接口小改，需批准。
- **D5**：R3 是否顺带修 feedback correct 幂等（否则索引可能把脏路径变成 500）。

## 10. 验收对照（生产复测检核项"版本协调后无重复 active claim"）

1. `meno migrate` 在 SQLite dev 库与 Postgres 生产库均幂等通过。
2. 重放全量事件后：DB 无 `(user_id, semantic_key)` 多 active；Qdrant 点集 ⊆ active。
3. 同版本重复重放：claim 集、revision 之外的状态零变化（幂等）。
4. A/B 两 run 报告：dataset hash 相同、extractor_version 不同、无共享 user_id；
   cache 跨版本复用被拒绝。
5. baseline pytest 全绿 + 新增协调协议测试（同键替代、withdrawn、dedupe、
   feedback 保护、并发冲突收敛）。
