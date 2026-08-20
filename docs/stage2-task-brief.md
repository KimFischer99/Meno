# 阶段二 Task Brief —— claim 版本协调 + extractor 重写 + evidence reinforcement

> 派给 Kimi（opencode 主线程），Trism 为 leader。按 AGENTS.md 执行：最小正确路径、外科式改动、
> 不留兼容层、每步验证、只对 Trism 汇报。本地 commit 到 `feat/personamem-e2e`，**禁止 push**。

## 背景（已完成的阶段一）
- 分支 `feat/personamem-e2e`，已含两个本地 commit：
  - `93f7c60` 阶段一（Outbox 退避/分批/熔断/DLQ、drain 端点、audit 异步化、temporal 下推）
  - `55378bf` SiliconFlow BAAI/bge-m3 OpenAI 兼容 embedder（provider=google|siliconflow）
- 小机实例已用硅基跑通全链路，baseline 40 个 pytest 全绿，ruff 全过。
- 现有 extractor 是**占位符**（正则单模式、每事件最多 1 个 candidate、未命中整条原文存 episodic、
  无跨事件合并/矛盾检测、无自动 supersede）。

## 阶段二范围（按序执行，第 5 项是 GATE）

### 5. claim 版本协调协议（GATE，先交【设计文档】）
**这是硬门禁：先输出 `docs/claim-version-coordination.md` 设计文档，停下手等 Trism/Anoki 审核，通过后才允许做 6/7。**

背景：新 `extractor_version` 重放旧事件会生成平行 claim 集、旧 claims 仍 active，无跨版本 supersede/清理协议。
不解决就开工，第一个抽取器对比试验会被重复 claim 污染，数据不可信。

设计文档必须回答：
1. 重放语义：extractor_version bump 后旧事件重放，新产生的 claim 与原 claim 的关系；如何保证每个 (user, 语义) 只有一条 active claim。
2. supersede/清理：用什么字段/状态标记旧 claim 失效（如 claim 上 `version`、`superseded_by`、state 变化），跨版本如何 supersede。
3. 迁移：DB（Postgres/SQLite 兼容路径）与 Qdrant payload 各需要什么变化；付费/生产的 collection 怎么办。
4. 幂等：重放与可重复实验（dataset hash、retrieval cache）在版本协调下如何保持。
5. 对比实验策略：extractor 新旧版本对照时，怎么隔离才能不被重复 claim 污染。

验收：该文档能被 review 直接批准着手 6/7。文档写好后**本地 commit 并停下汇报**，不要继续实现。

### 6. extractor 重写（GATE 通过后）
现在 extractor 是正则占位符，是 track_evolution / suggest_new_ideas 最弱的根因。重写要建立"演化"语义：
- 跨事件合并同一偏好的演化、压缩高度重叠 claim、识别矛盾并自动 supersede。
- 未命中时如何优雅降级（别再一股脑把整条原文存成 episodic）。
- dependency：新增逻辑主要用标准库/现有依赖；确需新依赖要说明理由，不允许随便加。
- 每个新行为配测试/评估。

### 7. half-life 与 evidence reinforcement 挂钩
- `_effective_confidence` 现在只看 `valid_from` 年龄，同一偏好反复确认也衰减——要让"证据越多越稳"生效，
  同一偏好确认 N 次提升置信度，而不是纯按时间衰减。
- `_score` 的 evidence 项是死代码常数 0.6（每 claim 恒 1 条 evidence）——令其真正反映证据数量。
- 补测试覆盖：多证据提升、单证据衰减、混合。

### 生产复测
extractor 升级 + 版本协调落定后，在小机（硅基 embedding，1024 维）跑一次全量端到端复测，检核：
retrieval p95 回落到 900ms 内、drain 收敛、temporal 过滤仍正确、audit 证据链完整、版本协调后无重复 active claim。

## 硬边界（严格遵守）
- 一次只做一个子项；第 5 项没审过，绝不动 6/7 的实现代码。
- 不改公开接口 / 数据模型 / 依赖 / 生产配置除非任务明确要求；改了要先说明。
- git 本地 commit，不 push；不碰 `/root/.hermes`、记忆、session DB。
- 不新增第三方依赖除非说明理由。

## 汇报格式（每阶段结束时给 Trism）
- 做了什么 / 改了哪些文件 / 测试结果（命令+输出摘要）/ 需 Trism 决策的点 / 下一步。
- 第 5 项：只交设计文档路径 + 关键决策摘要，**明确停在等审**。
