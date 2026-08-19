# Meno v1.0 实现、部署与评估报告

测试窗口：2026-08-18 至 2026-08-19 UTC
评估对象：本目录代码、独立 VPS 实验栈、Hermes Agent 独立 Profile
结论版本：final

## 1. 结论

### 1.1 发布判断

| 判断 | 结论 |
|---|---|
| 直接用于真实生产 | **No-Go** |
| 受控、可回滚的 shadow / opt-in pilot | **有条件 Go** |
| 以稳定版 `v1.0.0` 对外宣称生产就绪 | **No-Go** |
| 以 `alpha` / `preview` 形式开源 | **Go** |
| v1 现在加入独立 graph database | **不需要** |

Meno 已经具备可运行的 evidence-backed memory sidecar、Hermes 自动注入、用户与 purpose 隔离、consent、修订、删除、审计哈希链、PostgreSQL canonical store、Qdrant derived index，以及 Sidecar/Qdrant 故障降级路径。生产 embedding 已于 2026-08-19 完整切换到 Google `gemini-embedding-001`；部署包不再包含 FastEmbed、ONNX Runtime、SentenceTransformer 或其他本地 embedding backend。真实 Hermes 跨 session 回忆、并发写入、负向隔离和故障注入均通过。

云端迁移解决了本地模型常驻和 ONNX 依赖问题，但没有消除生产门槛：Google 路径的 LongMemEval 10-item `Recall@10=0.65`、检索 `p95=1663.8 ms`（含一个网络尾延迟样本），仍缺少全量质量评测、provider quota/429 soak、真人 pairwise、正式 schema migration、备份恢复、WORM 封存、监控告警和稳定进程监管。因此当前更准确的定位仍是“可验证的开源 alpha”，而不是生产稳定版。

### 1.3 Google cloud embedding 迁移结论

- production provider 固定为 Google `gemini-embedding-001`，document/query 分别使用 `RETRIEVAL_DOCUMENT` / `RETRIEVAL_QUERY`；输出截断为 768 维并重新 L2 normalize。
- 历史 active claim 已重建到新 collection `meno_claims_gemini_768_v1`；旧 384 维 collection 与旧 release 暂留作可回滚备份，但旧进程已停止，不参与请求。
- VPS 当前环境确认 `fastembed=False`、`onnxruntime=False`；Meno 云端进程约 `132.6 MiB PSS / 0 MiB SwapPSS`。同一时刻 PostgreSQL 全部 15 个进程合计约 `30.2 MiB PSS / 4.1 MiB SwapPSS`，Qdrant 约 `33.1 MiB PSS / 10.9 MiB SwapPSS`。
- 在当前 842 MiB 实验机上仍有约 342 MiB available memory。由此看，先前失败不能简单归因于“两块数据库太重”；更主要的问题是本地 embedding runtime、极小主机上的整体叠加，以及 benchmark 吞吐/延迟。对 2 GiB 单机，小数据量 PostgreSQL + Qdrant + Meno 在内存上具备可行性，但仍需用目标数据量做 24 小时 soak，不能仅凭本次小样本直接宣称 production-ready。
- API key 只从 mode `600` 的 credentials file 加载，未写入源码、报告或 benchmark artifact。敏感 claim 默认不会发送给 Google。

### 1.2 Graph 决策

**当前版本不应新增 Neo4j、FalkorDB 或其他独立图数据库。**

理由：

1. `Meno_SPEC.md` 已明确规定 v1 用 PostgreSQL adjacency table 表达 community、claim relationship 和 `supersedes`，只有多跳 social graph 查询成为实测瓶颈时才拆出图数据库。
2. 当前实现已有 `meno_entities`、`meno_claim_entities`、`meno_claim_edges` 和 `Claim.supersedes_id`，数据模型已经保留了图投影入口；真正缺少的是实体抽取、关系生成、community 计算和多跳 retrieval 策略，而不是图数据库本身。
3. Hindsight 当前会并行执行 semantic、keyword、graph 和 temporal retrieval，并将 entity、relationship、time series 与向量组合；这证明 graph retrieval 有效，但不证明每个 v1 都需要独立 graph service。Hindsight 的实现与说明见 [Hindsight repository](https://github.com/vectorize-io/hindsight)。
4. 本次 VPS 只有 842 MiB RAM，现有 PostgreSQL、Qdrant、Meno 与 Hermes 已产生 swap；再增加一个数据库会扩大故障面和资源压力，却不能直接修复当前暴露的 temporal recall、preference evolution 和排序问题。

建议先实现 PostgreSQL 上的 typed edge 写入、entity normalization、`supersedes/contradicts/supports` 查询和可重放 graph projection。只有同时满足以下条件才立项外部 graph store：

- 产品确实需要 `>2` hop 的 entity/social reasoning；
- 代表性数据规模下 PostgreSQL recursive CTE / adjacency query 的 `p95` 持续违反 SLO；
- ablation 证明 graph retrieval 对目标任务有显著增益，而不是只增加召回噪声；
- 已具备 dual-write、rebuild、deletion propagation、consent filtering 和故障回退方案。

## 2. 实现范围

### 2.1 已实现

| 模块 | 状态 | 说明 |
|---|---:|---|
| Event / claim / evidence canonical model | 完成 | PostgreSQL 为真相源，claim 保留 evidence ID |
| Consent / purpose / sensitivity gate | 完成 | 默认拒绝敏感信息，支持 grant/revoke |
| User revision | 完成 | PostgreSQL 原子 UPSERT；并发 revision 无丢失 |
| Outbox | 完成 | ingest 与 outbox 同事务，异步 materialization |
| Vector index | 完成 | Qdrant 为可重建 derived index；故障时回退 PostgreSQL |
| Temporal validity / decay | 完成 | `valid_from`、`valid_to`、half-life、`as_of` |
| Correction / supersession | 完成 | feedback 可 confirm/reject/correct，保留 lineage |
| Deletion | 完成 | canonical 与 vector 在线传播，返回 receipt hash |
| Audit | 完成基础版 | append-only hash chain + PostgreSQL advisory lock |
| HTTP API / auth / idempotency | 完成 | production 强制 bearer token 与正确基础设施 |
| Hermes provider | 完成 | prefetch、post-turn sync、工具、SQLite durable spool、fail-open |
| Local/VPS benchmark harness | 完成 | LongMemEval、PersonaMem adaptation、negative、concurrency |
| License | 完成 | Apache-2.0 文本已随项目提供 |

### 2.2 未完成或仅有骨架

- `meno_entities` / `meno_claim_edges` 尚未进入完整 entity extraction、normalization 与 multi-hop retrieval 流程。
- SPEC 中的 preference probability distribution、calibration / ECE、community/social prior、personal residual 尚未落地。
- 没有 Alembic 或等价的 versioned migration、dual-read/write 和 rollback window。
- 没有 WORM/Object Lock audit root 封存、定期 Merkle root 或外部审计锚点。
- 没有 OpenTelemetry metrics/traces、SLO dashboard、告警、容量与 backpressure 管理。
- 没有自动 backup/restore 演练与 disaster recovery 验证。
- 没有真人 pairwise personalization utility 评测。
- PersonaMem 运行的是 retrieval-only option-ranking adaptation，不是官方 end-to-end LLM pipeline。

## 3. 部署环境

### 3.1 隔离策略

部署在一台隔离的低内存 VPS。没有修改现有 dirty Hermes checkout，而是使用独立 worktree、独立 virtualenv、独立 Hermes Profile 和独立 Meno 数据库/collection。没有执行 `sudo` 或系统级安装。公开报告不记录主机地址、登录用户或 SSH 凭据路径。

| 项目 | 实际状态 |
|---|---|
| OS / CPU | Ubuntu 24.04，2 vCPU |
| RAM / swap | 842 MiB RAM，1 GiB swap；Google 切换后约 342 MiB available、swap 使用约 120 MiB |
| Disk | 29 GiB，总使用约 6.3 GiB |
| Hermes | `v2026.8.18` / `0.20.4`，commit `e624e9f...`；测试时为官方 latest release |
| PostgreSQL | 18.1，由 user-space `pg0` 提供，database `meno_v1_final` |
| Qdrant | 1.19.0，active collection `meno_claims_gemini_768_v1`，768 dimensions |
| Embedding | Google `gemini-embedding-001` cloud API；本地 embedding runtime 已移除 |
| Meno | `1.0.0`，production validation 开启 |
| Binding | PostgreSQL `127.0.0.1:55432`、Qdrant `127.0.0.1:6333`、Meno `127.0.0.1:8765` |
| Secrets | 独立 env/token 文件，mode `600`；报告不记录密钥 |

Hermes Profile `meno-prod` 禁用了 Hermes 内建 memory 与 user-profile injection，仅启用 Meno provider；因此跨 session 命中可以归因于 Meno，而不是旧内建记忆。Hermes release 的版本证据见 [Hermes Agent v0.20.4 release](https://github.com/NousResearch/hermes-agent/releases/tag/v2026.8.18)。

### 3.2 当前运行状态与限制

最终检查时三个服务都只监听 loopback，Meno `/health/ready` 返回 database/vector `ok`。PostgreSQL 旧 PID 文件未保留，但实际 PostgreSQL 进程、监听端口和 Meno 数据库 probe 均正常。

由于目标用户没有启用 systemd user lingering，本次服务以隔离的 `nohup` + PID file 方式维持，**重启后不会自动恢复，也没有 restart policy**。这足以做独立实验，不满足 production supervision 要求。

## 4. 验证结果

### 4.1 自动化与一致性

| 检查 | 结果 |
|---|---:|
| 本地 pytest | `27 passed` |
| VPS pytest | `27 passed` |
| Ruff | 通过 |
| 并发 ingest | 16/16 accepted，8 workers，revision 32/32，16 facets，passed |
| Google 切换后审计链 | 759 rows；roots 1；dangling 0；forks 0；hash mismatch 0；tips 1 |
| 负向安全套件 | 6/6 passed |

pytest 仅有一条来自 FastAPI/Starlette 测试依赖的外部 deprecation warning，不影响测试结果；后续依赖升级时应消除。

并发结果：[vps-concurrency-final.json](artifacts/benchmarks/results/vps-concurrency-final.json)
审计结果：[vps-audit-chain-final.json](artifacts/benchmarks/results/vps-audit-chain-final.json)
负向结果：[vps-negative-final.json](artifacts/benchmarks/results/vps-negative-final.json)

Google 切换后的结果：[vps-concurrency-google.json](artifacts/benchmarks/results/vps-concurrency-google.json)、[vps-audit-chain-google.json](artifacts/benchmarks/results/vps-audit-chain-google.json)、[vps-negative-google.json](artifacts/benchmarks/results/vps-negative-google.json)

负向套件覆盖：

- cross-user leakage = 0；
- prompt/source poisoning 不进入可用 memory；
- assistant output 不作为 primary user evidence；
- sensitive memory 默认拒绝；
- consent revoke 后不可检索；
- deletion 后 canonical/online retrieval 不可见。

### 4.2 LongMemEval

数据源为官方 500-item oracle set；数据 SHA-256：`821a2034d219ab45846873dd14c14f12cfe7776e73527a483f9dac095d38620c`。官方项目说明该基准覆盖 information extraction、multi-session reasoning、knowledge updates、temporal reasoning 与 abstention，见 [LongMemEval repository](https://github.com/xiaowu0162/LongMemEval)。

| 环境 | 样本 | Embedder | Hit@10 | Recall@10 | p95 | 判断 |
|---|---:|---|---:|---:|---:|---|
| Local full | 500 | deterministic hashing（仅开发） | 0.9440 | **0.8645** | **97.8 ms** | Recall 与 latency 均未过 SPEC gate |
| VPS production sample | 10 | FastEmbed semantic | 0.9000 | 0.6500 | **152.5 ms** | 小样本、不可作为全量结论；latency 未过 gate |
| VPS cloud sample | 10 | Google `gemini-embedding-001` | 0.9000 | 0.6500 | **1663.8 ms** | 9/10 命中；小样本；出现一个 1.66 s 网络尾延迟 |

全量结果：[local-longmemeval-full-v3.json](artifacts/benchmarks/results/local-longmemeval-full-v3.json)
VPS 小样本：[vps-longmemeval-10-prod.json](artifacts/benchmarks/results/vps-longmemeval-10-prod.json)
Google VPS 小样本：[vps-longmemeval-10-google.json](artifacts/benchmarks/results/vps-longmemeval-10-google.json)

本地全量按类型的主要弱项：

- temporal reasoning：`Recall@10=0.6677`；
- knowledge update：`Recall@10=0.7500`；
- multi-session：`Recall@10=0.9772`；
- single-session preference/user：`Recall@10=1.0`。

VPS 全量 semantic run 预计需要较长时间，本轮只执行固定 10-item cloud sample；没有把小样本包装成完整结果。它只用于验证真实 Google/Qdrant 路径和粗略 latency，不能与全量 leaderboard 或本地 500-item run 等价比较。迁移后的并发 smoke 为 16/16 accepted、revision 32/32、16 facets，写入 ACK mean 72.0 ms/max 147.6 ms；独立 API smoke 的检索约 458 ms。

### 4.3 PersonaMem adaptation

使用官方 589 questions / 20 personas / 37 contexts。questions SHA-256：`cccd34cf53e0bc4d9536c04cff5ca045156d9a4e227e83327112482840bbc93c`；contexts SHA-256：`217247ebfec9e8442fc53570c795ab69f21aad08745f7de78d9beab51b122d4a`。官方项目见 [PersonaMem repository](https://github.com/bowen-upenn/PersonaMem)。

该适配器把候选项作为 retrieval-only ranking 任务，结果为：

- option accuracy：`0.3888`，随机四选一基线为 `0.25`；
- MRR：`0.6078`；
- mean latency：`56.3 ms`；
- degraded：`0`。

主要类别：

| 类型 | Accuracy |
|---|---:|
| generalizing to new scenarios | 0.7895 |
| preference-aligned recommendations | 0.4545 |
| recalling facts mentioned by user | 0.7647 |
| reasons behind updates | 0.7778 |
| track full preference evolution | **0.2734** |
| suggest new ideas | **0.0323** |

结果：[local-personamem-full.json](artifacts/benchmarks/results/local-personamem-full.json)

`correct_support@k=1.0` 在此 adaptation 中过于宽松，不应作为质量证明。该结果也不能和 PersonaMem 官方 end-to-end LLM accuracy 或 Hindsight 公布分数直接比较。它仍然揭示了 Meno 当前是“可靠召回证据”导向，而不是已经完成的动态个性化推断系统。

### 4.4 Hermes 真实模型与跨 session

使用 VPS 中已有的真实 Nous provider credentials；Google embedding key 与推理 provider key 均没有写入报告。

测试流程：

1. 新 session 告诉 Hermes：“部署评审先给 Go 或 No-Go，再列最多两项阻塞原因”。
2. Meno materialize 后，在完全新建 Hermes session 中询问该偏好，并明确要求不要调用工具。
3. Hermes 回答：“你希望部署评审先给出 Go 或 No-Go 结论，再附上最多两项阻塞原因。”

由于独立 Profile 已关闭 Hermes 内建 memory/user profile，且第二轮禁止工具调用，该结果验证了 Meno prefetch 的跨 session 自动注入路径。它是一个真实集成 smoke test，不是统计意义上的效果评测。

Google 切换后以同一独立 Profile 再次执行无工具 one-shot，仍准确回答上述偏好，确认 Hermes prefetch 没有因 projection 切换而退化。

观察到 Hermes quiet CLI 仍会输出较多内部 reasoning 文本；这是当前 Hermes CLI 行为，不是 Meno 数据泄漏证据，但生产 channel 应另做输出清理与隐私复核。

### 4.5 故障注入

| 注入 | 结果 |
|---|---|
| 停止 Qdrant | readiness 返回 503；retrieve 返回 `degraded=true`，从 PostgreSQL 提供 1 个 facet；恢复后 readiness `ok` |
| 停止 Meno Sidecar | Hermes 正常回答 `2+2=4`，退出码 0；2 个待同步事件进入 SQLite spool |
| 重启 Meno | spool 自动 drain 至 0，事件最终进入 Meno |

这验证了“memory 失效不能让 Hermes 失效”的核心 fail-open 目标，同时证明 post-turn 数据不会在短暂 Sidecar outage 中静默丢失。

## 5. 实验中发现并修复的问题

1. **Proxy 污染 loopback HTTP：** `httpx` 会继承环境 proxy，导致本地 Sidecar 访问异常。Hermes provider 改为 loopback client `trust_env=False`。
2. **Temporal benchmark 泄漏未来状态：** retrieval 未正确应用 `as_of`。已加入 `as_of` gate，避免未来 claim 进入历史问题。
3. **并发 revision 丢更新：** read-modify-write 在并发事务中产生 lost update。PostgreSQL 改为原子 UPSERT increment，并加入 24-thread regression test。
4. **审计链分叉：** 多事务同时读取同一个 tail；先加入 PostgreSQL advisory transaction lock，随后发现 `autoflush=False` 导致同事务多次 audit 仍可读到旧 tail，最终在每次 append 后显式 `flush()`。最终数据库 84 条审计记录零分叉、零 hash mismatch。
5. **模块级 FastAPI app 泄漏连接：** import 时创建全局 DB engine。改为 Uvicorn factory lifecycle。
6. **远端 editable install 指向旧复制源码：** `uv` 环境中的 editable wheel 实际使用 site-packages copy。已用 `--reinstall-package meno-memory` 明确刷新，并验证加载路径来自最终 release source。

## 6. 对照 SPEC 的 Go/No-Go

| SPEC gate | 结果 | 状态 |
|---|---:|---:|
| 每条注入信息有 evidence | API/集成路径均返回 evidence IDs | 通过 |
| 显式纠正替换旧 belief | 单元/API 路径通过，保留 supersedes lineage | 通过（功能级） |
| Sidecar 关闭不破坏 Hermes | Hermes outage test exit 0 | 通过 |
| 跨用户与 consent bypass 为零 | deterministic negative suite 6/6 | 通过（覆盖仍有限） |
| 删除传播到 canonical/vector/cache | negative suite 通过 | 通过（未做 backup purge） |
| Recall@10 ≥ 0.90 | 0.8645 | **失败** |
| Cloud retrieval p95 < 900 ms、total p95 < 1000 ms | Google VPS 10-item p95 1663.8 ms | **失败（小样本，需扩大验证）** |
| 真人 pairwise personalized > generic | 未执行 | **缺失** |
| Production migration / rollback | 未实现 | **缺失** |
| WORM audit archive | 未实现 | **缺失** |
| 监督、自动恢复、监控、备份 | 未实现 | **缺失** |

因此总判定必须是 **No-Go**，不能用部分通过项覆盖硬门槛失败。

## 7. 上生产前的最小整改清单

### P0：阻止生产发布

1. 修复 temporal/knowledge-update retrieval，使官方固定数据集全量 `Recall@10 ≥ 0.90`，并对每次变更保存 dataset hash、配置和结果。
2. 对 Google API 做 provider latency、429/quota、timeout 和重试 soak；目标保持 cloud retrieval `p95 < 900 ms`、端到端 `p95 < 1000 ms`。2 GiB 可作为小规模候选配置，但必须按目标 claim 数量和并发实测，不再把 4 GiB 当作未经测量的硬下限。
3. 引入 Alembic 等 versioned migration，完成 backup/restore、replay/rebuild 和 rollback 演练。
4. 使用 systemd/container/orchestrator 管理 Meno、PostgreSQL、Qdrant；增加 readiness、restart policy、resource limits 与 alerting。
5. 增加 durable audit root 封存/WORM、密钥轮换和最小权限部署；验证删除在 backup retention 生命周期内的处理。
6. 执行多人、长时间、多用户的真实 Hermes shadow traffic，覆盖 provider timeout、数据库故障、重启、磁盘满、backpressure 与高并发。
7. 完成真人 pairwise evaluation，证明 personalized response 相比 generic response 有稳定正收益。

### P1：形成产品差异

1. 实现 preference distribution、校准与时间演化，优先解决 PersonaMem `track evolution`。
2. 实现 entity normalization 与 PostgreSQL typed-edge retrieval；先做 graph ablation，再决定是否外置 graph store。
3. 完成 community/social prior 的 consent、安全、curation 和删除模型。
4. 增加 end-to-end LongMemEval/PersonaMem LLM answer evaluation，区分 retrieval quality、answer quality、latency 和 token cost。

## 8. 开源价值判断

Meno **有开源价值**，但价值不在“又一个向量记忆库”，而在以下组合：

- canonical/derived 分离，Qdrant 可重建；
- 每个 facet 带 evidence、revision、policy decision 与 audit lineage；
- consent、sensitivity、correction、deletion 是核心模型，不是外围补丁；
- Hermes provider 可自动 prefetch/post-turn sync，并能在 outage 时 fail-open + durable replay；
- graph 能力按测量结果渐进引入，而不是把独立图数据库变成 v1 运维税。

建议首个公开标签使用 `v1.0.0-alpha`、`0.1.x` 或 `preview`，README 明确列出已验证范围与缺失项，并附本报告中的可复核 JSON。达到 Recall、latency、真人 utility、migration/restore、supervision 和 audit archive 门槛后，再发布稳定 `v1.0.0`。

## 9. 复核命令

以下命令不包含真实 token、数据库密码或 SSH key：

```bash
.venv/bin/ruff check .
.venv/bin/pytest
python3 -m json.tool artifacts/benchmarks/results/local-longmemeval-full-v3.json >/dev/null
python3 -m json.tool artifacts/benchmarks/results/local-personamem-full.json >/dev/null
python3 -m json.tool artifacts/benchmarks/results/vps-negative-final.json >/dev/null
python3 -m json.tool artifacts/benchmarks/results/vps-concurrency-final.json >/dev/null
python3 -m json.tool artifacts/benchmarks/results/vps-audit-chain-final.json >/dev/null
python3 -m json.tool artifacts/benchmarks/results/vps-longmemeval-10-google.json >/dev/null
python3 -m json.tool artifacts/benchmarks/results/vps-negative-google.json >/dev/null
python3 -m json.tool artifacts/benchmarks/results/vps-concurrency-google.json >/dev/null
python3 -m json.tool artifacts/benchmarks/results/vps-audit-chain-google.json >/dev/null
rg -n "Mono|mono_(events|claims|audit|feedback|hermes)" .
```

生产凭据应只通过受限 env/secret store 注入，不进入命令历史、测试 artifact 或版本库。
