# Meno v2.0

Meno is an evidence-backed personal agent memory sidecar. A relational database is
the canonical store and the vector index is a rebuildable derived view. Every
injected facet is gated by user, purpose, status, confidence, sensitivity,
evidence, and consent state.

What real data is still needed to close the remaining acceptance gates — and why
two of them cannot be closed with historical logs — is in
[MENO_DATA_REQUIREMENTS.md](MENO_DATA_REQUIREMENTS.md).
Deployment configuration for a single small host is in [deploy/](deploy/).
The evaluation lanes, what each one measures, and the rules they follow are in
[benchmarks/README.md](benchmarks/README.md).
Every capability beyond the v1 core is behind a default-off feature flag.

References to `Meno_SPEC.md` and `Meno_v2.0_Gate_Charter.md` in comments and docs
point at the internal design spec and its per-gate status record. Those are not
published; the acceptance criteria they encode are summarized below and the lanes
that measure them are in this repository.

**Scope of the v2.0 goal.** v2.0 targets an auditable *architecture candidate*:
every component-level correctness and safety gate passing. Currently 12 of 17 are
measured and passing — cross-user leakage, sensitive-inference denial, consent
revocation, deletion propagation, evidence lineage and audit-chain integrity,
explicit-correction propagation, retrieval latency, grounding attribution,
stale-active rate, sidecar-outage isolation, and full-state replay determinism.

`production GO` is a separate, stricter claim. Three gates — calibration against
real feedback, human pairwise preference, and social-prior residual uplift —
cannot be established by any offline lane, and two more need a corpus whose
queries name what they ask about, which no public memory benchmark supplies.
Passing the component gates does not imply production readiness, and PersonaMem
scores gate neither one (see [benchmarks/README.md](benchmarks/README.md)).

## Development and tests

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[test]'
.venv/bin/pytest
```

Production Meno v2.0 supports SiliconFlow `BAAI/bge-m3` embeddings only. Unit tests inject
a deterministic test double and never call SiliconFlow. There is no deployable local
FastEmbed, SentenceTransformer, Qwen, or hashing backend; `BAAI/bge-m3` is consumed only
through the pinned SiliconFlow provider.

## Production configuration

Two supported profiles. Both require `MENO_API_TOKEN` (32+ characters) and a
SiliconFlow embedding key; `Settings.validate()` refuses to start without them,
because the API middleware skips authentication entirely when the token is empty.

### Single small host — SQLite, no containers

For a 2 GB VPS running Meno beside Hermes. Canonical state in one SQLite file,
vectors in the Meno process at ~4.1 KB each (1024 dims, packed float32), bounded by
`MENO_VECTOR_MAX_RESIDENT`. Memory floor is ~130 MB versus ~400–600 MB for the
container profile.

Full configuration, systemd unit, and Hermes wiring: **[deploy/](deploy/)**.

```bash
export MENO_ENV=production
export MENO_DATABASE_URL='sqlite:////var/lib/meno/meno.db'   # four slashes = absolute
export MENO_VECTOR_MODE=memory
export MENO_VECTOR_MAX_RESIDENT=20000
export MENO_API_TOKEN='a-random-secret-with-at-least-32-characters'
export MENO_OPENAI_API_KEY='replace-with-siliconflow-api-key'
```

The SQLite path must be absolute and outside `/tmp`, `/var/tmp`, and `/dev/shm`:
canonical state has to survive a reboot to stay replayable (SPEC :1393). SQLite as
canonical does not breach the SPEC No-Go — that forbids the *vector store* becoming
the source of truth, and the #12 replay lane already runs on SQLite.

### Containers — PostgreSQL and Qdrant

For multiple processes or larger volumes. `docker-compose.yml` at the repository
root brings up both services.

```bash
export MENO_ENV=production
export MENO_DATABASE_URL='postgresql+psycopg://meno:...@127.0.0.1:5432/meno'
export MENO_VECTOR_MODE=qdrant
export MENO_QDRANT_URL=http://127.0.0.1:6333
export MENO_QDRANT_COLLECTION=meno_claims_bge_m3_1024_v1
export MENO_EMBEDDING_PROVIDER=siliconflow
export MENO_EMBEDDING_MODEL=BAAI/bge-m3
export MENO_EMBEDDING_DIMENSION=1024
export MENO_EMBEDDING_PROJECTION_VERSION=siliconflow-bge-m3-1024-v1
export MENO_OPENAI_API_KEY='replace-with-siliconflow-api-key'
export MENO_OPENAI_BASE_URL=https://api.siliconflow.cn/v1
export MENO_SEMANTIC_ROUTING_ENABLED=false
export MENO_CONTEXT_ACTIVATION_ENABLED=false
export MENO_USER_TOKEN_MATERIALIZATION_ENABLED=false
export MENO_PREFERENCE_DISTRIBUTION_ENABLED=false
export MENO_PREFERENCE_DISTRIBUTION_V2_ENABLED=false
export MENO_CLARIFICATION_OPPORTUNITIES_ENABLED=false
export MENO_SEMANTIC_ROUTER_CONFIG_FILE=benchmarks/fixtures/semantic-router-phaseb-frozen-config.json
export MENO_SEMANTIC_ROUTER_CONFIG_SHA256=c67d3d0b1b279116910219d53be1b99ed053397eb8e739b56a9912711f9e73c9
export MENO_PREFERENCE_HISTORY_RETRIEVAL_ENABLED=false
export MENO_PREFERENCE_HISTORY_MAX_FACETS=2
export MENO_PREFERENCE_HISTORY_MAX_EVENTS=4
export MENO_VECTOR_UPSERT_BATCH_SIZE=128
export MENO_API_TOKEN='a-random-secret-with-at-least-32-characters'
meno migrate
meno serve
```

Do not commit `.env`, database passwords, provider authentication,
or raw benchmark conversations. Bind PostgreSQL, Qdrant, and Meno to loopback unless a
separate authenticated network boundary is in place.

SiliconFlow document embeddings are generated asynchronously in batches. Sensitive claims
are not sent to the embedding provider. A new model, dimension, or provider requires a
new collection and a canonical replay:

```bash
meno rebuild-projection --batch-size 32
```

After migration and a successful default-off staging check, enable the frozen Phase B router
with `MENO_SEMANTIC_ROUTING_ENABLED=true`. Provider failures fail closed to the existing
canonical value key and are recorded in the audit chain.

Phase B2 preference-history retrieval is independently default-off. When enabled after
semantic routing, Meno keeps Qdrant active-only and enriches at most two retrieved current
preference facets from their PostgreSQL `supersedes` chains. Historical values are clearly
marked in `rendered_context` and pass the same user, purpose, consent, sensitivity, role,
confidence, and evidence gates before use.

Clarification opportunities are also default-off and require context activation,
materialized user tokens, and preference distributions. When enabled, an ambiguous
preference is exposed in `clarification_opportunities` and withheld from retrieval facets
until explicit feedback confirms it. Meno does not ask or act automatically.

Experimental preference distribution v2 is independently default-off. It preserves the
categorical `mode_probability`, adds a unit unknown prior and an evidence-backed Beta
belief, and exposes their product as `actionable_probability`. The strategy remains
`uncalibrated`; the extra probability must not be used as a production authorization gate.

## API contract

- `POST /v1/ingest` stores an event and transactional outbox record.
- `POST /v1/ingest/batch` durably accepts up to 256 stable-ID events.
- `POST /v1/retrieve` returns evidence-backed, purpose-gated facets and optional
  clarification opportunities.
- `POST /v1/feedback` confirms, rejects, or supersedes a claim.
- `POST /v1/predict` ranks candidates but never authorizes an action.
- `POST /v1/consents` grants or revokes source/purpose access.
- `POST /v1/deletions` removes canonical and derived user state.
- `GET /v1/audit/{claim_id}` explains lineage.
- `GET /v1/revisions/{user_id}` reports current state revision.
- `GET /v1/users/{user_id}/drain` reports pending/failed canonical and projection outbox counts for drain detection.

All mutating requests require `Idempotency-Key`.
