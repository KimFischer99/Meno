# Meno v1.0

Meno is an evidence-backed personal agent memory sidecar. PostgreSQL is the canonical
store; Qdrant is a rebuildable derived index. Every injected facet is gated by user,
purpose, status, confidence, sensitivity, evidence, and consent state.

The architecture and acceptance criteria are in [Meno_SPEC.md](Meno_SPEC.md).
The verified VPS deployment, benchmark results, graph decision, and production
Go/No-Go assessment are in [Meno_v1.0_Test_Report.md](Meno_v1.0_Test_Report.md).

## Development and tests

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[test]'
.venv/bin/pytest
```

Production Meno supports Google cloud embeddings only. Unit tests inject a deterministic
test double and never call Google. There is no deployable FastEmbed,
SentenceTransformer, Qwen, BGE, or hashing backend.

## Production configuration

```bash
export MENO_ENV=production
export MENO_DATABASE_URL='postgresql+psycopg://meno:...@127.0.0.1:5432/meno'
export MENO_VECTOR_MODE=qdrant
export MENO_QDRANT_URL=http://127.0.0.1:6333
export MENO_QDRANT_COLLECTION=meno_claims_gemini_768_v1
export MENO_EMBEDDING_PROVIDER=google
export MENO_EMBEDDING_MODEL=gemini-embedding-001
export MENO_EMBEDDING_DIMENSION=768
export MENO_EMBEDDING_PROJECTION_VERSION=gemini-embedding-001-768-v1
export MENO_GOOGLE_CREDENTIALS_FILE=/run/secrets/meno-google-embedding
export MENO_GOOGLE_BATCH_SIZE=32
export MENO_VECTOR_UPSERT_BATCH_SIZE=128
export MENO_API_TOKEN='a-random-secret-with-at-least-32-characters'
meno serve
```

The credentials file accepts `Model: ...` and `Key: ...` lines and must be mode `0600`
in production. Do not commit it, `.env`, database passwords, provider authentication,
or raw benchmark conversations. Bind PostgreSQL, Qdrant, and Meno to loopback unless a
separate authenticated network boundary is in place.

Google document embeddings are generated asynchronously in batches. Sensitive claims
are not sent to the embedding provider. A new model, dimension, or provider requires a
new collection and a canonical replay:

```bash
meno rebuild-projection --batch-size 32
```

## API contract

- `POST /v1/ingest` stores an event and transactional outbox record.
- `POST /v1/ingest/batch` durably accepts up to 256 stable-ID events.
- `POST /v1/retrieve` returns evidence-backed, purpose-gated facets.
- `POST /v1/feedback` confirms, rejects, or supersedes a claim.
- `POST /v1/predict` ranks candidates but never authorizes an action.
- `POST /v1/consents` grants or revokes source/purpose access.
- `POST /v1/deletions` removes canonical and derived user state.
- `GET /v1/audit/{claim_id}` explains lineage.
- `GET /v1/revisions/{user_id}` reports current state revision.
- `GET /v1/users/{user_id}/drain` reports pending/failed outbox counts for drain detection.

All mutating requests require `Idempotency-Key`.
