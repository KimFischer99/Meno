# Meno v3.0

Persistent, evidence-backed memory for **Hermes Agent**. Meno runs as a local
sidecar: SQLite stores canonical state, an in-process vector index serves recall,
and a durable Hermes spool retries turns when the sidecar is unavailable.

v3 brings the deployed runtime fixes back into the open-source repository.
It keeps the existing API and embedding stack rather than introducing another
memory architecture.

## Install with Hermes

Requires Python 3.11+, an installed Hermes Agent, and a SiliconFlow API key for
`BAAI/bge-m3` embeddings. Linux with systemd is the primary deployment target.
The embedding API is an external service and may incur charges.

```bash
git clone https://github.com/KimFischer99/Meno.git
cd Meno
# Use the Python interpreter from your Hermes installation:
/path/to/hermes/venv/bin/python deploy/install-hermes.py --start
```

The installer prompts for the embedding key, creates a private sidecar environment,
wires the Meno plugin into the active Hermes profile, and selects
`memory.provider: meno`. Existing profile files are backed up before editing.
Restart Hermes after installation so it loads the new provider.

See [Hermes installation and diagnostics](deploy/hermes-integration.md) for
custom profiles, runtime paths, existing installations, and foreground startup.
Only one external memory provider can be selected in Hermes at a time.

This is a **single-user sidecar**. Use a separate profile and Meno service/token
for each user; the shared bearer token is not a multi-tenant authorization system.

## What changed in v3

- Compact, lossless user-token snapshot deltas with periodic full snapshots.
- Stored idempotent results for feedback, consent, and deletion requests.
- Deletion watermarks prevent old queued events from restoring forgotten data.
- SQLite pool/cache limits and compact JSON serialization.
- Audit overflow persists to a local file and returns to the audit chain.
- Memory vectors rebuild from canonical state after service restart.
- A Hermes installer and diagnostics that check the actual provider loader.

The policy/extractor identifiers remain at their deployed versions: changing the
package version does not change extraction semantics or invalidate stored state.
Advanced routing remains opt-in. Both frozen routing fixtures are shipped in the
wheel; no production conversation, database, credential, or host identity is needed.

## Run the sidecar manually

For system-level installation and PostgreSQL/Qdrant, see [deployment](deploy/README.md).
The small-host profile uses SQLite and bounded in-process vectors.

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
# Copy deploy/.env.production.example to a private file outside the repository,
# fill in credentials, and create the configured durable database directory.
MENO_ENV_FILE=/path/to/private/meno.env deploy/start.sh --check
MENO_ENV_FILE=/path/to/private/meno.env deploy/start.sh
```

Keep Meno bound to loopback. Production configuration requires a random API token
of at least 32 characters and the embedding key. Non-sensitive claim text is sent
to the embedding provider; sensitive claims are excluded from embeddings.

After a restart the API can answer while the memory index warms in the background;
initial recall may be incomplete. Check the service log for warmup completion.
Canonical SQLite state remains the source of truth.

## Upgrade from v2

Stop the sidecar and back up the database and private environment file before
upgrading. Install v3, run `meno migrate` in the same configured environment,
then restart the sidecar and Hermes. Migration creates the new idempotency and
snapshot-delta tables; existing full snapshots remain readable. Keep the stable
user ID, API token, database path, embedding model, and dimension unchanged.

Do not run v2 against a database that has v3 delta snapshots; restore the
pre-upgrade backup if rolling back.

## API

All `/v1/*` requests use `Authorization: Bearer <MENO_API_TOKEN>`.
Mutating requests require `Idempotency-Key`.

| Endpoint | Purpose |
| --- | --- |
| `POST /v1/ingest`, `POST /v1/ingest/batch` | Durably accept events |
| `POST /v1/retrieve` | Retrieve purpose-gated, evidence-backed context |
| `POST /v1/feedback` | Confirm, reject, or correct a claim |
| `POST /v1/consents` | Grant or revoke source/purpose access |
| `POST /v1/deletions` | Delete canonical and derived state |
| `GET /v1/audit/{claim_id}` | Inspect lineage |
| `GET /v1/revisions/{user_id}` | Read state revision |
| `GET /v1/users/{user_id}/drain` | Check ingestion/projection queues |
| `GET /health/ready` | Check database/vector availability |

The running service exposes its full OpenAPI schema at `/docs`.

## Development

```bash
.venv/bin/pip install -e '.[test]'
.venv/bin/python -m meno.snapshot_delta
.venv/bin/python -m pytest tests/test_api.py tests/test_config.py tests/test_plugin.py tests/test_user_token.py
```

Tests use synthetic data and a deterministic embedding double. Historical
benchmark scripts remain available for research; large generated result caches
and raw corpora are local-only and are not included in v3 source distributions.
No new benchmark campaign is required to install or use Meno.

[Contributing](CONTRIBUTING.md) · [Security](SECURITY.md) · [Apache-2.0](LICENSE)
