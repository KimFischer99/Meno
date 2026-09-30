# Manual Meno deployment

For a Hermes profile on Linux, use the [installer](hermes-integration.md).
This document covers an independently managed system service.

## SQLite and in-process vectors

Requires Python 3.11+, git, Linux/systemd, and a SiliconFlow API key.
SQLite stores canonical state; vectors are rebuilt after restart.
The production template caps the index at 10,000 vectors (about 40 MB at 1024 dimensions),
in addition to the Python service's memory usage.

```bash
git clone https://github.com/KimFischer99/Meno.git /opt/meno
cd /opt/meno
python3 -m venv .venv
.venv/bin/pip install -e .

sudo useradd --system --home /var/lib/meno --shell /usr/sbin/nologin meno
sudo install -d -m 700 -o meno -g meno /var/lib/meno
sudo install -m 600 deploy/.env.production.example /etc/meno.env
sudoedit /etc/meno.env
```

Replace the API token placeholder with a random value of at least 32 characters
and fill in the SiliconFlow key. Keep the database path absolute and outside
temporary directories. The service stays on loopback.

To generate a token locally:

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

The service account needs read/execute access to the checkout and virtualenv.
Keep them outside a protected home directory, as in `/opt/meno`.

```bash
sudo env MENO_ENV_FILE=/etc/meno.env /opt/meno/deploy/start.sh --check
sudo sh -c 'set -a; . /etc/meno.env; set +a; /opt/meno/.venv/bin/meno migrate'
sudo cp deploy/meno.service /etc/systemd/system/meno.service
sudo systemctl daemon-reload
sudo systemctl enable --now meno
curl -fsS http://127.0.0.1:8765/health/ready
```

The supplied unit reads `/etc/meno.env` before switching to `User=meno`.
It restricts writes to `/var/lib/meno` and applies a 400 MB memory ceiling.
Adjust the vector cap and memory ceiling together if you need more capacity.

The production template enables the deployed deterministic state/reflection
features and evidence selection, and keeps semantic routing and layered retrieval off.
`MENO_SNAPSHOT_BASE_INTERVAL` controls periodic full snapshots.
Audit overflow defaults to a file next to the SQLite database;
`MENO_AUDIT_SPILL_PATH` can override it.

## PostgreSQL and Qdrant

The existing `docker-compose.yml` starts PostgreSQL and Qdrant on loopback.
Set `MENO_DATABASE_URL` to a PostgreSQL URL and `MENO_VECTOR_MODE=qdrant`
in the private service environment, then run `meno migrate` and start Meno.
Do not expose the database or vector service directly to the public network.

Keep the embedding provider/model/dimension and projection version consistent.
Changing any of them requires a new vector collection and a canonical replay:

```bash
meno rebuild-projection --batch-size 32
```

## Hermes

Install the profile plugin and set `memory.provider: meno`, as described in
[hermes-integration.md](hermes-integration.md). The profile and sidecar must use
the same token. Use one stable user identifier per single-user profile.

## Backups and upgrades

Back up the private environment file, the SQLite database, and any pending
audit overflow file. The Hermes profile spool lives under `HERMES_HOME/meno`
and is separate from the canonical database.

```bash
sudo systemctl stop meno
sudo sqlite3 /var/lib/meno/meno.db ".backup '/var/backups/meno.db'"
# Back up /etc/meno.env and /var/lib/meno/meno-audit-spill.jsonl if present.
sudo systemctl start meno
```

Stop the sidecar before updating the checkout. Reinstall, apply the schema,
and restart using the same private environment:

```bash
cd /opt/meno
git pull --ff-only
.venv/bin/pip install -e .
sudo sh -c 'set -a; . /etc/meno.env; set +a; /opt/meno/.venv/bin/meno migrate'
sudo systemctl restart meno
```

Vectors warm in the background at startup. Initial recall can be incomplete;
the service log reports completion or an embedding-provider failure.

For a v3-to-v2 rollback, restore the pre-upgrade database backup.
v2 cannot read v3 delta snapshots. Never discard a live database to downgrade.
