# Meno production profile: single small host

Configuration for running Meno as a sidecar next to Hermes on a small VPS
(2 GB RAM, ~1 GB available). SQLite is the canonical store and vectors live in
the Meno process; no PostgreSQL, no Qdrant, no Docker.

**This directory contains no secrets.** Every credential is a `replace-with-...`
placeholder. Fill them in on the host, in a file that is never committed.

## Why this profile exists

The default deployment (`docker-compose.yml` at the repository root) runs
PostgreSQL and Qdrant as separate containers. Measured resident sizes make that
unworkable here: PostgreSQL typically holds 150–250 MB, Qdrant 100–200 MB, and the
Meno process itself 130 MB — over the budget before storing anything.

This profile trades horizontal headroom for fitting on one host:

| | Default | This profile |
|---|---|---|
| Canonical store | PostgreSQL container | SQLite file |
| Vector view | Qdrant container | In-process, `array("f")` |
| Processes | 3 | 1 |
| Memory floor | ~400–600 MB | ~130 MB |
| Per-claim vector cost | (in Qdrant) | 4.1 KB at 1024 dims |

SQLite as the canonical store does not weaken the audit guarantees. What
`Meno_SPEC.md` forbids is the *vector store* becoming the source of truth; the
full-state replay lane (Gate Charter #12) already runs on SQLite and passes.

## Memory budget

```text
Meno process baseline                    ~130 MB   (measured)
Vectors, 1024 dims, array("f")            4.1 KB   per claim
  5,000 claims                            ~20 MB
 20,000 claims  (default cap)             ~80 MB
 50,000 claims                           ~199 MB
```

`MENO_VECTOR_MAX_RESIDENT` (default 20,000) bounds the vector count. Past the cap
writes fail and `/health/ready` reports degraded, rather than the process growing
until the kernel kills it — an OOM kill would take the sidecar down, and Charter
#11 requires that a sidecar failure never block Hermes. Vectors are recoverable:
`rebuild_projection` restores them from canonical storage.

`meno.service` also sets `MemoryMax=` as a kernel-level backstop. Two independent
limits, because the application-level one only covers vectors.

Retrieval scans one user's vectors, not the whole store. Measured single-user
scans at 1024 dims: 200 claims 7.4 ms, 500 claims 15.6 ms, 2,000 claims 66.9 ms —
inside the 1,000 ms budget of Charter #7 with room to spare.

## Install

Requires Python 3.11+ and git on the host.

```bash
# 1. Fetch the repository (the Hermes host pulls updates the same way)
sudo mkdir -p /opt/meno /var/lib/meno
sudo chown "$USER" /opt/meno /var/lib/meno
git clone <your-fork-url> /opt/meno
cd /opt/meno

# 2. Virtualenv and install
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e .

# 3. Configuration
cp deploy/.env.production.example /etc/meno.env
sudo chmod 600 /etc/meno.env        # contains the API token and embedding key
sudo chown root:root /etc/meno.env
# Now edit /etc/meno.env and replace every `replace-with-...` value.

# 4. Generate the API token (32+ characters, required in production)
python3 -c "import secrets; print(secrets.token_urlsafe(32))"

# 5. Verify the configuration before installing the service
deploy/start.sh --check
```

`start.sh --check` loads the environment file, constructs `Settings.from_env()`,
and exits non-zero with the reason if the profile would be unsafe.
`Settings.validate()` refuses to start a production profile with a missing or short
`MENO_API_TOKEN` (the API middleware skips authentication entirely when the token
is empty), a relative or `/tmp` SQLite path (canonical state must survive a
reboot), or an uncapped in-memory vector store.

Then apply the schema once:

```bash
set -a; . /etc/meno.env; set +a
/opt/meno/.venv/bin/meno migrate
```

## Run as a service

```bash
sudo cp deploy/meno.service /etc/systemd/system/meno.service
# Review User=, WorkingDirectory=, and MemoryMax= before enabling.
sudo systemctl daemon-reload
sudo systemctl enable --now meno
systemctl status meno
curl -s http://127.0.0.1:8765/health/ready
```

`/health/ready` reports `database`, `vector`, and whether the store is degraded.
It is not behind authentication; `/v1/*` is.

## Connect Hermes

See [hermes-integration.md](hermes-integration.md). Three environment variables in
the Hermes profile, no changes to the Hermes core loop.

## Back up and restore

Everything durable is one SQLite file. Use SQLite's own backup so a copy taken
mid-write is consistent:

```bash
sudo systemctl stop meno        # or use .backup while running
sqlite3 /var/lib/meno/meno.db ".backup '/var/backups/meno-$(date +%F).db'"
sudo systemctl start meno
```

To restore: stop the service, put the file back, start it. Vectors rebuild from
canonical storage on demand; they are a derived view and are not backed up.

## Roll back

```bash
sudo systemctl stop meno
cd /opt/meno && git checkout <previous-tag>
.venv/bin/pip install -e .
sudo systemctl start meno
```

The canonical store is forward-compatible within a minor version. Rolling back
across a schema change requires restoring the matching database backup.

## Upgrade

```bash
cd /opt/meno && git pull
.venv/bin/pip install -e .
sudo systemctl restart meno
```

Restarting drops the in-process vectors; they are rebuilt as retrieval requests
arrive, so the first few requests after a restart may return fewer facets. The
canonical store is untouched.

## Scope

This profile is component-level production readiness: it runs, it is bounded, and
it passes the correctness and safety gates. It is not a `production GO` claim —
that additionally needs calibration, human-preference, and social-prior evidence
which cannot exist until real traffic has been flowing. See
`Meno_v2.0_Gate_Charter.md` §4.
