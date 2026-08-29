#!/usr/bin/env bash
# Start (or validate) the Meno sidecar in the single-host production profile.
#
#   deploy/start.sh --check   validate configuration and exit
#   deploy/start.sh           validate, then exec the server
#
# systemd invokes this with EnvironmentFile=/etc/meno.env already applied. Run it
# by hand and it will source that file itself, so the same checks apply either way.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${MENO_PYTHON:-$ROOT/.venv/bin/python}"
ENV_FILE="${MENO_ENV_FILE:-/etc/meno.env}"

if [[ ! -x "$PYTHON" ]]; then
  echo "start.sh: no interpreter at $PYTHON" >&2
  echo "  create it with: python3 -m venv $ROOT/.venv && $ROOT/.venv/bin/pip install -e $ROOT" >&2
  exit 1
fi

# Only load the file when the environment is not already populated, so systemd's
# EnvironmentFile stays authoritative and we never override it.
if [[ -z "${MENO_ENV:-}" ]]; then
  if [[ -r "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    . "$ENV_FILE"
    set +a
  else
    echo "start.sh: $ENV_FILE is not readable and MENO_ENV is unset" >&2
    echo "  cp $ROOT/deploy/.env.production.example $ENV_FILE && chmod 600 $ENV_FILE" >&2
    exit 1
  fi
fi

# Fail before binding a port. Settings.validate() is where the production profile
# is enforced -- API token length, durable SQLite path, bounded vector residency.
# Printing the resolved profile (never the secrets) makes a misconfigured host
# obvious in the journal instead of surfacing later as odd behavior.
"$PYTHON" - <<'PYCHECK'
import sys

from meno.config import Settings

try:
    settings = Settings.from_env()
except ValueError as exc:
    print(f"configuration rejected: {exc}", file=sys.stderr)
    raise SystemExit(2) from None

print(
    "meno config ok: "
    f"env={settings.environment} "
    f"store={settings.database_url.split('://')[0]} "
    f"vectors={settings.vector_mode} "
    f"max_resident={settings.vector_max_resident} "
    f"listen={settings.api_host}:{settings.api_port}"
)
PYCHECK

if [[ "${1:-}" == "--check" ]]; then
  exit 0
fi

# exec so systemd supervises the server directly rather than this wrapper.
exec "$PYTHON" -m meno.cli serve
