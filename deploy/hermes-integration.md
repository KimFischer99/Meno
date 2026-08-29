# Connecting Hermes to Meno

Three environment variables in the Hermes profile. No changes to the Hermes core
loop, and no code changes on either side.

## Prerequisites

Meno running and healthy on the same host:

```bash
systemctl status meno
curl -s http://127.0.0.1:8765/health/ready
```

## Register the provider

Meno ships a `hermes_agent.memory_providers` entry point, so installing the
package into the environment Hermes runs from is enough for discovery:

```bash
# In the environment Hermes itself uses
pip install -e /opt/meno
```

Confirm Hermes can see it before wiring anything up — if discovery fails, the
variables below will silently do nothing:

```bash
python3 -c "
from importlib.metadata import entry_points
found = [e.name for e in entry_points(group='hermes_agent.memory_providers')]
print('discovered providers:', found)
"
```

## Configure the profile

Add to the active Hermes profile's `.env`:

```bash
MENO_API_URL=http://127.0.0.1:8765
MENO_API_TOKEN=<the same token as MENO_API_TOKEN in /etc/meno.env>
MENO_USER_ID=<stable identifier for this profile>
```

Optional, with its default:

```bash
MENO_PROVIDER_TIMEOUT_SECONDS=0.8
```

Notes on each:

- **`MENO_API_TOKEN` must match the sidecar's token exactly.** A mismatch produces
  401s that the provider swallows — it is fail-open by design — so Hermes keeps
  working with no memory and no error. Verify with the check below rather than
  assuming.
- **`MENO_USER_ID` must be stable across sessions.** Meno's whole purpose is
  cross-session user state; a per-session identifier makes preference evolution
  invisible. It must also be stable across restarts, so do not derive it from
  anything ephemeral.
- **`MENO_PROVIDER_TIMEOUT_SECONDS` bounds every call.** The default 0.8 s was
  chosen so a hung sidecar cannot stall a turn; the reliability suite measures a
  hanging server at 0.808 s, absorbed by this timeout.

## Verify the connection

```bash
# 1. The token works (401 means the tokens differ)
curl -s -o /dev/null -w '%{http_code}\n' \
  -H "Authorization: Bearer $MENO_API_TOKEN" \
  -X POST http://127.0.0.1:8765/v1/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"probe","purpose":"response_personalization",
       "context":{"query":"hello","task_type":"conversation_recall"},
       "constraints":{"max_facets":4}}'
# Expect 200. 401 = token mismatch. Connection refused = sidecar not running.

# 2. State is accumulating after some real usage
curl -s -H "Authorization: Bearer $MENO_API_TOKEN" \
  "http://127.0.0.1:8765/v1/revisions/$MENO_USER_ID"
```

If the revision stays at zero while Hermes is in use, the provider is not being
invoked — recheck entry-point discovery and that the variables are in the profile
Hermes actually loaded.

## What failure looks like

Meno is a sidecar, not a dependency. Measured behavior (Gate Charter #11): with the
port dead or the server accepting connections but never answering, all seven
provider entry points stay non-blocking and Hermes continues. Turns that happen
during an outage are held in a durable spool rather than dropped.

So the failure mode is *degraded personalization*, not a stalled agent. The
corollary is that a broken connection is silent: nothing will complain, which is
why the verification step above matters.

## Making corrections easy is not optional

Calibration (Charter #15) needs at least 10 negative outcomes — cases where the
user said a remembered preference was wrong. If correcting a memory is hard to
reach in the interface, that gate never closes no matter how long Meno runs.

The endpoint is `POST /v1/feedback` with `reject` or `correct`. Surface it
somewhere a user will actually use.

## Turning it off

Remove the three variables from the profile and restart Hermes. Nothing in
Hermes depends on Meno being present. The sidecar can keep running, or:

```bash
sudo systemctl stop meno
```

The canonical store is untouched either way; re-adding the variables resumes with
all prior state intact.
