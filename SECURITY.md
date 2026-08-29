# Security Policy

## Reporting a vulnerability

Please report security issues privately rather than in a public issue: open a
[GitHub security advisory](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
on this repository. Include the version or commit, what you observed, and a
reproduction if you have one.

Please do not include real user data, API keys, or database contents in a report.

## What this project treats as a vulnerability

Meno stores personal memory and injects it into agent prompts, so its security
properties are about disclosure, not just availability. The following are
vulnerabilities, and each has a gate in `Meno_v2.0_Gate_Charter.md`:

- **Cross-user disclosure.** Any path by which one user's claim reaches another
  user's context.
- **Consent bypass.** Retrieval returning a claim whose consent was revoked, or
  serving one under a purpose its source never consented to.
- **Deletion that does not propagate.** A deleted claim still reachable through
  the vector index or a materialized snapshot.
- **Unauthenticated access.** Reaching `/v1/*` without a valid bearer token.
- **Provenance loss.** An injected facet that cannot be traced to the events it
  came from, or an audit chain that can be altered without detection.
- **Prompt injection through stored memory.** Content in an ingested event
  causing Meno to emit a claim that changes agent behavior.

Denial of service against a self-hosted single-tenant sidecar is lower severity,
but report it if a remote input can crash or wedge the process.

## Operator responsibilities

Some properties depend on deployment, not on code:

- **`MENO_API_TOKEN` must be set.** Authentication is skipped entirely when the
  token is empty. Production configuration validation rejects a missing or short
  token, but a `development` profile will not.
- **Bind to loopback.** Meno is a sidecar for a process on the same host.
  Exposing it publicly puts a personal memory store on the network; if you must,
  terminate TLS in front of it and keep the token secret.
- **Protect the environment file and the database.** `/etc/meno.env` holds the API
  token and embedding key; the SQLite file holds every stored claim. Both should be
  `600` and owned by the service user.
- **Embedding text leaves the host.** Non-sensitive claim values are sent to the
  configured embedding provider. Claims marked sensitive are never embedded.

See `deploy/README.md` for the configuration that satisfies these.

## Supported versions

This is pre-1.0 software under active development. Fixes land on the default
branch; there are no maintained release branches yet.
