# Sealed historical sources

These files preserve the exact, previously uncommitted source bytes bound by the A3, A3.1, and A4 experiment configs and artifacts. They are audit inputs only: current runners and application code must not import them.

| Experiment | File | SHA-256 |
|---|---|---|
| A3 invalidated holdout | `a3/semantic_router.py` | `c90fa64f887e104943d66ef9b21e7b03ad1cb8212fd9c5d8c1e9d5ad335809fd` |
| A3.1 provider holdout | `a31/semantic_router.py` | `dd1fe5f08c91779c08849f508d0abc2febf02c4319dbc2d353a3277840a326bf` |
| A3/A3.1 development freeze | `a3_a31/run_semantic_router_a3_development.py` | `1a3230b85f585e04b2a3ca26c3036a54746df6d33aa2e72f02b9252c772af118` |
| A4 reviewed development selection | `a4/semantic_router.py` | `7f6edd01e4345525255fa6f39e49d8fb34c26809f92d40e18f80b4aaa957d6a1` |

The snapshots were recovered read-only from the experiment host, where pre-change backups had been retained. Their hashes exactly match the frozen configs/manifests. The invalidated A3 snapshot is retained only to make the invalidation auditable; it does not restore GO status. The A4 snapshot was sealed from the committed pre-A5 router before the Phase A5 topic-gated strategy was introduced; its `router_sha256` was verified against the A4 reviewed development-selection artifact, which is not published here because of its size.

## Redaction of 2026-08-29

Before publication, `artifacts/vps-stage4-a31-provider-holdout.json` had the
experiment host's hostname and home-directory paths stripped, because publishing
real infrastructure identifiers is not acceptable and those fields carry no
evaluative meaning. The A4 and A5 development runners pin that file by SHA-256, so
the pin in each was updated in the same change:

| | SHA-256 |
|---|---|
| Before redaction (as produced on the experiment host) | `eb0b49b06028b076a18eb14c3792e6d1cea470633ec2861dc391234d390ae8eb` |
| After redaction (current, pinned in the runners) | `073a14779f3980d8139e88ce6dea137340ac5b65cb3586491f47f8697d3253ed` |

Only `host` and filesystem-path strings changed; every score, threshold, decision,
and router hash in the artifact is untouched. The pre-redaction hash is recorded
here so the chain from the original run remains checkable. Note that
`artifacts/vps-stage4-a5-development-selection.json` still records the
pre-redaction hash as its input, which is correct — it documents the run as it
happened.
