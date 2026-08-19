# Benchmark protocol

Meno v1.0 uses three separate evaluation lanes:

- `run_longmemeval.py`: component-level oracle evidence retrieval on the official
  LongMemEval dataset. The dataset is MIT licensed; raw data is ignored and is not
  redistributed by this repository.
- `run_personamem.py`: a retrieval-only adaptation of the official PersonaMem 32k
  multiple-choice set. It ranks answer options from retrieved facets with a fixed,
  non-LLM scorer. Its accuracy is **not comparable** to PersonaMem's official
  end-to-end LLM leaderboard.
- `run_negative_suite.py`: deterministic tenant-isolation, injection, provenance,
  sensitive-data, consent-revocation, and deletion checks created for Meno.
- `run_concurrency_smoke.py`: concurrent ingest, atomic revision, outbox completion,
  multi-facet retrieval, and latency smoke test.

Record dataset hashes, exact limits, backend configuration, latency, degraded
responses, and hardware for every reported run. LoCoMo is useful for research but
its CC BY-NC 4.0 terms make it unsuitable as a default commercial release gate.

LongMemEval and PersonaMem use `/v1/ingest/batch` by default so historical imports
exercise the same durable outbox while avoiding one HTTP transaction per event.
Production benchmark reports must record the Google model, output dimension,
projection version, batch sizes, API error/429 counts, and whether the run used
cached vectors. Never store the Google API key or raw credential file in results.
