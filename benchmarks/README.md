# Benchmark protocol

Meno v1.0 uses three separate evaluation lanes:

- `run_longmemeval.py`: component-level oracle evidence retrieval on the official
  LongMemEval dataset. The dataset is MIT licensed; raw data is ignored and is not
  redistributed by this repository.
- `run_personamem.py`: a retrieval-only adaptation of the official PersonaMem 32k
  multiple-choice set. It ranks answer options from retrieved facets with a fixed,
  non-LLM scorer. Its accuracy is **not comparable** to PersonaMem's official
  end-to-end LLM leaderboard.
- `run_personamem_e2e.py`: an end-to-end LLM-answer evaluation on the same
  PersonaMem 32k set. It ingests contexts, retrieves facets, injects only the
  rendered `<user_context>` into the answer LLM (official `<final_answer>`
  protocol), and scores answer-level accuracy per question type. Retrieval
  quality (correct support, MRR of the legacy scorer on the same facets),
  answer quality, latency, and token cost are recorded separately. LLM and Meno
  credentials are read from files/environment at runtime and never written to
  result JSON.
- `serve_local_dev.py`: starts a local Meno API with the deterministic
  `tests.fakes.TestEmbedder` (SQLite + in-memory vectors) via the public
  `create_app(embedder=...)` injection point, so the end-to-end lane runs
  without an embedding provider. Results are comparable across Meno revisions
  on this backend, not to the production Google-embedding deployment.
- `run_negative_suite.py`: deterministic tenant-isolation, injection, provenance,
  sensitive-data, consent-revocation, and deletion checks created for Meno.
- `run_concurrency_smoke.py`: concurrent ingest, atomic revision, outbox completion,
  multi-facet retrieval, and latency smoke test.

## Running the PersonaMem end-to-end lane

```bash
python benchmarks/serve_local_dev.py --port 8766 --api-token "$MENO_API_TOKEN" &
python benchmarks/run_personamem_e2e.py \
  --questions artifacts/benchmarks/raw/questions_32k.csv \
  --contexts artifacts/benchmarks/raw/shared_contexts_32k.jsonl \
  --output artifacts/benchmarks/results/local-personamem-e2e-full.json \
  --api-url http://127.0.0.1:8766 --api-token-file artifacts/runtime/meno_api_token \
  --llm-base-url http://127.0.0.1:8317/v1 --llm-model glm-5.2 \
  --llm-api-key-file artifacts/runtime/llm_api_key --cleanup
```

The full 589-question local run is stored at
`artifacts/benchmarks/results/local-personamem-e2e-full.json` (dataset hashes and
run configuration embedded). On the deterministic TestEmbedder backend it scores
answer accuracy 0.579 overall versus 0.409 for the legacy retrieval-only scorer
on the same facets; the largest per-type gains are `track_full_preference_evolution`
(0.31 → 0.64) and `suggest_new_ideas` (0.02 → 0.28).

Record dataset hashes, exact limits, backend configuration, latency, degraded
responses, and hardware for every reported run. LoCoMo is useful for research but
its CC BY-NC 4.0 terms make it unsuitable as a default commercial release gate.

LongMemEval and PersonaMem use `/v1/ingest/batch` by default so historical imports
exercise the same durable outbox while avoiding one HTTP transaction per event.
Production benchmark reports must record the Google model, output dimension,
projection version, batch sizes, API error/429 counts, and whether the run used
cached vectors. Never store the Google API key or raw credential file in results.
