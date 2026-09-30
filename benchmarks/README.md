# Benchmark protocol

These historical evaluation tools are retained for reproducibility and are not
required for v3 installation. Generated results and corpora under `artifacts/`
are local-only. Tests requiring archived caches skip when files are absent;
synthetic checks remain runnable.

> **PersonaMem no longer gates this project.** As of 2026-08-27 it is regression
> monitoring only -- useful for "did a change break existing behavior", not for
> "is the user model better". Decisive tokens appear in 8.7% of queries, 55.5% of
> questions have none at all, and the answer-key oracle overlaps a deployable
> query-conditioned selector by only 0.116 (`probe_discriminability.py`). Four
> architecture changes each improved their mechanism while this metric stayed flat
> or fell. The intended acceptance metrics are in `Meno_SPEC.md` 1340-1372; see
> `Meno_v2.0_Gate_Charter.md` for the evidence and the route forward.

Meno v2.0 uses three separate evaluation lanes:

- `run_longmemeval.py`: component-level oracle evidence retrieval on the official
  LongMemEval dataset. The dataset is MIT licensed; raw data is ignored and is not
  redistributed by this repository. **"oracle" is load-bearing**: in the
  `longmemeval_oracle` variant the haystack equals the answer set for 500/500
  questions, so no retrieved session can fall outside it and `precision@10`
  collapses to "did anything come back at all". Measured 500-question result is
  `recall@10 0.8645`; it is an upper bound at session granularity and does not
  evidence Gate Charter #13 (see `Meno_v2.0_Gate_Charter.md` §3.4).
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
  on this backend, not to the production SiliconFlow-embedding deployment.
- `run_negative_suite.py`: deterministic tenant-isolation, injection, provenance,
  sensitive-data, consent-revocation, and deletion checks created for Meno.
- `run_state_replay_determinism.py`: the SPEC :1393 full-state replay lane (Gate
  Charter #12). See below.
- `run_grounding_temporal_suite.py`: evidence attribution coverage, unsupported
  injected claim rate, and stale-active rate (Gate Charter #8, #9, #10). See below.
- `run_reliability_outage_suite.py`: sidecar and Qdrant outage behavior (Gate
  Charter #11). See below.
- `probe_curated_viability.py`: pre-check for whether a curated retrieval set is
  worth building on a given corpus (Gate Charter #13/#14). See below.
- `validate_external_corpus.py`: validates an externally exported corpus against
  the contract in `MENO_DATA_REQUIREMENTS.md` and predicts annotation viability via
  `query_signal`. Run it on a 20-30 query sample before annotating the full set.
- `run_concurrency_smoke.py`: concurrent ingest, atomic revision, outbox completion,
  multi-facet retrieval, and latency smoke test.
- `run_preference_calibration.py`: deterministic Brier Score, ECE, NLL, and
  coverage diagnostic for `PreferenceDistribution`. Prospective outcome labels
  are not ingested before scoring. Its bundled labels are synthetic, so this
  lane can block but never grant production GO.

Run the calibration diagnostic from the repository root:

```bash
PYTHONPATH=. .venv/bin/python benchmarks/run_preference_calibration.py \
  --output artifacts/benchmarks/results/local-preference-calibration-v2.json \
  --enable-v2
```

The v1 diagnostic compares `mode_probability` with the active extractor claim
confidence. This intentionally tests whether relative probability among observed
labels can be interpreted as probability that the current mode is correct. A
single observed label currently receives probability `1.0`, so a failed gate is
evidence that those semantics must remain `uncalibrated`, not a threshold-tuning
request. With `--enable-v2`, the candidate score is `actionable_probability`, which
multiplies the conditional mode probability by an independent Beta belief containing
unit unknown mass. This remains an uncalibrated experiment until evaluated against
real prospective feedback.

`run_real_feedback_calibration.py` is the next evidence lane. It opens the canonical
database in a read-only transaction, pairs each preference feedback audit at revision
`r` with User Token revision `r - 1`, and maps `confirm` to a positive outcome and
`reject`/`correct` to a negative outcome. Pairing failures are excluded and counted;
the report contains no user text, correction text, claim values, token payloads, database
URL, or feedback timestamps.

```bash
PYTHONPATH=. .venv/bin/python benchmarks/run_real_feedback_calibration.py \
  --database-url "$MENO_DATABASE_URL" \
  --output artifacts/benchmarks/results/real-feedback-calibration.json
```

The frozen readiness gate requires at least 100 paired preference outcomes and a
30-sample holdout containing both positive and negative labels. Insufficient data,
imperfect revision pairing, or an `uncalibrated` strategy always remains NO-GO.

## Full-state replay determinism (SPEC :1393, Gate Charter #12)

`run_state_replay_determinism.py` replays the frozen `ucm-oracle-v1.json` event log
from an empty store — twice, into two independent databases — and compares the whole
semantic state per dimension rather than an aggregate score. SPEC :1393 states the
purpose directly: detect whether swapping the extractor, the embedding, or the decay
policy quietly changed one person's model.

```bash
# Compare against the frozen baseline. Exits 1 on drift or a failed self-check.
PYTHONPATH=. .venv/bin/python benchmarks/run_state_replay_determinism.py \
  --output artifacts/benchmarks/results/local-state-replay-determinism-v1.json

# Re-freeze after an intended architecture change. A changed digest means the
# stored user model changed, so record why.
PYTHONPATH=. .venv/bin/python benchmarks/run_state_replay_determinism.py \
  --output /tmp/replay.json --write-baseline
```

Seven of the eight dimensions SPEC :1393 names are covered.
**`community_memberships` is reported as `covered: false`, not digested as an empty
set** — it has no table, projection, or retrieval path (Gate Charter #17), and an
empty set would hash to a constant and compare equal forever, reading as a passing
check for something that does not exist. The report therefore always carries
`spec_1393_fully_covered: false`; seven passing dimensions is not SPEC compliance.

Two properties keep the lane from passing vacuously:

- Fields that no replay can reproduce (`uuid4` audit and trace ids, `created_at`,
  the supersede `valid_to` instant) are excluded, or digests would never match. But
  over-exclusion makes them always match, so each dimension's `excluded_fields` is
  listed in the report for review instead of being implicit in the digest code.
  `claim_id` is a uuid5 of the derivation key, is reproducible, and stays in scope.
- `--self-check` (on by default) perturbs the inputs and **requires** the named
  dimensions to respond: truncating each case's event log, changing the extractor
  version, and changing the embedding dimension. A digest that no longer reacts to
  its own perturbation fails the lane exactly like drifted state would. Skipping the
  self-check is a diagnostic mode and can never pass the lane.

The baseline is bound to the fixture's SHA-256, so editing the event log invalidates
the frozen digests instead of silently comparing against the wrong log.



## Grounding and temporal gates (Gate Charter #8, #9, #10)

`run_grounding_temporal_suite.py` measures three SPEC :1340-1372 metrics over one
shared surface — the facets Meno actually injects — because all three are
properties of that same set.

```bash
# Fixture only: 5 facets, so #9 and #10 report resolvable:false and do NOT pass.
PYTHONPATH=. .venv/bin/python benchmarks/run_grounding_temporal_suite.py \
  --output /tmp/grounding.json

# Real corpus: 4712 facets, all three thresholds resolvable.
PYTHONPATH=. .venv/bin/python benchmarks/run_grounding_temporal_suite.py \
  --output artifacts/benchmarks/results/local-grounding-temporal-suite-v1.json \
  --contexts artifacts/benchmarks/raw/shared_contexts_32k.jsonl \
  --questions artifacts/benchmarks/raw/questions_32k.csv --contexts-limit 37
```

Measured on 37 contexts / 589 questions / 3363 events / 4712 injected facets:
coverage 4712/4712, unsupported 0/4712, stale-active 34/4712 (0.72%, under the 2%
bound). Sample resolution 0.00021.

Two properties keep the lane from passing vacuously:

- **Sample resolution is reported, not assumed.** With N facets the finest rate
  distinguishable from zero is 1/N, so a "< 1%" claim from 15 facets is not
  evidence. When 1/N exceeds a threshold the metric reports `resolvable: false`
  and does not pass — which is what happens on the committed fixture.
- **Queries come from the frozen PersonaMem question set, not invented probes.**
  Invented queries measure whichever wording happens to clear the lexical
  activation threshold: on one context, 15 hand-written queries yielded 4 facets
  while 10 real questions yielded 80. `--questions` is therefore required with
  `--contexts`.

**#9 cannot fail under the current implementation.** The deterministic extractor
copies event text into claim values, so token-level support holds by construction
(measured 0/449). Do not read 0% as "injected inference is grounded" — the metric
is a regression guard that would catch a future generative extractor fabricating
content. `--self-check` (on by default) therefore injects three synthetic defects
at the storage layer and requires each metric to catch its own: a fabricated value
(#9), a stripped evidence link (#8), and a "today" claim still injected 60 days
later (#10).

One implementation detail worth knowing: episodic values are truncated at
`EPISODIC_MAX_CHARS` mid-word, leaving fragments ("experie") present in no event.
Excluding the trailing partial token removes 157/285 false violations.

## Reliability outage behavior (Gate Charter #11)

`run_reliability_outage_suite.py` covers the two SPEC reliability conditions at the
layers where they happen.

```bash
PYTHONPATH=. .venv/bin/python benchmarks/run_reliability_outage_suite.py \
  --output artifacts/benchmarks/results/local-reliability-outage-suite-v1.json
```

**Sidecar outage** runs through `MenoMemoryProvider`, the real Hermes integration
surface, against both a dead port and a server that accepts the connection and
never answers. The hang is the case that actually threatens an agent loop; a
refused connection fails fast. All 7 provider entry points must return without
raising and within budget. Measured: `blocked_rate 0.0`, with `prefetch` bounded at
0.808s by the provider's own timeout rather than the 30s hang.

The lane also asserts the durable spool retained the turn taken during the outage:
**non-blocking must not mean silently dropped.**

**Qdrant outage** fails the vector read path inside a live service. Retrieval must
report `degraded: true` *and* still serve from canonical — claiming health would
hide the outage, and returning nothing would block personalization as surely as an
exception would.

Scope: this is component-level evidence. It does not run inside Hermes, because no
Hermes deployment consumes Meno today (the plugin is registered but not wired).

## Curated retrieval set viability (Gate Charter #13/#14 pre-check)

`probe_curated_viability.py` answers one question before any annotation budget is
spent: *is a curated set on this corpus worth building, and at what cost.* It emits
no relevance labels and computes no Recall@10 or nDCG@10.

```bash
PYTHONPATH=. .venv/bin/python benchmarks/probe_curated_viability.py \
  --output artifacts/benchmarks/results/local-curated-viability-probe-v1.json
```

**Measured verdict on PersonaMem: `CORPUS_UNSUITABLE_QUERY_SIGNAL`.** Decisive
tokens appear in the query for 0.0871 of questions — indistinguishable from the
0.087 that made this corpus unusable as a gate in the first place. A curated set
over the same corpus inherits the same ceiling regardless of how its labels are
produced, so the annotation budget would buy nothing.

The checks run in a deliberate order — query signal first, label quality second.
Reversed, a corpus with clean labels and no query signal reads as "ready to
annotate", which is exactly the trap.

On label sources, the probe also settles a question worth recording: PersonaMem's
`distance_to_ref_proportion_in_context` resolves for all 589 questions with exact
`context_length_in_letters` agreement, and it is dataset provenance rather than
answer-reading, so it *is* admissible under the annotation constraint. But it is
not precise enough — 0.161 decisive recall at the reference message itself, rising
to 0.486 only across a 9-message window against a 0.074 random baseline. That
localizes a region, not a claim; Recall@10 built on it would track label noise.

If a different corpus is sourced, the required annotation volume is derived rather
than guessed: separating Recall@10 0.90 from 0.85 at 95% confidence needs 139
labeled queries, the same sample-resolution rule the grounding lane applies. All
four verdicts are reachable (`ANNOTATE_WITH_SEEDS`, `MANUAL_ANNOTATION_REQUIRED`,
`CORPUS_UNSUITABLE_QUERY_SIGNAL`, `CORPUS_UNSUITABLE_NO_LABEL_SOURCE`) — the probe
is not hardwired to reject.

## Screening an external corpus (Gate Charter #13/#14 candidate intake)

`validate_external_corpus.py` runs the same go/no-go on a corpus exported to the
`MENO_DATA_REQUIREMENTS.md` §1.3 contract. `convert_longmemeval_corpus.py` is the
worked example of getting a public dataset into that contract, and the pattern
generalizes: emit no labels of your own, take them from the dataset's own
provenance, and record every conversion decision in the report so a reader can
audit what the resulting number means.

```bash
PYTHONPATH=. .venv/bin/python benchmarks/convert_longmemeval_corpus.py \
  --data <longmemeval_oracle.json> \
  --sessions-out /tmp/lme/sessions.jsonl --queries-out /tmp/lme/queries.jsonl \
  --report-out artifacts/benchmarks/results/local-external-corpus-longmemeval-conversion-v1.json
PYTHONPATH=. .venv/bin/python benchmarks/validate_external_corpus.py \
  --sessions /tmp/lme/sessions.jsonl --queries /tmp/lme/queries.jsonl \
  --output artifacts/benchmarks/results/local-external-corpus-longmemeval-v1.json --sample
```

**Measured verdict on LongMemEval: `STOP_DO_NOT_ANNOTATE`, query_signal 0.0558**
(500 queries, 68.8% with zero signal) — below the 0.087 that disqualified
PersonaMem. Not a denominator artifact: the ceiling if a query named every token
it could is 0.7592. LongMemEval's questions are largely cross-event comparisons
("which did I do first, A or B?") that name the compared objects while the
deciding tokens are dates and ordinals inside the evidence.

Two things this screening turned up that generalize to the next candidate:

- **A dataset's own abstention subset must be found, not assumed absent.**
  LongMemEval marks 30 questions with an `_abs` id suffix whose gold answer is
  "the information provided is not enough". The first conversion treated them as
  ordinary positives and used their `has_answer` flags — which point at evidence
  for a *related* question — as labels. Corrected, the corpus has 30 abstain
  queries and drops nothing. `tests/test_convert_longmemeval_corpus.py` locks this.
- **Per-session timestamps need within-session offsets.** A session date applied
  to every turn makes the validator's `sessions_with_one_timestamp` check fire,
  and correctly so: decay, stale-active, and supersede ordering all read that
  field.

Its abstain share (30/500 = 0.06) is also below the contract's 0.20–0.40, so even
a passing query signal would have needed more negatives before use.

### The candidate pool is exhausted

Four public memory corpora were screened, all with the same validator and
thresholds (STOP < 0.15, GOOD ≥ 0.35). All four land in a narrow band well below
STOP, and a fifth was excluded one step earlier for having no admissible label:

| Corpus | query_signal | Ceiling | Verdict | Screener |
|---|---:|---:|---|---|
| PersonaMem | 0.0871 | — | STOP | `probe_curated_viability.py` |
| LoCoMo | 0.0836 | 0.8893 | STOP | `convert_locomo_corpus.py` |
| LongMemEval | 0.0558 | 0.7592 | STOP | `convert_longmemeval_corpus.py` |
| BEAM 100K | 0.0410 | 0.9881 | STOP | `convert_beam_corpus.py` |
| HaluMem Medium | not measured | — | no label source | `screen_halumem_label_source.py` |

"Ceiling" is the score a query would get by naming every distinctive token it
could. Each measured value sits an order of magnitude below its own ceiling, so
none of these is a denominator artifact — the queries genuinely do not name the
evidence.

**These are not defects in the datasets.** Each serves its own purpose (LoCoMo
multi-hop dialogue QA, BEAM long-horizon agent memory, HaluMem memory
hallucination); none happens to satisfy the one precondition a Recall@10 gold set
needs. Cite this table with that sentence attached.

Per-corpus failure modes differ while pointing the same way — the query describes
*what is being asked*, not *which evidence answers it*:

- **BEAM** has the cleanest label pointers (`source_chat_ids` resolved 55/55 on a
  first pass) and the lowest signal. `event_ordering`, `summarization`, and
  `preference_following` score exactly 0.0000 — asking "list the order in which I
  brought things up" cannot name any single piece of evidence. Caveat: 163 of 335
  inject queries were skipped for having no distinctive tokens and the median
  distinctive set is 1 token, so BEAM's number has the lowest resolution here.
- **LoCoMo** has the best label granularity (`evidence` names exact `dia_id`s;
  9 of 2815 pointers malformed) and the highest signal of the four, still 0.0836.
- **HaluMem** stops before query signal is measured, on purpose. Its `evidence`
  names author-written third-person summaries, not message indices, and the
  dataset's own `event_source` pointer covers only 1727/14948 memory points. A
  text-matching fallback is ambiguous for 34.7% of evidence items (best match ties
  the runner-up) with only 3.1% resolving to a single message at ≥0.8 overlap.
  Measuring query signal against labels the screener invented would be measuring
  the matcher, which Charter §3 rule 2 excludes.

Screening any further corpus follows the same shape. Two rules earned the hard
way, both locked by tests:

- **Find the dataset's own abstain split; do not assume it has none.** All three
  new corpora have one under a different name: LongMemEval's `_abs` id suffix,
  LoCoMo's category 5 (`adversarial_answer`), BEAM's `abstention` category. The
  first LongMemEval pass mislabeled its 30 as positives *and* used their
  `has_answer` flags — which point at evidence for a related question — as labels.
- **Per-session timestamps need within-session offsets.** A session date applied
  to every turn trips the validator's `sessions_with_one_timestamp` check, and
  correctly so: decay, stale-active, and supersede ordering all read that field.
  BEAM additionally sets `time_anchor` only when it changes, so it must be
  forward-filled.

Raw datasets are not redistributed here. Licenses: LoCoMo CC BY-NC 4.0, BEAM
CC BY-SA 4.0, HaluMem CC BY-NC-ND 4.0, LongMemEval MIT. BEAM ships as Parquet and
needs `pyarrow`, imported inside `main()` only — it is not a Meno runtime
dependency and is not in `pyproject.toml`.

```bash
PYTHONPATH=. .venv/bin/python benchmarks/convert_locomo_corpus.py \
  --data <locomo10.json> --sessions-out /tmp/locomo/sessions.jsonl \
  --queries-out /tmp/locomo/queries.jsonl \
  --report-out artifacts/benchmarks/results/local-external-corpus-locomo-conversion-v1.json

PYTHONPATH=. .venv/bin/python benchmarks/convert_beam_corpus.py \
  --data <beam_100k.parquet> --sessions-out /tmp/beam/sessions.jsonl \
  --queries-out /tmp/beam/queries.jsonl \
  --report-out artifacts/benchmarks/results/local-external-corpus-beam-conversion-v1.json

PYTHONPATH=. .venv/bin/python benchmarks/screen_halumem_label_source.py \
  --data <HaluMem-Medium.jsonl> \
  --output artifacts/benchmarks/results/local-external-corpus-halumem-screen-v1.json
```


## Reporting an intervention (Gate Charter §6)

Every architecture intervention reports a mechanism-level delta **and its
denominator**. Four interventions in this project each moved their mechanism while
the gate metric stayed flat; "preference facet/question +83%" was not reported
alongside that facet's share of total injection, so a +0.0017 score change could
not be separated from a denominator too small to matter.

`local-layered-retrieval-diagnostic-v1.json` is the format: `baseline` / `layered`
/ `delta` sections where `state_layer_share_rendered` (0.0909 → 0.5) sits next to
the absolute `rendered_kind_totals` it derives from. Nested variants follow
BIST-POI's ablation shape — add one mechanism per layer, report each layer's
delta, and when a layer's gain is small, explain the denominator from the data
rather than attributing it to the mechanism.


## Stage 4 semantic-router evidence and integration lane

Historical A3–A5 experiments are immutable evidence. Runtime/write-path integration
is separately controlled by the default-off Phase B feature flag:

- `run_semantic_router_a3_development.py` uses the revealed v2 fixture for
  prototype-ensemble development selection.
- `run_semantic_router_a3_holdout.py` preserves the invalidated original A3
  protocol for audit. Its apparent 49/49 result had zero provider-scored cases
  and must not be cited as GO evidence.
- `run_semantic_router_a31_holdout.py` is the corrected sealed provider holdout.
  Its valid result is NO-GO because it contains one false merge and a robustness
  failure.
- `run_semantic_router_a4_development.py` pools only revealed evidence and compares
  `top2_mean` with `role_top2_mean`. It pins all inputs, requires exact provider
  coverage, records strategy provenance, and exits 3 when no eligible policy
  exists. A4 did not produce a fresh holdout because its development gate failed.
- `run_semantic_router_a5_diagnostics.py` dumps per-anchor/per-role cosine
  diagnostics for the revealed development set. Its evidence refuted the A4
  topic-gated direction: topic-only ranking misclassifies answer_style in
  15/15 cases because first-person register similarity dominates noun signal.
- `run_semantic_router_a5_anchor_search.py` embeds the development set against
  several redesigned anchor variants in one provider batch. The winning variant
  was frozen as `semantic-prototypes-a5-v2.json` (English-only anchors with
  broadened answer_style polarity coverage).
- `run_semantic_router_a5_development.py` selects the A5 policy across three
  independent provider runs; a threshold point is eligible only if every run
  satisfies the frozen conditions. Result: GO for `top2_mean` at score 0.475 /
  margin 0.005 (recall 0.9286 stable, zero false merges).
- `run_semantic_router_a5_holdout.py` is the corrected one-shot sealed holdout:
  frozen config + out-of-band manifest SHA authored by an agent that never read
  development texts. The A5 holdout passed 44/44 (recall 1.0, zero false merges)
  and is the current strategy-level GO evidence.
- `run_semantic_router_a5_replay_shadow.py` completed Phase A5.4 read-only integration
  with fail-closed behavior and zero invariant drift.
- Phase B local write-path integration is complete; production GO remains blocked on
  isolated VPS validation and the frozen PersonaMem before/after product gates.

Historical source snapshots under `sealed_sources/` are audit-only and are never
imported by current runners. Invalidation JSON files are authoritative when an old
artifact and a later reviewed artifact coexist. The A5 holdout result is the current
strategy-level GO evidence; `Meno_v2.0_Gate_Charter.md` records per-gate status.

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
Production benchmark reports must record the SiliconFlow model, output dimension,
projection version, batch sizes, API error/429 counts, and whether the run used
cached vectors. Never store the SiliconFlow API key or raw credential file in results.
