# Contributing

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[test]'
PYTHONPATH=. .venv/bin/pytest
.venv/bin/ruff check src/ tests/ benchmarks/
```

Tests use a deterministic embedding double and never call an embedding provider,
so the suite runs offline with no credentials.

## What this project optimizes for

Meno's value proposition is that injected memory is auditable, revocable, and
explainable — not that it scores well on a benchmark. Preserve user isolation,
consent, deletion, evidence lineage, and outage recovery. Use focused checks for
the behavior being changed; routine fixes do not need a new benchmark campaign.

Two conventions follow from that, and they are unusual enough to state plainly:

**A new evaluation lane must be able to fail.** A gate that passes by construction
is worse than no gate, because it reads as evidence. This project has shipped two
such lanes and had to retract both. If you add a lane, add a self-check that
injects a synthetic defect and requires the metric to catch it; if the metric
cannot fail on real data, say so in the report rather than letting the pass stand
unqualified. `benchmarks/run_grounding_temporal_suite.py` is the worked example.

**Report the denominator, not just the delta.** Four architecture changes here each
improved their own mechanism while the gate metric stayed flat, because the
mechanism's share of the whole was never reported. "Preference facets per question
+83%" without "and that is 40 of 480 injected facets" is not an interpretable
result. See Gate Charter §6.

## Pull requests

- Keep the diff scoped to the change. Don't reformat surrounding code or clean up
  unrelated dead code in the same PR.
- Add or update tests for behavior changes. Bug fixes should include a test that
  fails before the fix.
- New capabilities go behind a default-off feature flag, following the existing
  ones in `src/meno/config.py`.
- Run the relevant gate lanes if you touch retrieval, extraction, storage, or
  consent handling, and include the numbers. If a frozen baseline legitimately
  changes, re-freeze it and state why in the PR — an unexplained digest change is
  indistinguishable from a regression.
- If a change relaxes a safety or correctness check, say so explicitly in the
  description. Do not let it pass as an implementation detail.

## Security

Report vulnerabilities privately — see [SECURITY.md](SECURITY.md). Please do not
open a public issue for a disclosure bug.

## Data and privacy in contributions

- Never commit real user data, conversation logs, API keys, host addresses, or
  database files. Benchmark corpora are gitignored and are not redistributed.
- Test fixtures use synthetic data. If you need a realistic case, write one rather
  than pasting a real transcript.
- Keep result artifacts under the ignored `artifacts/` directory. They may contain
  rendered memory context and must not be added to source distributions.

## License

Contributions are accepted under the Apache License 2.0, the same terms as the
rest of the project.
