# Oracle + Extractor Eval Harness

Automated, repeatable validation of the visual-verification oracle
(`services/step_oracle.py` + `POST /session/{id}/verify`) and the Gemini-Pro
extractor — **without hand-reviewing every video**. Standalone scripts (no
pytest), matching the repo's probe idiom. Every script prints PASS/MARGINAL/FAIL
gates (80/50 thresholds) and writes JSONL to `evals/reports/`.

## Ground truth, for free

Labels are self-supervised from the expert clips already on disk: a step's
**last frame = "done"** (positive), its **first frame / the prior step's last
frame = "not done"** (negative). Drop real learner attempts into
`evals/fixtures/staged/{procedure_id}/` (named `step_{n}_{done,notdone,errorK}_*`)
for higher-fidelity labels.

## Run

```bash
cd main-local-server-test && source .venv/bin/activate

python evals/harvest_frames.py        # cut labeled frames from clips (do this first)
python evals/eval_oracle.py           # oracle confusion matrix + interject-FP (no server/Live)
python evals/eval_extractor.py        # structural lint + closed-loop consistency (DB procedures)
python evals/eval_extractor.py --reextract --judge   # re-run the LIVE prompt on test_data + LLM-judge
python evals/probe_oracle_endpoint.py --server http://localhost:8000   # endpoint state machine (server up)
python evals/probe_oracle_live.py     # Live [ADVANCE]/[CORRECTION] + silence regression (spends tokens)
python evals/run_all.py [--reextract --judge --server URL --live --limit N]   # aggregate report
```

## What each script proves

| Script | Question it answers |
|---|---|
| `eval_oracle.py` | Does the oracle fire "complete" only when actually done? Headline = **false-positive (early-advance) rate**, per grounding mode (image+text vs text-only) and per verifiability level. Plus interject false-positive rate on correct frames. |
| `eval_extractor.py` | Are criteria well-formed spatial end-states, correctly graded? **Closed-loop consistency**: a step graded `reliable` the oracle can't confirm on its own done-frame is a re-extract candidate (this is how legacy "what-changed" criteria surface). Reports criterion-coherence (text-only) *and* production-confirmability (image+text) separately. |
| `probe_oracle_endpoint.py` | Does the `/verify` router gate/dwell/advance correctly? (active_monitoring off → skip; negative → pending; positive → advanced; step increments.) |
| `probe_oracle_live.py` | Does Live speak on `[ADVANCE]`/`[CORRECTION]` and stay **silent when idle** (the heartbeat-hallucination regression)? |

## Notes

- `_common.py` runs `init_db()` on import (idempotent) so the `visual_verifiability`
  column exists even on a DB snapshot that predates the oracle migration; legacy
  rows backfill to `reliable`.
- Positives use the ~90% frame, never the exact reference JPEG, so the oracle
  can't win by matching identical bytes.
- Legacy procedures were extracted under the old "what-changed" framing — expect
  low consistency until re-extracted. `eval_extractor.py --reextract` validates
  the *new* prompt end-to-end.
