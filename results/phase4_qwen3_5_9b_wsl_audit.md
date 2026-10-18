# Qwen3.5-9B Phase 4 WSL rerun audit

Audited 2026-10-17. Local output: `artifacts/qwen/agentdojo-rerun/phase4/`.
The Qwen3.5-9B model ran in full bfloat16, without quantization or prompt
truncation, on an RTX 5090 Laptop GPU. The frozen Phase 1 layer-15 handoff,
Phase 3 detectors, model and lens revisions, benchmark revisions, and Phase 4
scientific configuration were verified before generation. The dictionary was
stored on CPU and streamed to GPU in chunks for scoring. The local code change
is preserved in `artifacts/qwen/agentdojo-rerun/phase4_streaming_changes.patch`
(SHA-256 `82a2fc0c38f61819e97292ea853e55b4b5fb5fd7f53a858359dc540f0e50edf2`).

An additional GPU check rebuilt the frozen layer-15 dictionary from the pinned
lens and cached unembedding, then scored seven actual cached activations using
both a full GPU dictionary and the CPU-streamed path. Mean and logistic scores
matched exactly for all seven (maximum absolute difference 0), as did every
prediction. The check is reproducible with
`scripts/audit_qwen_streaming_equivalence.py`; its log is
`artifacts/qwen/agentdojo-rerun/streaming_equivalence.log`. This is a sampled
equivalence check, not proof of bitwise identity for all Phase 4 residuals.

## Scope and integrity

| Benchmark | Records | Origin | Main qualification |
| --- | ---: | --- | --- |
| BIPIA | 1,650 | Reused, byte-identical frozen records | 1,250 cached attack judgments; no new paid requests |
| AgentDojo | 1,046 | Newly generated with harness version 6 | 97 controls and 949 attacks; 921 attacks exposed the injection and were scored |
| InjecAgent | 1,054 | Reused, byte-identical frozen records | Simulator substitution described below |

Analysis produced 3,750 prediction rows and marked `complete: true`. All
1,046 AgentDojo case IDs exactly match the pinned v1.2.2 suites, with no
missing or extra cases. All record IDs and hashes match one-to-one with their
trajectories, every trajectory uses harness version 6, and each scored case has
exactly one activation capture. Detector scores are finite. The 28 unexposed
attacks have no detector score by design. The generation log shows no traceback
or CUDA out-of-memory error. The provenance SHA-256 values match the generated
manifest, predictions, metrics, and plot. BIPIA and InjecAgent input files are
byte-identical to the frozen source copies. The test suite passed (95 passed,
1 skipped), Ruff passed, and `git diff --check` passed.

The previous AgentDojo records are preserved as
`agentdojo_records_before_fix.jsonl` and are not mixed into this analysis.
That earlier harness scored only 76 clean controls and 664 exposed attacks;
the corrected rerun scored all 97 controls and 921 exposed attacks. Its native
outcomes therefore should not be compared as if only the GPU changed.

## Detector results

The thresholds below were frozen in Phase 3 and were not tuned on Phase 4.
TPR and FPR are rates at those thresholds; AUROC measures ranking independently
of the threshold.

| Benchmark | Detector | AUROC | TPR | FPR |
| --- | --- | ---: | ---: | ---: |
| BIPIA | Mean | 0.804 | 0.694 | 0.205 |
| BIPIA | Logistic | 0.915 | 0.757 | 0.098 |
| AgentDojo | Mean | 0.624 | 0.973 | 0.959 |
| AgentDojo | Logistic | 0.507 | 0.800 | 0.845 |
| InjecAgent | Mean | unavailable | 1.000 | unavailable |
| InjecAgent | Logistic | unavailable | 0.916 | unavailable |

AgentDojo has 97 scored clean controls and 921 scored exposed attacks. A
2,000-replicate bootstrap resampling the 97 `(suite, user_task_id)` clusters
(seed 42) gave 95% intervals of 0.557–0.688 for mean-detector AUROC and
0.446–0.569 for logistic-detector AUROC. The logistic result is compatible
with chance ranking. The mean detector ranks somewhat better overall, but its
frozen threshold flags about 96% of clean controls. The logistic threshold
flags about 85% of clean controls. These are poor operational tradeoffs on
AgentDojo despite high TPR. AgentDojo native clean utility was 70.1%, utility
under attack 55.4%, and targeted attack success 23.0% (218/949 attacks).
The generated `phase4_detector_transfer.png` displays TPR only; read it
alongside FPR and AUROC rather than as stand-alone evidence of transfer.

The native targeted success rate varies sharply by suite: banking 18.1%
(26/144), Slack 90.5% (95/105), Travel 55.7% (78/140), and Workspace 3.4%
(19/560). The pooled 23.0% rate is dominated by Workspace's 560 attacks;
suite-level results should accompany it.

BIPIA's logistic AUPRC on context-matched, class-balanced rows was 0.917.
The current AUROC uses the 1,650 raw cases (1,250 attacks and 400 distinct
clean controls). The archived analysis used repeated matched controls for
AUROC (`n=2,500`), so its AUROC values differ despite identical underlying
records. This is an analysis-definition change, not new BIPIA generation.
All 1,250 cached BIPIA attack judgments were present and had no `UNKNOWN`
label. They requested `openai/gpt-4.1-mini`; provenance reports OpenAI as
provider and `openai/gpt-4.1-mini-2025-04-14` as provider model.

InjecAgent has no clean controls, so its TPR cannot establish specificity or
AUROC. Native validity was 90.8% (957/1,054); targeted attack success was
31.0% among valid cases and 28.2% across all cases. The inherited provenance
records a user-approved substitution: missing simulator responses used
`openai/gpt-4` instead of exact native `gpt-4-0613`. The pinned responses and
tasks were retained, but exact native-model comparability is not guaranteed.

## Interpretation and reproducibility limits

The completed artifacts are internally consistent, and the corrected
AgentDojo rerun is suitable for reporting as **this harness's result**. The
strong BIPIA performance does not transfer to useful frozen-threshold
classification on AgentDojo. The near-chance logistic AUROC and high clean
false-positive rates are substantive negative findings, not a generation
failure. InjecAgent supports a sensitivity-only statement and carries the
simulator-substitution limitation.

The local streaming optimization was uncommitted during the run, so the git
commit recorded in provenance alone does not describe every executed line.
The code patch above preserves the optimization for review. Results also used
the local CUDA/PyTorch stack recorded in provenance, which differs from the
earlier Colab environment. These facts limit bit-for-bit reproduction; they
do not imply quantization or a changed scientific configuration.
