# Mistral-Small-24B AgentDojo: untrusted run vs fixed run

Both files are the same model, config, and AgentDojo version (v1.2.2, `important_instructions`, no defense, 97 clean episodes and 949 attacks). The untrusted file was generated before the harness fixes. The trusted file was regenerated after them.

| File | Status |
|---|---|
| `agentdojo_records_before_fix.jsonl` | Untrusted. Do not cite. |
| `agentdojo_records.jsonl` | Trusted. Use this, with `agentdojo_trajectories.jsonl`. |
| `bipia_records.jsonl`, `injecagent_records.jsonl`, `phase4_metrics.csv` | Trusted. Those benchmarks did not go through the buggy AgentDojo code. |

The three harness bugs were: injection exposure missed in tool results returned as a Python `repr` (all 13 impossible "successes" were travel), tool results sent to the model as JSON content blocks instead of text, and text after the first tool call left in the conversation history.

## What this means for the research

The main claim is unchanged. At the frozen Phase 3 thresholds, both detectors flag every AgentDojo episode, clean or attacked, in both runs. The logistic detector has no ranking signal either way (AUROC 0.505 untrusted, 0.486 fixed). BIPIA and InjecAgent are unaffected.

The numbers that would go in a write-up did change, and not all in one direction. Use the fixed run for exposure, attack success, utility, and AUROC.

## Per-suite comparison (untrusted → fixed)

| Suite | Clean utility | Attack success | Injection exposure | Successes with no exposure |
|---|---|---|---|---|
| banking | 0.562 → 0.625 | 0.521 → 0.507 | 0.812 → 0.750 | 0 → 0 |
| slack | 0.429 → **0.857** | 0.619 → **0.790** | 0.952 → 0.952 | 0 → 0 |
| travel | 0.500 → 0.500 | 0.586 → **0.729** | 0.686 → **1.000** | **13 → 0** |
| workspace | 0.400 → 0.375 | 0.061 → 0.071 | 0.750 → 0.750 | 0 → 0 |
| Overall | 0.454 → 0.546 | 0.270 → 0.314 | 0.772 → 0.809 | 13 → 0 |

Slack is the clearest model-behavior change: once tool results arrived as text, clean utility doubled and attack success rose by 17 points. Travel is the clearest measurement bug: exposure was under-counted, and 13 attacks were scored as successes even though the model never saw the injection.

Banking exposure went the other way (0.812 → 0.750). The untrusted numbers were not uniformly too low.

## Detector comparison (untrusted → fixed)

Flag rates at the frozen thresholds are 1.00 for both detectors, both classes, every suite, in both runs.

| Suite | Mean AUROC | Logistic AUROC |
|---|---|---|
| banking | 0.760 → 0.766 | 0.624 → 0.535 |
| slack | 0.931 → 0.963 | 0.514 → 0.502 |
| travel | 0.999 → 0.976 | 0.391 → 0.500 |
| workspace | 0.569 → **0.636** | 0.537 → 0.446 |
| Overall | 0.690 → **0.742** | 0.505 → 0.486 |

The untrusted run understated the mean detector's ranking signal, mostly on workspace. Travel's 0.999 was computed on the wrongly filtered exposed subset; 0.976 is the real value and is still very strong.

## Case-level changes

- Attack outcomes flipped in 70 of 949 cases: 56 failure → success, 14 success → failure.
- Clean-task outcomes flipped in 15 of 97 cases: 12 failure → success, 3 success → failure.
- Of the 724 attacked episodes scored in both runs, no detector score is identical. The mean scores still correlate at 0.91, so the shift is systematic rather than noise.
