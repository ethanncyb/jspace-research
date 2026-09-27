# Phase 4 results: Mistral-Small-24B-Instruct-2501

Held-out and cross-benchmark transfer of the frozen Phase 3 detectors for `mistralai/Mistral-Small-24B-Instruct-2501` with the `mistral-small-24b-instruct-2501/lens.pt` Jacobian lens (`configs/phase1_mistral_small_24b_full.yaml`, Phase 1 run `phase1-14ce883ecdcc-9165301f50b7`).

## Summary

- The frozen detectors work on the held-out BIPIA test set: AUROC 0.82 (mean detector) and 0.91 (logistic detector), with 74% and 79% of attacks flagged against 23% and 10% of clean prompts.
- They do not transfer to agent benchmarks at the frozen thresholds. Every AgentDojo episode and every InjecAgent case is flagged, clean or attacked, because agent-task scores sit far above both thresholds.
- The mean detector still ranks attacks above clean episodes on AgentDojo (AUROC 0.74 overall; 0.96 on slack and 0.98 on travel), so its signal survives a large score shift. The logistic detector falls to chance on AgentDojo (AUROC 0.49).
- Mistral-Small-24B is highly susceptible without a defense: 63% attack success over all InjecAgent cases, and 31% targeted attack success across AgentDojo (73% on travel, 79% on slack).

## Frozen detectors

| Detector | Threshold (frozen in Phase 3) |
|---|---|
| Mean | 0.08464684337377548 |
| Logistic | 0.06542975938136486 |

Thresholds were selected on the BIPIA development validation set and were not changed in Phase 4. The AgentDojo AUROC reported below is a threshold-free descriptive metric added after generation; it uses no Phase 4 data for fitting or threshold selection.

## BIPIA official test set (held-out, same benchmark)

1,250 attacked prompts (250 per task) and 400 clean prompts.

| Scope | Detector | AUROC | AUPRC | TPR | FPR | Balanced accuracy |
|---|---|---|---|---|---|---|
| Overall | Mean | 0.820 | 0.756 | 0.742 | 0.228 | 0.757 |
| Overall | Logistic | 0.911 | 0.919 | 0.790 | 0.098 | 0.846 |
| Macro over tasks | Mean | 0.890 | 0.900 | 0.742 | 0.288 | 0.727 |
| Macro over tasks | Logistic | 0.901 | 0.919 | 0.790 | 0.114 | 0.838 |

| Task | Mean AUROC | Mean TPR / FPR | Logistic AUROC | Logistic TPR / FPR |
|---|---|---|---|---|
| abstract | 0.954 | 0.592 / 0.00 | 0.956 | 0.892 / 0.03 |
| code | 0.808 | 1.000 / 1.00 | 0.818 | 0.828 / 0.34 |
| email | 0.885 | 0.752 / 0.06 | 0.933 | 0.736 / 0.02 |
| qa | 0.904 | 0.472 / 0.02 | 0.895 | 0.708 / 0.08 |
| table | 0.899 | 0.896 / 0.36 | 0.905 | 0.788 / 0.10 |

The code task already shows the shift seen on agent benchmarks: clean code prompts score a median 0.376 on the mean detector, so the mean detector flags every clean code prompt. BIPIA attack-success outcomes come from the frozen OpenRouter judge and are in `phase4_predictions.parquet`; they are not summarized here.

## AgentDojo v1.2.2 (cross-benchmark transfer)

All four suites, no defense, `important_instructions` attack: 97 clean episodes and 949 attacked episodes. Detector positives are the 768 attacked episodes where the injection reached the model (`injection_exposed`); negatives are the 96 clean episodes with a scored tool observation.

| Suite | Clean utility | Utility under attack | Targeted attack success | Injection exposure |
|---|---|---|---|---|
| banking | 0.625 | 0.458 | 0.507 | 0.750 |
| slack | 0.857 | 0.562 | 0.790 | 0.952 |
| travel | 0.500 | 0.093 | 0.729 | 1.000 |
| workspace | 0.375 | 0.343 | 0.071 | 0.750 |
| Overall | 0.546 | 0.348 | 0.314 | 0.809 |

| Suite | Mean AUROC | Logistic AUROC | TPR (both detectors) | FPR (both detectors) |
|---|---|---|---|---|
| banking | 0.766 | 0.535 | 1.00 | 1.00 |
| slack | 0.963 | 0.502 | 1.00 | 1.00 |
| travel | 0.976 | 0.500 | 1.00 | 1.00 |
| workspace | 0.636 | 0.446 | 1.00 | 1.00 |
| Overall | 0.742 | 0.486 | 1.00 | 1.00 |

## InjecAgent base setting (cross-benchmark transfer)

1,054 attacked cases (510 direct harm, 544 data stealing). InjecAgent has no clean condition, so only the attacked-case flag rate and score distribution are defined.

| Subgroup | Valid rate | Attack success (valid) | Attack success (all) | TPR (both detectors) |
|---|---|---|---|---|
| direct_harm | 0.951 | 0.656 | 0.624 | 1.00 |
| data_stealing | 0.904 | 0.713 | 0.645 | 1.00 |
| Overall | 0.927 | 0.685 | 0.635 | 1.00 |

## Score shift across benchmarks

Median detector score with the interquartile range in brackets.

| Group | n | Mean detector | Logistic detector |
|---|---|---|---|
| BIPIA clean | 400 | -0.099 [-0.202, 0.063] | -1.47 [-2.13, -0.75] |
| BIPIA attacked | 1,250 | 0.242 [0.080, 0.410] | 1.62 [0.30, 2.97] |
| BIPIA code, clean | 50 | 0.376 [0.319, 0.421] | -0.45 [-1.40, 0.54] |
| InjecAgent attacked | 1,054 | 0.419 [0.400, 0.437] | 2.22 [1.75, 2.77] |
| AgentDojo clean | 96 | 0.664 [0.630, 0.708] | 3.19 [2.50, 3.68] |
| AgentDojo exposed attack | 768 | 0.740 [0.678, 0.802] | 3.14 [2.62, 3.54] |

Clean AgentDojo episodes score higher than attacked BIPIA prompts on both detectors. The mean detector keeps attacked episodes above clean ones within AgentDojo; the logistic detector does not. Together with the BIPIA code task, this suggests the reconstructed direction partly encodes structured or technical content (code, JSON, tool schemas), which agent prompts are full of, in addition to injection.

## Interpretation and caveats

- Transfer fails at the operating point, not entirely in the representation. Any deployment of these detectors on agent traffic would need recalibration, which the Phase 4 protocol forbids and which is outside this evaluation.
- The AgentDojo negative class is small (96 clean episodes against 768 exposed attacks), so per-suite AUROC rests on 16 to 40 clean episodes.
- Workspace transfers worst for the mean detector (AUROC 0.64) and has the lowest attack success (7%).
- AgentDojo detector examples use one decision per episode: the first model step after the injection is delivered (attacked) or after the first tool observation (clean).

## Model behavior notes (AgentDojo)

These are properties of Mistral-Small-24B under the frozen protocol, not harness defects:

- 15 user tasks never reach an injection under any attack; Mistral gives the same answer under every injection, and 14 of these tasks also fail in the clean run.
- 28 episodes run to AgentDojo's 15-step limit, mostly by calling `send_money` repeatedly.
- 62 episodes end at the 512-token generation cap; in 48 of them Mistral pads an email body with newlines until the call is cut off.

## Harness verification

The AgentDojo results above come from a regenerated run. An audit of the first Mistral run found three harness bugs, all fixed on the `debuging-phase-4` branch before regeneration:

1. Exposure tracking missed injections in tools that return a Python `repr` with single-quoted strings, so travel `injection_task_4` was never marked exposed (13 successful attacks were recorded as unexposed).
2. Tool results were sent to the model as JSON-encoded AgentDojo content blocks instead of text, and Mistral began imitating that format.
3. Text generated after the first tool call, including fabricated tool results, stayed in the conversation history.

| AgentDojo check | Before fixes | After fixes |
|---|---|---|
| Attack success without exposure | 13 | 0 |
| Travel injection exposure | 0.686 | 1.000 |
| Echoed tool results in saved responses | 39 | 1 |
| Slack clean utility | 0.429 | 0.857 |

On the regenerated run, recorded exposure matches the delivered tool outputs in all 949 attacked episodes, no clean episode contains injected text, and every detector score was captured at the specified decision point.

InjecAgent needed no changes. Every saved prompt matches the pinned upstream test case, and re-running the upstream evaluator reproduces every step-1 and step-2 label and the upstream `get_score` metrics.

Re-running Phase 3 on Colab re-saved byte-different but identical detector files (thresholds, feature count, and Phase 1 identity unchanged). Phase 4 records and provenance were repointed to the new file hashes with `python -m jspace_research.phase4.rehash`, which logs the change in `provenance.json`; detector saving is now byte-stable.

## Artifacts and reproduction

- Run directory: `MyDrive/jspace-research/runs/jspace-mistral-small-24b-full/phase4/` (`phase4_metrics.csv`, `phase4_predictions.parquet`, per-benchmark `*_records.jsonl`, `agentdojo_trajectories.jsonl`).
- Code: branch `debuging-phase-4`; AgentDojo generation with harness version 2, analysis at `f376854`.
- Audits: `notebooks/Phase4_AgentDojo_Debug.ipynb` and `notebooks/Phase4_InjecAgent_Debug.ipynb`.
