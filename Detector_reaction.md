# Detector reaction to code and tool output

Question: do the frozen Phase 3 detectors react to code, JSON, and tool output, as opposed
to the injection itself?

Short answer: both detectors react strongly to the tool-result wrapper and fire on it
whether or not an injection is present. Evidence for a reaction to code or JSON content
specifically is weak and confounded. The mean detector keeps a usable ranking inside tool
output; the logistic detector does not.

## Detectors

Both read the residual at the last prompt token of the Phase 1 selected layer, decomposed
into 25 nonnegative J-space token coefficients (screened pursuit, 512 candidates). Neither
has any feature for text format.

| detector | score | threshold |
| --- | --- | --- |
| mean | projection of the J-space reconstruction onto the Phase 1 attack-minus-clean direction | 0.0473 |
| logistic | L2 logistic regression (C=1, liblinear) on the sparse coefficients, keyed by token ID, fit on Phase 1 BIPIA train | -0.3937 |

Thresholds were chosen once on Phase 1 validation by task-macro balanced accuracy and are
frozen. Phase 4 and the grid below retune nothing.

## Metrics

A prediction is positive (flagged as an injection) when the score is at or above the
frozen threshold. TP (true positives) are attacked prompts flagged, FN (false negatives)
attacked prompts missed, FP (false positives) clean prompts flagged, TN (true negatives)
clean prompts not flagged.

- TPR, true positive rate = TP / (TP + FN). Of the attacked prompts, the share flagged.
- FPR, false positive rate = FP / (FP + TN). Of the clean prompts, the share wrongly
  flagged.
- ROC, receiver operating characteristic curve. Sweeps every threshold and plots TPR
  against FPR.
- AUROC, area under the receiver operating characteristic curve. Equal to the probability
  that a random attacked prompt scores higher than a random clean one. 0.5 is chance, 1.0
  is perfect separation, below 0.5 means clean prompts score higher. It ignores the frozen
  threshold, so it measures ranking only.

TPR and FPR describe the detector as deployed, at its threshold. AUROC describes whether
the score could separate the classes at some threshold. A detector with TPR and FPR both
1.0 but AUROC 0.9 ranks well and has a miscalibrated threshold; one with AUROC near 0.5
has no signal. Full definitions, with AUPRC and balanced accuracy, are in Definitions.md
section 3.

## Existing evidence

BIPIA code task, Phase 4 held-out test, at the frozen thresholds:

| task | mean FPR | logistic FPR | mean AUROC | logistic AUROC |
| --- | --- | --- | --- | --- |
| code | 0.24 | 0.54 | 0.87 | 0.77 |
| abstract | 0.08 | 0.40 | 0.82 | 0.89 |
| email | 0.00 | 0.04 | 0.83 | 0.86 |
| qa | 0.01 | 0.03 | 0.86 | 0.91 |
| table | 0.00 | 0.17 | 0.86 | 0.84 |

Clean code prompts trip both detectors far more often than other clean content.

AgentDojo, fixed Phase 4 run (Phase4Results.md), clean versus exposed attacked episodes:

| suite | mean AUROC | logistic AUROC | logistic FPR |
| --- | --- | --- | --- |
| banking | 0.60 | 0.30 | 1.00 |
| slack | 0.16 | 0.00 | 1.00 |
| travel | 0.39 | 0.12 | 1.00 |
| workspace | 0.71 | 0.40 | 1.00 |

The logistic detector flags every clean episode and scores clean tool-output episodes as
more injection-like than attacked ones. InjecAgent has no clean condition, so its high
TPR (0.93 to 0.98) cannot be read as detection.

The AgentDojo comparison has a timing confound. Clean episodes are scored at the first
tool result; attacked episodes at the first tool result that contains the injection, which
can be several turns later. Attacked prompts are therefore often longer and deeper in the
agent loop than clean ones.

## Controlled format grid

To remove that confound, a grid was scored on Gemma 4 12B-it with the frozen stack
(2026-11-12, Colab A100, about 5 minutes). Factors:

- format: prose, code (Python function with the note as a comment), JSON
- wrapper: inline (question then content in one user turn), or tool result (question,
  assistant tool call, then a user-role {"result": ...} message, as AgentDojo delivers it)
- condition: clean, or the same content with one of three injection strings appended to
  the note

60 base records, 720 prompts. Clean and attacked twins are identical apart from the
injection and are read at the same position. clean shift is the cell's mean clean score
minus the prose/inline mean clean score.

Mean detector:

| format | wrapper | clean shift | TPR | FPR | AUROC |
| --- | --- | --- | --- | --- | --- |
| prose | inline | 0.000 | 0.03 | 0.00 | 1.00 |
| code | inline | +0.004 | 0.48 | 0.00 | 0.93 |
| json | inline | +0.018 | 0.58 | 0.22 | 0.61 |
| prose | tool result | +0.035 | 1.00 | 1.00 | 0.75 |
| code | tool result | +0.044 | 1.00 | 1.00 | 0.82 |
| json | tool result | +0.050 | 1.00 | 1.00 | 0.92 |

Logistic detector:

| format | wrapper | clean shift | attack minus clean | FPR | AUROC |
| --- | --- | --- | --- | --- | --- |
| prose | inline | 0.00 | +0.41 | 1.00 | 0.67 |
| code | inline | -0.49 | +0.46 | 1.00 | 0.68 |
| json | inline | +0.13 | +0.34 | 1.00 | 0.71 |
| prose | tool result | +1.02 | +0.29 | 1.00 | 0.96 |
| code | tool result | +1.17 | -0.03 | 1.00 | 0.52 |
| json | tool result | +1.48 | +0.02 | 1.00 | 0.53 |

Reading:

- The tool-result wrapper shifts mean-detector clean scores by 0.035 to 0.05, enough to push
  every clean prompt over the 0.0473 threshold. This reproduces the AgentDojo false
  positives without the timing confound.
- Code alone barely moves the mean detector (+0.004). JSON moves it a little (+0.018) and
  costs discrimination (AUROC 0.61).
- Inside tool output the mean detector still ranks attacks above clean (AUROC 0.75 to
  0.92), so its failure there is mostly an offset. A context-specific threshold could
  recover part of it.
- The logistic detector fires on every prompt in every cell. It keeps an injection signal
  for prose inside tool output (AUROC 0.96) and loses it for code or JSON inside tool output
  (AUROC 0.52, 0.53).

## Limits of the grid

The grid shows that the tool-result setup shifts scores. It does not yet show which part
of it does.

1. The wrapper changes four things at once: an extra assistant turn with tool-call markup,
   the {"result": ...} envelope, JSON escaping, and the question moving away from the end
   of the prompt.
2. Code inside the tool result is JSON-escaped (\n and \"), so that cell tests escaped
   strings, not code-shaped tokens.
3. Only the last token is read. Inline prompts end on the content after the question; tool
   results end on the JSON blob. Part of the wrapper effect may be what sits nearest the
   readout.
4. Attacked twins are one sentence longer than clean ones. There is no length-matched
   benign control.
5. One record template, three injection strings, records drawn from 5x5x5 choices, no
   confidence intervals.
6. AgentDojo's native tool output is YAML inside the envelope. The grid uses JSON and
   Python-like text only.

## Next cells

| cell | separates |
| --- | --- |
| tool turn with raw text, no envelope | envelope and escaping from the tool turn |
| multi-turn with no tool call | turn structure from tool semantics |
| question repeated at the end in both wrappers | readout position |
| clean twin with a benign same-length sentence | length from injection |
| more templates and injections, AgentDojo-style YAML, per-record bootstrap CIs | generality and noise |
| tool-output span deleted and re-scored | whether the score depends on the content at all |

All are extra cells in the same script; a full run costs a few minutes of A100 time.

## Reproducing

Inputs, pulled from PxlNexus/jspace-checkpoints (jspace-full, --minimal): the Phase 1
handoff (phase1/selected_layer.json and the selected layer's decomposition), the Phase 3
detectors (phase3/mean_detector.pt, logistic_detector.pt), and the Jacobian lens. The grid
ran at repo revision c891f73 with format_grid.py and format_grid_cli.py copied in, and
configs/phase1_full.yaml.

Code path, in call order:

- phase4/format_grid_cli.py: argument parsing, phase4/config.py load_config.
- phase4/format_grid.py build_grid, render_content, wrap_messages: the prompts.
- phase1/artifacts.py load_phase1_handoff and phase4/detectors.py FrozenDetectors.load:
  frozen layer and detectors, with identity checks against Phase 1.
- model.py HuggingFaceModelAdapter.generate_with_capture: last-token residual at the
  selected layer (called with max_new_tokens=1).
- phase1/data.py render_ids: chat template.
- phase1/adapters.py JacobianLensAdapter and phase1/jspace.py build_normalized_dictionary:
  the J-space dictionary.
- phase4/detectors.py FrozenDetectors.score, which calls phase1/jspace.py
  screened_nonnegative_pursuit: both scores and frozen predictions.
- phase4/format_grid.py summarize: the tables above.

Command, on a CUDA machine with the inputs in place:

    python -m jspace_research.phase4.format_grid_cli \
      --config configs/phase1_full.yaml \
      --phase1 <run>/phase1/selected_layer.json --phase3 <run>/phase3 \
      --bipia-root <BIPIA>/benchmark --output-dir <out> --records 60 --seed 0

The existing-evidence tables come from phase4_predictions.parquet of the fixed Phase 4 run
via tools_local/tag_content_shape.py. The AgentDojo capture rule behind the timing
confound is phase4/agentdojo.py, _make_llm (the eligible condition) and _contains_injection.

The grid files and this document are not committed yet.

## Files

- src/jspace_research/phase4/format_grid.py, format_grid_cli.py: grid builder, scorer,
  summary. Resumable per row.
- tests/test_format_grid.py: CPU tests.
- tools_local/tag_content_shape.py: per-tag rates on existing Phase 4 predictions.
- tools_local/vm_colab_format_grid.py, launch_format_grid.py, results/format-grid/orchestrate.py:
  Colab runner, launcher, keep-alive and Hub watcher.
- results/format-grid/out/jspace-full/format_grid/: format_grid_summary.csv,
  format_grid_scores.parquet. Also on the Hub at PxlNexus/jspace-checkpoints under
  jspace-full/format_grid.
