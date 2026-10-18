# Trust4RSI-Agent submission game plan

Decision record for the [AAAI 2027 Workshop on Trustworthy Evolution of AI Agents](https://trust4rsi-agent.github.io/AAAI2027/). Written 2026-10-18 from the tree at `e81f081` on `debuging-phase-4`. This file changes no detectors, benchmarks, or configs.

The workshop question is whether an agent's memory, tools, policy, or shared state can change over time while a substrate keeps that change trustworthy. A detector that scores a single prompt is relevant only after it sits inside a boundary the runtime enforces, and after some later decision depends on an earlier update.

Deadline: **2026-11-20, 11:59 p.m. Anywhere on Earth** (OpenReview renders this as 2026-11-21, 05:00 Pacific). Notification: 2026-12-02. Workshop: 2027-02-22 or 2027-02-23, Montréal; the one-day placement is still TBA on the supplied site. Paper types on that site: full paper, 8 pages; short or position paper, 4 pages. English, one PDF, AAAI 2027 author kit, double-blind.

## How to read this inventory

Every local and remote tip is listed once. A remote that matches a local tip is named on that local line. Three labels are used throughout:

- **Citable.** A post-harness measurement with a written audit of what was wrong in earlier files.
- **Prototype.** Code and sometimes a demo. The demo is not a benchmark result.
- **Superseded or infrastructure.** Kept so a later draft does not resurrect it as a separate experiment.

## Phase 1–4 detection pipeline

This line is the frozen protocol in `PLAN.md`: select a J-lens layer, remove the reconstructed J-space component, freeze a mean-difference detector and a sparse logistic detector, then score BIPIA official test, AgentDojo v1.2.2, and InjecAgent base. Phases 5–7 (recognition versus compliance, direction-specific intervention, read-only gating) are specified in `PLAN.md` and have no implementation on any branch.

| Branch | Tip | Date | What it is | Numbers | Workshop track |
| --- | --- | --- | --- | --- | --- |
| `debuging-phase-4`, `origin/debuging-phase-4` | `e81f081` | 2026-10-18 | Current scientific line. Gemma 4 12B, Mistral-Small-24B, and Qwen3.5-9B through Phase 4, plus the Qwen harness-v6 audit. | Citable Phase 4 audits below. | Tool-use evaluation. Memory and policy evolution are absent. |
| `main` | `a6163d2` | 2026-09-05 | Pipeline through the Mistral Colab launcher. | Behind the current branch on the Qwen AgentDojo rerun. | Same detection protocol. |
| `origin/main`, `origin` | `fcbfbdd` | 2026-10-01 | `main` plus the Phase 1 robustness plan from `origin/codex/phase1-robustness`. | No Qwen rerun. | Detection, plus a reconstruction check. |
| `origin/remaining-phases` | `7601966` | 2026-09-02 | Notebook pointer and phase scaffolding absorbed into the pipeline. | None separate. | Infrastructure. |
| `origin/codex/phase4-progress` | `05da299` | 2026-09-02 | Resumable Phase 4 progress reporting. | None separate. | Infrastructure. |
| `origin/codex/resumable-provenance` | `48d7d68` | 2026-09-02 | Provenance that survives a resume across code revisions. | None separate. | Audit trail, already merged. |
| `prompt-injection-experiment` | `9a16ad9` | 2026-08-31 | Earlier Phase 1–4 checkout. | Superseded. | — |
| `origin/prompt-injection-experiment` | `52e9372` | 2026-09-05 | Two commits ahead of the local branch (`2a0a01e`, `52e9372`). Same protocol. | Superseded. | — |
| `origin/gemma4_agentdojo_injecagent_fixed` | `c891f73` | 2026-10-04 | AgentDojo harness fixes, plus deferral of InjecAgent cases whose simulator response cannot be generated. | Ancestor of the trusted reruns, not a second result set. | Explains which AgentDojo files are citable. |
| `origin/codex/phase1-robustness` | `3da7e8c` | 2026-10-01 | Post-hoc comparison of screened greedy pursuit and gradient pursuit at k=25. | Gemma CSV is on the current branch: `Records/Gemma/jspace-full-Gemma/phase1_robustness_k25/reconstruction_robustness.csv`. | Limits any claim that the reconstructed component is the residual. |
| `origin/codex/raw-activation-baseline` | `325296a` | 2026-10-03 | Same-layer raw-residual mean and logistic detectors. Fit on saved Phase 1 activations, scored at frozen Phase 4 decision points, compared with a paired J-space-minus-raw bootstrap. `fit` and `analyze` are CPU-only. `transfer` needs CUDA. | Code and tests. No checked-in raw-baseline table. | The control for "the signal is in J-space." |

The robustness CSV covers Gemma layers 24–28 only. At k=25, macro AUROC on the validation split runs from 0.71 to 0.90, while validation variance explained stays on the order of 1e-5, cosine similarity stays near 0.003–0.005, and support overlap between the two pursuit methods stays under 0.06. The direction ranks injections. The sparse reconstruction is a tiny piece of the residual. That belongs in the limitations section of any paper that talks about removing J-space.

## Intervention and sweep line

| Branch | Tip | Date | What it is | Numbers | Workshop track |
| --- | --- | --- | --- | --- | --- |
| `origin/prompt-injection-experiment-L4` | `82280d1` | 2026-10-12 | Configurable sparsity K and output-token window W. Phase 2 edits generated tokens rather than the prompt token. Also multi-layer direction ablation, top-token analysis, and Qwen3.5-9B launchers for an L4 host. | `PHASE1_PHASE2_SWEEP_PLAN.md` on that branch says the sweep code and CPU tests exist and the CUDA sweep was not run. | Would speak to model-update mechanism. It does not yet. |
| `origin/prompt-injection-experiment-qwen` | `f300fe7` | 2026-08-31 | Early Qwen3.5-4B launcher and BIPIA train-file validation. | Superseded by later Qwen configs on `debuging-phase-4`. | — |

## Capability and lens line

These branches measure clean task behavior or expose the lens. They are context for utility and for how the instrument works. They are not a security evaluation.

| Branch | Tip | Date | What it is | Numbers | Workshop track |
| --- | --- | --- | --- | --- | --- |
| `Qwen` | `3d4ac1c` | 2026-08-14 | Jacobian-lens library and `jspace_steer.ipynb`. | Steering demo below. | Mechanism demo. |
| `origin/Qwen` | `2402a66` | 2026-08-18 | Two commits ahead of local `Qwen` (`625bcdc`, `2402a66`): GSM8K accuracy benchmark and a Qwen POC analysis, on top of the same steering demo. | Capability runs, if present on that branch, stay out of the security claim. | Clean-task context. |
| `origin/qwen-gsm8k` | `7e3aceb` | 2026-08-21 | `qwen-gsm8k-jlens/`: exact-answer GSM8K under a no-op hook and a `mean_replace` intervention, for Qwen3.5-4B and 9B. | Accuracy study. | Utility context. |
| `origin/qwen-basebenchmark` | `3f8d866` | 2026-08-29 | Adds command docs and smoke configs for BIPIA and AgentDojo inside `qwen-gsm8k-jlens`. | Smoke adapters. | Setup for later Phase 4, already replaced. |
| `qwen-HumanEval`, `origin/qwen-HumanEval` | `7f341f9` | 2026-08-29 | HumanEval activation dumps under `qwen-humaneval-jlens/outputs-remote-5090/`. | Activation archive. | Utility context. |
| `codex/qwen-model-size-test`, `origin/codex/qwen-model-size-test` | `274d2c2` | 2026-08-17 | Model-size plumbing on the Qwen lens stack. | No security table. | Infrastructure. |
| `origin/Gemma` | `8625a31` | 2026-08-03 | Initial commit. | Historical. | — |
| `origin/Gemma-lens-web` | `4d235c4` | 2026-08-19 | Interactive layer-by-position lens panel (`scripts/lens_panel.py`). | None. | Infrastructure. |

Checked-in steering demo, `Qwen:results/steering_demo_qwen3.5-4b.txt` (2026-08-05): on one overdose prompt, J-space steering at strength 0.2 moves the target token `emergency` from rank 60253 to rank 2 at the final prompt token. A same-norm random control moves it the other way. This shows the lens can move a next-token rank on that prompt. It is not an attack-success result, and Gemma Phase 2 (below) is the attack-success measurement of a J-space edit.

## Systems prototypes

These two lines are the closest existing match to the workshop. Neither has been run on AgentDojo or on a real model outcome for prompt injection.

| Branch | Tip | Date | What it is | Numbers | Workshop track |
| --- | --- | --- | --- | --- | --- |
| `Qwen-Security-Probe` | `a8f6f56` | 2026-08-10 | Append-only JSONL incident memory (`jlens/guard_memory.py`) and an allow / flag / block probe (`jlens/guard_probe.py`). Learning means appending an incident. A later user can match an earlier embedding. | CPU mock, `results/security_probe_demo.txt`. | Memory evolution: write, retrieve, transfer across users. |
| `feature/jacobian-drift-detector`, `origin/feature/jacobian-drift-detector` | `6ee6ec3` | 2026-08-17 | `promptguard/` first cut: drift probe, hooks, evolving attacker and guard. | Tests. No `outputs/` directory. | Policy evolution, observability. |
| `feature/strong-attacker-eval`, `origin/feature/strong-attacker-eval` | `25c5ed9` | 2026-08-18 | Same package plus semantic-preservation checks, stronger baseline attacks, and the drift-detector update. Configured for Qwen3.5-4B, 10 evolution rounds, seed 7, circuit-breaker or hard-stop intervention, replay of previous rounds, and a held-out malicious list that is never used for guard training. | Tests in `tests/test_promptguard.py` and `tests/test_promptguard_benchmarks.py`. No checked-in evolution output. | Policy co-evolution and an adaptive attacker. |

SecurityProbe demo, treated as a design sketch: 10 intents, 8 strategies, 10 benign lookalikes; thresholds 0.85 block, 0.75 repeat-offender, 0.6 flag. Block rate for the repeating user moves 0.20 → 0.79 → 0.87 → 0.87 across four rounds. A fresh user is blocked on 8 of 10 round-1 attacks through the any-user threshold. All 10 benign lookalikes pass. The target is a mock that "answers" when the strategy id is listed as effective. Those rates stay out of the paper.

## Citable detection numbers

Only post-harness Phase 4 runs are detection evidence. Thresholds were frozen on BIPIA development data and were not retuned on AgentDojo or InjecAgent.

### Gemma 4 12B, harness v5

Source: `Records/Gemma/README.md` and `Records/Gemma/jspace-full-Gemma/rerun/phase4_metrics.csv`. The first run and the middle rerun are instrumentation (unparsed tool calls, then a thinking-channel loop). Do not cite them.

| | BIPIA AUROC | AgentDojo AUROC | AgentDojo clean utility | Targeted ASR | Injection seen |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mean detector | 0.783 | 0.463 | 0.804 | 0.428 | 0.993 |
| Logistic detector | 0.853 | 0.189 |  |  |  |

Logistic AgentDojo ranking is inverted: clean episodes score higher than attacked ones. Workspace is the only suite where the mean detector ranks attacks above clean (AUROC 0.671).

### Mistral-Small-24B, fixed harness

Source: `results/phase4_mistral_small_24b.md` and `Records/Mistral/`. The pre-fix AgentDojo file recorded 13 successes in which the injection was never delivered. Do not cite it.

| | BIPIA AUROC | AgentDojo AUROC | Frozen-threshold flag rate | Clean utility | Targeted ASR |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mean detector | 0.820 | 0.742 | 1.00 attack and 1.00 clean | 0.546 | 0.314 |
| Logistic detector | 0.911 | 0.486 | 1.00 attack and 1.00 clean |  |  |

The mean detector still ranks attacks above clean episodes inside AgentDojo. Both operating points flag every episode, because agent prompts sit far above the BIPIA thresholds. The write-up already notes that code, JSON, and tool schemas land in the same score region as injections. Median mean-detector score: BIPIA clean −0.10, BIPIA attacked 0.24, AgentDojo clean 0.66, AgentDojo exposed attack 0.74.

### Qwen3.5-9B, harness v6

Source: `results/phase4_qwen3_5_9b_wsl_audit.md` (audited 2026-10-17). The rerun artifacts are under gitignored `artifacts/qwen/agentdojo-rerun/`. `Records/Qwen/phase4_metrics.csv` is the pre-fix run (76 scored clean controls, 664 exposed attacks, targeted ASR 0.095). Do not cite that CSV for AgentDojo outcomes.

| | BIPIA AUROC | AgentDojo AUROC (95% bootstrap) | TPR / FPR at frozen threshold | Clean utility | Targeted ASR |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mean detector | 0.804 | 0.624 (0.557–0.688) | 0.973 / 0.959 | 0.701 | 0.230 (218/949) |
| Logistic detector | 0.915 | 0.507 (0.446–0.569) | 0.800 / 0.845 |  |  |

Suite ASR is not the pooled 23%: banking 18%, Slack 90%, travel 56%, workspace 3%. Workspace's 560 attacks dominate the pool. A streaming dictionary path was checked against a full GPU dictionary on seven cached activations (maximum absolute score difference 0). That is a sample check. Provenance records a commit that does not include the uncommitted streaming patch; the patch hash is in the audit.

InjecAgent, on all three models, has no clean class. Report attack-flag rate and native ASR. Do not report InjecAgent AUROC. Qwen InjecAgent also substituted `openai/gpt-4` for missing `gpt-4-0613` simulator responses.

### Gemma Phase 2 null

Source: `Records/Gemma/jspace-full-Gemma/phase2/phase2_summary.csv`. Removing the entire reconstructed J-space component at the selected layer leaves attack success unchanged: 0.4312 at α=0, 0.432 at α=0.5, 0.4312 at α=1.0, on 1,250 validation attacks. Clean ROUGE-L barely moves. Phase 2 is a coarse disruption, and on Gemma it does not change attack behavior. The overdose steering demo does not override this measurement.

## Combined claim

State this only as the conjunction of the inventory:

1. A BIPIA-trained J-space direction is readable on held-out BIPIA for Gemma 4 12B, Mistral-Small-24B, and Qwen3.5-9B (logistic AUROC 0.85, 0.91, and 0.92).
2. The threshold frozen on BIPIA is not an operating point for tool-using agents. Gemma's AgentDojo ranking is at or below chance. Mistral's mean detector still ranks (AUROC 0.74) and flags every clean episode. Qwen's mean detector ranks weakly (AUROC 0.62) and flags 96% of clean controls.
3. On Gemma, deleting the reconstructed J-space component does not reduce attack success.
4. An append-only incident memory and an attacker/guard co-evolution loop exist as prototypes on other branches. They have not been evaluated on AgentDojo or on real model compliance.

The paper can claim (1)–(3) from records already on disk. Claim (4) becomes evidence only after Step B.

## What is still missing for the workshop

Scored against the workshop's own ingredients, the repository today has a strong evolving-failure story only in the weak sense that a BIPIA operating point fails on agent traffic. It does not yet have an enforceable mechanism or a later decision that depends on an earlier write.

| Ingredient | Present now | Where it would come from |
| --- | --- | --- |
| Evolving object | Single-episode AgentDojo trajectories. No persistent store. | SecurityProbe incident log, or a second step that reads an earlier block. |
| Longitudinal failure | Indirect injection inside one episode. Score shift across benchmarks. | A blocked or quarantined observation that changes a later tool call. |
| Enforceable mechanism | Detectors return scores. Nothing blocks a call. | Phase 7 gate, using the promptguard hard-stop as the action and the SecurityProbe log as the write. |
| Evidence and limits | Three-model transfer, harness audits, Phase 2 null, tiny reconstruction variance. | Safety–utility frontier, false interventions, and an explicit irreversibility statement. |

A submission that stops at AUROC answers a representation question. The workshop asks for the substrate.

## Additional work

### Step A — CPU, this week, across lines

No new GPU run. Finish by **Sunday, 2026-10-25**. This is the go/no-go.

1. **Phase 5 split** on the three citable AgentDojo runs (Gemma harness v5, Mistral fixed, Qwen harness v6). Cross the frozen high/low score with native attack success or resistance. Report the share of attacks that are high-score and successful, and attack success given high versus low. Use records already saved. This shows whether the signal is present on attacks that succeed.
2. **Raw-baseline note.** One page: what `origin/codex/raw-activation-baseline` would compute (same layer, same mean and logistic families, paired J-space-minus-raw intervals), which artifacts it needs, and that no table exists yet. Do not fit it during Step A.
3. **Prototype note.** One page listing what `promptguard` and `Qwen-Security-Probe` already implement (incident append, cosine retrieval, replay, hard stop, a held-out attacker list) and what they have never been run on (AgentDojo, a real model outcome, a J-space score).

Step A is coherent if the Phase 5 counts are computable from the trusted files and the three models agree on the qualitative picture: BIPIA separation, agent-threshold failure, and a mix of ranking that survives (Mistral mean, Qwen mean) and ranking that does not (Gemma both, Mistral logistic, Qwen logistic).

### Step B — one systems study, starting 2026-10-26

Put a frozen J-space score in front of a real action boundary. Reuse the prototypes for the boundary and the log. Do not port their demo rates.

**Default boundary.** Pre-tool-call allow/block on AgentDojo. Fit the threshold on two suites. Freeze it. Evaluate the other two. State in the paper that this is recalibration for agent traffic. It is a different protocol from the frozen Phase 4 threshold.

**Baselines, all on the same episodes.**

- No defense (the citable native run, if the harness matches; otherwise a paired rerun).
- A prompt reminder that untrusted tool output is data.
- The J-space gate.
- A random score matched to the same flag rate.

**Report.** Native targeted attack success, clean utility, utility under attack, false-block rate, and whether the gate fires on the decision where the injection is first delivered, before the unsafe tool call. Keep detector AUROC as a diagnostic.

**Evolution piece, if the gate stands up quickly.** Append each block or quarantine to a JSONL log in the SecurityProbe shape. On a later step of the same episode, or on one repeated session, the runtime consults that log before the next consequential call. The measured object is a later decision that depends on an earlier write. That is the memory-evolution claim, and it is small on purpose.

**Adaptive case.** `promptguard`'s evolving attacker supplies one held-out round only if that round can be run against the frozen gate and scored with AgentDojo's own checks. If it cannot, the paper names the adaptive attacker as future work. The mock evolution curve stays out.

**Model.** Qwen3.5-9B is the primary system model: the harness-v6 audit is the newest trusted agent run, and both prototypes are already aimed at Qwen. Repeat the gate on Mistral-Small-24B if the suite-held-out result is stable and time remains. Gemma's AgentDojo ranking is at chance, so Gemma is the negative control (a gate at a BIPIA threshold, or a recalibrated threshold that still fails to rank), not the system to build first.

### Optional controls, in this order

1. **Raw-residual baseline** on one citable model, from `origin/codex/raw-activation-baseline`. Answers whether J-space beats the raw residual at the same layer. Run this before any sweep if a control fits in the month.
2. **L4 K×W sweep.** Answers whether editing generated tokens, or another sparsity, changes the Phase 2 null. It does not answer the evolution question. Skip it if Step B is unfinished on 2026-11-08.

### Calendar

| Window | Work | Stop rule |
| --- | --- | --- |
| 2026-10-18 – 2026-10-25 | Step A | Go/no-go on Sunday 2026-10-25 |
| 2026-10-26 – 2026-11-08 | Step B gate, then the quarantine log if the gate's frontier is real | Freeze code and thresholds on 2026-11-08 |
| 2026-11-09 – 2026-11-18 | Write, internal mock review, anonymize | Submit on 2026-11-19 local time |
| 2026-11-20, 23:59 AoE | Deadline | Do not use the last evening for a new run |

### Paper type

- **4-page short paper** if Step B shows a safety–utility frontier and at least one later decision that depends on an earlier block or quarantine.
- **4-page negative-results paper** if the gate does not reduce attack success at a tolerable false-block rate. The evidence is the three-model shift, the Phase 2 null, the tiny reconstruction variance, and a precise statement that the memory and co-evolution prototypes were not yet evidence.
- **8-page full paper** only if both the gate and the delayed effect are measured, with the suite holdout frozen before the final run.
- **Do not submit** a detection-only AUROC paper to this workshop.

Working title for the short paper: *An internal injection score is not yet a reference monitor: gating tool calls in AgentDojo.* Working title for the negative paper: *J-space injection directions transfer across models and fail as an agent operating point.*

### Stays out of the claims

- SecurityProbe demo block rates and the mock target.
- Overdose steering ranks.
- Pre-fix AgentDojo files for Gemma, Mistral, and Qwen.
- InjecAgent AUROC.
- GSM8K accuracy and HumanEval activations as a security result.
- A statement that rollback undoes an email already sent, a credential already disclosed, or any other external side effect.
- "Solves prompt injection," "guarantees safety," "long-horizon" for a single prompt, and "agent operating system" for a score plus a threshold.

## Go / no-go on 2026-10-25

Score each row 0, 1, or 2 after Step A. Step B's design assumes the detection rows are already at least 1.

| Dimension | 0 | 1 | 2 |
| --- | --- | --- | --- |
| Evolution | Static prompts only | Multi-step episode, no store | A later decision reads an earlier write |
| Failure | Generic model error | Agent attack in one episode | Delayed or cross-session effect |
| Enforcement | Score only | Prompt reminder | Runtime allow / block / quarantine |
| Evidence | One model, one file | Three citable models, known harness limits | Held-out suites plus a matched random gate |
| Utility | Unmeasured | Native clean utility of the undefended agent | Safety–utility frontier of the gate |
| Cost | Unmeasured | A rough timing note | Latency and intervention rate |
| Recovery | Unmentioned | Qualitative irreversibility paragraph | Measured repair plus an external side effect that remains |
| Reproducibility | Mixed pre-fix files | Trusted runs named in this document | Frozen suite split, config hash, and the streaming-patch hash |

Before Step B, expect roughly: evolution 1, failure 1, enforcement 0, evidence 2, utility 1, cost 0, recovery 0, reproducibility 2. That is about 7. The negative-results paper is already defensible from that score. The short systems paper requires Step B to move enforcement to 2 and evolution to 2.

**Go** on Step B if Step A reproduces the qualitative claim from the trusted files and the suite split is written down before any threshold is fit.

**Stop and write the negative paper** if the trusted files disagree with the audits, or if a gate cannot be wired to an existing AgentDojo tool-call boundary by 2026-11-01.

## Organizer email

Send to trust4rsi.agent@gmail.com. Do not wait on the reply to start Step A.

Subject: Trust4RSI-Agent submission-policy and scope questions

Dear Trust4RSI-Agent organizers,

A submission is being prepared on an internal-state monitor for indirect prompt injection in tool-using agents. The planned system uses the monitor as a runtime gate before consequential tool calls, records blocked observations in an append-only log, and evaluates whether a later decision depends on that log. Clean utility and intervention overhead are part of the measurement.

Could you please clarify:

1. Are references and appendices excluded from the 8-page and 4-page limits, and is there an appendix limit?
2. Is the workshop archival, and what are the policies on concurrent submissions, previously published work, and preprints?
3. Are supplementary code and anonymous repository links allowed, and is the AAAI reproducibility checklist required?
4. Will accepted papers and reviews become public on OpenReview?
5. Is in-person presentation required for acceptance?
6. Does a monitor-plus-runtime framing sit inside the tool/skill and memory-evolution scope when model weights stay fixed?

Thank you.

## Source map

Trusted files for Step A:

- `Records/Gemma/README.md`
- `Records/Gemma/jspace-full-Gemma/rerun/phase4_metrics.csv`
- `Records/Gemma/jspace-full-Gemma/phase2/phase2_summary.csv`
- `Records/Gemma/jspace-full-Gemma/phase1_robustness_k25/reconstruction_robustness.csv`
- `results/phase4_mistral_small_24b.md`
- `Records/Mistral/README.md`
- `results/phase4_qwen3_5_9b_wsl_audit.md`

Untrusted for AgentDojo outcomes, kept only as the before-fix record:

- Gemma paths named as old in `Records/Gemma/README.md`
- `Records/Mistral/agentdojo_records_before_fix.jsonl` (see that README)
- `Records/Qwen/phase4_metrics.csv`

Prototype entry points, for the Step A notes and for Step B design:

- `Qwen-Security-Probe`: `jlens/guard_memory.py`, `jlens/guard_probe.py`, `scripts/security_probe_demo.py`
- `feature/strong-attacker-eval`: `promptguard/evolution_loop.py`, `promptguard/evolving_guard.py`, `promptguard/intervention.py`, `config.yaml`
- `origin/codex/raw-activation-baseline`: `src/jspace_research/raw_baseline/`
