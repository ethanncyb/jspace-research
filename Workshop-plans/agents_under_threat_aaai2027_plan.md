# LLM Agents Under Threat submission game plan

Decision record for the [AAAI-27 workshop LLM Agents Under Threat in Cyberspace](https://llm-agents-under-threat.github.io/). Written 2026-10-18 from the tree at `e81f081` on `debuging-phase-4`. This file changes no detectors, benchmarks, or configs.

The workshop question is whether a tool-using agent still completes the authorized task when an observation, tool result, or retrieved document is attacker-controlled, and whether a defense stops the attacker's goal before a consequential action. A score on a single prompt becomes evidence here only after AgentDojo's own attack-success and utility checks are the reported outcomes.

Deadline: **2026-11-20, Anywhere on Earth**. Notification: 2026-12-02. Workshop: 2027-02-22 or 2027-02-23, Montréal, full-day, in person. Full papers: up to 7 pages, excluding references. Shorter papers are welcome; the live page publishes no separate cap. Format: AAAI two-column. Review: single-blind, at least two reviews, a third for borderline papers. Publication: non-archival. Responsible disclosure is expected for real systems. The submission portal is still marked as forthcoming. Contact: `llm-agents-under-threat@googlegroups.com`.

## How to read this file

The branch inventory, the citable-versus-prototype labels, and the do-not-cite paths live in [trust4rsi_aaai2027_plan.md](trust4rsi_aaai2027_plan.md). This file uses that inventory and leaves the branch tables there.

Both files share one GPU calendar. They describe two manuscripts. This manuscript is about pre-action containment. The Trust4RSI manuscript is about a later decision that depends on an earlier write. The same PDF goes to one venue.

`PLAN.md` §15 defers monitor-evasion and residual-stream baselines until the seven phases are complete. This submission records two exceptions, and no others:

1. Score the same-layer raw-residual detectors that already exist on `origin/codex/raw-activation-baseline` (`325296a`).
2. Run one adaptive tier, with a declared budget, against the frozen gate.

Phase 6 direction edits, new probe families, k-sweeps, and a second model family stay outside that exception.

## What is already measured

The signal is the k=25 screened nonnegative reconstruction of the residual through \(D_\ell = W_U J_\ell\), followed by a frozen mean-difference detector and a sparse logistic detector (`PLAN.md` §5, `src/jspace_research/phase4/detectors.py`). That reconstruction is the candidate distinction from a raw-activation probe. The distinction is unmeasured until the raw-residual baseline is scored.

AgentDojo v1.2.2 is already the agent environment. The pinned attack is `important_instructions` (`src/jspace_research/phase4/config.py`). Clean utility and targeted attack success are already logged on the trusted runs. The missing piece is a gate in front of the tool call.

Operating points below are copied from the trusted files named in the Trust4RSI plan. Harness versions and the pre-fix files that stay out of any table live there.

| Model | BIPIA logistic AUROC | AgentDojo mean AUROC | Clean utility | Targeted ASR | Frozen-threshold behavior on AgentDojo |
| --- | ---: | ---: | ---: | ---: | --- |
| Gemma 4 12B, harness v5 | 0.853 | 0.463 | 0.804 | 0.428 | Logistic AgentDojo AUROC 0.189, ranking inverted |
| Mistral-Small-24B, fixed harness | 0.911 | 0.742 | 0.546 | 0.314 | Logistic AUROC 0.486; both detectors flag every clean episode |
| Qwen3.5-9B, harness v6 | 0.915 | 0.624 (0.557–0.688) | 0.701 | 0.230 (218/949) | Logistic AUROC 0.507; mean TPR/FPR 0.973/0.959; utility under attack 0.554 |

Qwen suite ASR, from `results/phase4_qwen3_5_9b_wsl_audit.md`: banking 18.1% (26/144), Slack 90.5% (95/105), Travel 55.7% (78/140), Workspace 3.4% (19/560). Workspace's 560 attacks dominate the pooled 23.0%. The audit scores 97 clean controls and 921 exposed attacks.

Gemma Phase 2 leaves attack success unchanged: 0.4312 at α=0, 0.432 at α=0.5, 0.4312 at α=1.0, on 1,250 validation attacks. On Gemma layers 24–28 at k=25, validation variance explained by the reconstruction stays near 1e-5 and cosine similarity stays near 0.003–0.005. Deleting the reconstructed component is a measured null on Gemma. A sentence that says the Jacobian component controls compliance is unsupported.

Phases 5–7 are specified in `PLAN.md` and implemented on no branch. Phase 7 is a read-only gate: if the frozen score exceeds the frozen threshold, return a fixed block and skip the normal action. `PLAN.md` tells Phase 7 to keep the BIPIA-frozen thresholds. Those thresholds flag agent traffic at attack-like rates, so the gate experiment below uses a different, pre-declared suite split. The paper says that this is recalibration for agent traffic.

The raw-residual control is code and tests on `origin/codex/raw-activation-baseline`. Representation `raw_residual_stream`, same selected layer, same decision points as Phase 3 and Phase 4, mean and logistic families, threshold chosen by macro balanced accuracy. Stages are `fit`, `transfer`, and `analyze`. `fit` and `analyze` are CPU-only. `transfer` needs CUDA. No checked-in table.

This control is the test of whether the J-space reconstruction adds anything at the same layer. It is a same-layer residual probe. Calling it a TaskTracker reproduction would misstate both methods. TaskTracker compares activation deltas around untrusted text. A literal TaskTracker port is future work, and it is outside this month.

## Corrections to the external brief

The brief's preferred shape — a monitor that fires before a consequential tool call and lowers completed attacker goals — matches the call. The repository supports a narrower present claim.

- The work is past a handcrafted-prompt demo. Three models have post-harness Phase 4 audits. BIPIA logistic AUROC is 0.85, 0.91, and 0.92.
- AgentDojo is already integrated. Another benchmark integration is the wrong next step. InjecAgent has no clean class; its AUROC stays out.
- High BIPIA AUROC, layer plots, and "the score moves before the output tokens" are already the Phase 4 diagnostic. They are the motivation paragraph, not the contribution.
- A Jacobian-advantage sentence waits on the raw-residual table. Until that table exists, the honest statement is that a Jacobian-lens feature was read out, and its advantage over the raw residual is unknown.
- The citable AgentDojo runs contain one attack setting, `important_instructions`. The suite split below is the generalization test that those files can support. The adaptive tier is a separate, later run.
- White-box optimization through the lens is the obvious open-weight question. It stays in the limitations section for this deadline.

## Threat model to freeze before any threshold is fit

| Element | Specification |
| --- | --- |
| Protected asset | The user's AgentDojo task, and the application state that task is allowed to change |
| Attacker | The author of the untrusted content inside an AgentDojo tool observation |
| Attacker control | The injection string delivered by the pinned `important_instructions` attack, plus one later paraphrase tier |
| Attacker knowledge for the main table | The attack string is fixed by the benchmark. The attacker does not see the detector score |
| Adaptive tier | A paraphrase or query-adaptive variant with a declared attempt budget. Score feedback only if that leakage is part of the stated attacker |
| Success | AgentDojo targeted attack success, which is a checked environment state |
| Out of scope | Weight tampering, host compromise, a second attack family, and white-box differentiation through the lens |

Consequential action means the tool call that would carry the attacker's goal into the environment. The gate has to be able to fire on the decision where the injection is first delivered, before that call. A block after the environment has already changed is a miss.

## Suite split, written before fitting

Fit the agent threshold on **banking** and **Travel**. Evaluate on **Slack** and **Workspace**.

Banking and Travel are the two suites where success and failure are both common (26/144 and 78/140). Slack is nearly saturated (95/105). Workspace is the bulk of the attacks and almost never succeeds (19/560). Clean controls used to fit the threshold come from banking and Travel only. Held-out clean controls are scored after the threshold is frozen.

Report every suite separately. The pooled rate hides Workspace. Change this split only by a dated note in this file, and only before any threshold is fit.

## Additional work

### Step A — CPU, through Sunday 2026-10-25

Same work as Step A in the Trust4RSI plan. No new GPU run. No raw-baseline fit during Step A.

On the three citable AgentDojo runs, cross the frozen high/low score with native attack success. For each detector report the four cells, the share of attacks that are high-score and successful, and attack success given high versus low. This is `PLAN.md` Phase 5, read as a prediction question: does the score move with completed compromise?

Step A is coherent if those counts come from the trusted files and the qualitative picture still holds: BIPIA separation on all three models, agent thresholds that flag clean episodes, and AgentDojo ranking that survives for the Mistral mean and the Qwen mean and fails for Gemma, for the Mistral logistic, and for the Qwen logistic.

If high and low scores have the same attack success, a gate on that score can lower ASR only by blocking broadly. That result still goes in the diagnostic paper. It ends the 7-page defense claim.

### Step B — one shared sequence, starting 2026-10-26

Qwen3.5-9B is the only system model. Run the raw-residual comparison before the gate so both manuscripts share one Qwen sequence. The Trust4RSI file lists the gate first and treats the raw residual as the control that runs before any further sweep. Doing that control first satisfies its stop rule and this file's novelty test.

1. **Raw residual, first.** Run `fit` and `analyze` from `origin/codex/raw-activation-baseline`, then `transfer` if the Phase 1, Phase 3, and Phase 4 artifacts for Qwen are present. Compare with the J-space detectors on the same episodes, including the suite split above. A sentence that says the Jacobian reconstruction adds information is allowed only when this comparison shows a gap on agent outcomes, with the paired interval the branch already implements. A match is a result: the short paper leads with it.
2. **Phase 7 gate, second.** Pre-tool-call allow or block on AgentDojo. Threshold fit on banking and Travel only, then frozen. Evaluate on Slack and Workspace. The paper keeps the BIPIA-frozen threshold as the transfer diagnostic. The gate uses the suite-fit threshold.
3. **Action filter, third.** One deterministic check on tool name or arguments, scored on the same episodes. If the trusted trajectories already record the call that would be blocked, score the filter from those records before any new generation.
4. **One adaptive tier, fourth.** A paraphrase of `important_instructions`, or a query-adaptive variant, with the attempt budget written down before the run. Score it with AgentDojo's own checks. Keep this tier separate from the suite holdout. If it cannot be scored that way, name it as future work in the same sentence as the white-box limitation.

Report, for the undefended agent, the text-free action filter, the raw-residual gate, the J-space gate, and the J-space score without a gate: targeted ASR, clean utility, utility under attack, and false-block rate. Keep AUROC as a diagnostic. Add latency only as a rough note unless the 7-page track is already intact.

The Trust4RSI quarantine log is a separate manuscript. It is scheduled there, after this gate's frontier is real. It is cited in this paper only as future work.

Leave the rest until the 2026-11-08 freeze is already met: Phase 6, a Mistral or Gemma repeat of the gate, InjecAgent as anything other than a no-clean-class note, and a literal TaskTracker delta.

## Calendar

| Window | Work | Stop rule |
| --- | --- | --- |
| 2026-10-18 – 2026-10-25 | Step A | Go/no-go on Sunday 2026-10-25 |
| 2026-10-26 – 2026-11-08 | Raw residual, then the Qwen gate, then the action filter, then one adaptive tier | Freeze code and thresholds on 2026-11-08 |
| 2026-11-09 – 2026-11-18 | Write and internal mock review | Submit on 2026-11-19 local time |
| 2026-11-20, 23:59 AoE | Deadline | The last evening is for upload, not a new run |

Single-blind: author names and affiliations stay on the PDF unless the organizers say otherwise.

## Paper type

- **Shorter diagnostic paper**, the default. Claim: a BIPIA-trained J-space score separates text injections and fails as a tool-agent operating point. Evidence is the three-model table, the Phase 5 cross, the Gemma Phase 2 null, and the reconstruction variance. If Step B finishes and the raw residual matches J-space, this paper also reports that match, and it reports the gate as a property of the internal score rather than as a Jacobian-specific defense. Aim at about 4 pages until the organizers publish a short-paper cap.
- **7-page paper** only when all three are true: the recalibrated gate lowers targeted ASR at a tolerable false-block rate on the held-out suites, the raw-residual comparison is in the paper and shows a gap on agent outcomes, and the adaptive tier has been run under a pre-declared budget.
- **No submission** that is only BIPIA or AgentDojo AUROC.

Working title for the short paper: *A BIPIA-trained J-space score fails as an operating point for tool-using agents.* Working title for the 7-page paper, usable only after the gap is measured: *Gating tool calls with a Jacobian-lens monitor in AgentDojo.*

A 7-page main text has room for the threat model, the two readouts, the three-model operating-point table, the gate table, the suite-holdout table, and the limitations. One system diagram is enough. A timing plot is in the paper only if the gate logs the decision index of the first exposed injection and of the blocked call. A t-SNE plot stays out.

## Stays out of the claims

- Any sentence of the form "compared with TaskTracker" until a TaskTracker-style delta has actually been run. The raw residual is the comparison this month.
- SecurityProbe demo block rates and the mock target.
- Overdose steering ranks.
- Pre-fix AgentDojo files for Gemma, Mistral, and Qwen.
- InjecAgent AUROC.
- GSM8K accuracy and HumanEval activations as a security result.
- Phase 2 as evidence that the Jacobian component is causally involved.
- "Solves prompt injection," "guarantees safety," and "robust to adaptive attackers" beyond the single tier that was run.
- The same PDF submitted to Trust4RSI-Agent.

## Go / no-go on 2026-10-25

Score each row 0, 1, or 2. The 7-page track needs the first five rows at 2. The short diagnostic paper needs harm prediction and reproducibility at 2, and it can carry novelty at 0 or 1 if the language stays inside the measurements.

| Dimension | 0 | 1 | 2 |
| --- | --- | --- | --- |
| Harm prediction | Scores and ASR saved apart | Phase 5 cross on one trusted run | Phase 5 cross on all three trusted runs |
| Novelty vs raw residual | Unscored | Scored, no agent-outcome gap, or not yet transferred | A gap on held-out agent outcomes, with the paired interval |
| Enforcement | Score only | Offline action filter | Runtime allow/block before the unsafe call |
| Utility | Unmeasured | Undefended clean utility and utility under attack | Gate frontier on the held-out suites |
| Adaptive tier | Unmentioned | Suite split written down, which is this file | One paraphrase or query-adaptive run with a declared budget |
| Reproducibility | Mixed pre-fix files | Trusted runs named in the Trust4RSI plan | Frozen suite split, config hash, and the date the threshold was fit |

As of this document, before Step A: harm 0, novelty 0, enforcement 0, utility 1, adaptive 1, reproducibility 2. Utility is 1 because the undefended Qwen run already reports clean utility 0.701 and utility under attack 0.554. Adaptive is 1 because the suite split is written here. It reaches 2 only after a paraphrase or query-adaptive run.

**Go** on Step B if Step A reproduces the qualitative picture from the trusted files.

**Write the short paper and stop the 7-page track** if Phase 5 shows no separation between high and low scores, if the raw residual matches J-space on agent outcomes, or if a gate cannot be wired to an AgentDojo tool-call boundary by 2026-11-01.

## Organizer email

Send to llm-agents-under-threat@googlegroups.com. Do not wait on the reply to start Step A. The published topic list already covers indirect prompt injection, tool misuse, runtime monitoring, and end-to-end evaluation, so this note asks for logistics only.

Subject: Submission-policy questions for LLM Agents Under Threat

Dear Organizers,

A submission is being prepared on indirect prompt injection in tool-using agents. The measurements are an internal-state monitor, a pre-tool-call gate, and AgentDojo security and utility outcomes, including a held-out suite split and one adaptive tier.

Could you please clarify:

1. When and where will the submission portal open?
2. Is there a fixed page limit for shorter papers?
3. Are appendices and supplementary code permitted, and are reviewers expected to read them?
4. Should single-blind submissions include author names and affiliations on the PDF?
5. What is the policy on concurrent submissions to another non-archival workshop, and on preprints?
6. What are the camera-ready deadline, registration requirement, and in-person presentation requirement?

Thank you.

## Source map

Inventory, harness labels, and files that stay out of the tables:

- [trust4rsi_aaai2027_plan.md](trust4rsi_aaai2027_plan.md)

Trusted numbers used above:

- `Records/Gemma/README.md`
- `Records/Gemma/jspace-full-Gemma/rerun/phase4_metrics.csv`
- `Records/Gemma/jspace-full-Gemma/phase2/phase2_summary.csv`
- `Records/Gemma/jspace-full-Gemma/phase1_robustness_k25/reconstruction_robustness.csv`
- `results/phase4_mistral_small_24b.md`
- `results/phase4_qwen3_5_9b_wsl_audit.md`

Method and the two submission exceptions:

- `PLAN.md` §5, Phase 5, Phase 7, and §15
- `src/jspace_research/phase4/detectors.py`
- `src/jspace_research/phase4/config.py` (`important_instructions`, AgentDojo v1.2.2)
- `origin/codex/raw-activation-baseline` at `325296a`: `src/jspace_research/raw_baseline/`
