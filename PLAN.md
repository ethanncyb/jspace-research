# Focused Experimental Plan: J-Space Monitoring for Prompt Injection

## 1. Research Question

> **Does prompt injection produce a reproducible representation in an LLM's J-space, and can that representation be used to detect, causally influence, or block prompt-injection behavior while preserving clean-task utility?**

The experiment is intentionally narrow.

It follows one sequence:

\[
\boxed{\text{Locate the signal}}
\rightarrow
\boxed{\text{Test whether it generalizes}}
\rightarrow
\boxed{\text{Relate it to behavior}}
\rightarrow
\boxed{\text{Manipulate it}}
\rightarrow
\boxed{\text{Use it as a monitor}}
\]

The project should not expand into unrelated interpretability, reasoning, reward-modeling, or general safety experiments.

---

# 2. Core Experimental Principle

The initial POC searched for a small hand-selected set of words such as `injection`, `malicious`, or `attack`.

The new experiment instead uses the model's **sparse J-space representation**.

At layer \(\ell\), let the residual-stream activation at the decision point be:

\[
h_\ell
\]

Approximate the J-space component as a sparse nonnegative combination of J-lens directions:

\[
h_\ell^J
=
\sum_{j \in S} c_j v_j,
\qquad
c_j \ge 0,
\qquad
|S|\le k
\]

where:

- \(v_j\) is a J-lens direction;
- \(c_j\) is its coefficient;
- \(k\) is the sparsity limit.

Use:

\[
k=25
\]

as the primary setting.

The remaining reconstruction residual is:

\[
h_\ell^R=h_\ell-h_\ell^J
\]

Version 1 may use a scalable screened nonnegative sparse-pursuit approximation for \(h_\ell^J\). The implementation must document the approximation clearly and should not describe it as an exact orthogonal projection.

Decoded vocabulary concepts and token ranks are **interpretability diagnostics only**. They do not define the detector.

---

# 3. Models and J-Lenses

The experiment should be configuration-driven so the same code can run with any causal language model that has a compatible fitted J-lens.

## Primary model

- Model: `google/gemma-4-12B-it`
- Revision: `5926caa4ec0cac5cbfadaf4077420520de1d5205`
- Precision: BF16

## Primary fitted lens

- Repository: `solarkyle/jspace-lenses`
- Revision: `1d95a2fc8a5c5a26c75a8c01c145173353e5fb65`
- File: `gemma-4-12b-it/lens.pt`
- SHA-256: `214ba70486c648d97cccb3c88d05cfb17adf9467c93b5d1f268fc4902e360048`

Do **not** fit a new J-lens as part of the core experiment.

The released fitted lens is treated as a fixed measurement instrument.

## Replication model

A second instruction-tuned model with a compatible fitted J-lens may be run through the same experimental pipeline after the primary-model implementation is stable.

The methodology must not change between models.

---

# 4. Hardware and Execution

The same experiment code should run on:

- Google Colab GPU;
- remote Linux CUDA hosts;
- local machines for small smoke tests.

Use GPU for:

- model inference;
- residual-stream capture;
- J-space reconstruction;
- activation interventions.

Use CPU for:

- AUPRC/AUROC;
- simple statistical analysis;
- plotting;
- lightweight detector logic.

The experiment should be implemented as normal Python code with a thin Colab launcher, not as Colab-only logic.

---

# 5. Benchmarks

## Development benchmark

**BIPIA**

Use BIPIA for:

- layer selection;
- learning the clean-to-injection direction;
- validation;
- selecting the detector threshold;
- selecting intervention strength.

Use all five BIPIA tasks:

- EmailQA
- WebQA
- TableQA
- Summarization
- CodeQA

Use BIPIA training data for development and leave the official BIPIA test split untouched until the frozen detector is evaluated.

For WebQA and Summarization, use researcher-provided BIPIA-format source files reconstructed according to the benchmark instructions.

## Transfer benchmarks

After BIPIA development is complete:

- AgentDojo
- InjecAgent

No retraining or threshold adjustment is allowed on these transfer benchmarks.

---

# 6. Decision Point

Read J-space after the model has processed the untrusted content and before it acts on that content.

For BIPIA:

> capture the residual-stream activation at the final non-padding prompt token immediately before response generation.

For agent benchmarks:

> capture the residual-stream state immediately after the untrusted tool result and before the next model action.

---

# Phase 1 — Layer Selection

## Goal

> **Which J-lens layer contains the most reproducible prompt-injection signal in sparse J-space?**

This phase is intentionally self-contained.

Its pipeline is:

\[
\boxed{
\text{BIPIA pairs}
\rightarrow
h_\ell^J
\rightarrow
d_\ell
\rightarrow
\text{validation scores}
\rightarrow
\text{macro-AUPRC by layer}
\rightarrow
\ell^*
}
\]

## 1.1 BIPIA train/validation pairs

Use only BIPIA training contexts and training attacks.

With seed `42`:

- create 500 training pairs per task;
- create 250 validation pairs per task;
- keep source contexts disjoint between train and validation;
- keep exact attack prompt variants disjoint between train and validation where possible;
- balance attack categories and insertion positions.

Each pair contains:

1. an attacked prompt;
2. a matched benign control using the same task, context, question, formatting, and insertion position, with the attack replaced by unrelated benign content of approximately matched token length.

Freeze and save the pair manifest before activation extraction.

## 1.2 Extract J-space at every eligible layer

For every prompt and every fitted J-lens layer:

1. capture \(h_\ell\);
2. reconstruct \(h_\ell^J\).

A practical Version 1 reconstruction may use:

- full-dictionary correlation screening;
- top 512 positive candidates;
- nonnegative sparse pursuit;
- maximum sparsity \(k=25\).

Cache activations/decompositions so interrupted runs can resume.

## 1.3 Learn one clean-to-injection direction per layer

Using the combined, task-balanced training set:

\[
d_\ell
=
\mu_{\ell,\text{attack}}^J
-
\mu_{\ell,\text{clean}}^J
\]

Store both the raw direction and its norm.

For scoring, use the normalized direction when the norm is nonzero:

\[
\hat d_\ell=
\frac{d_\ell}{\|d_\ell\|}
\]

There is **one shared direction per layer**, not one direction per task.

## 1.4 Score held-out validation examples

For validation example \(x\):

\[
s_\ell(x)
=
\hat d_\ell^\top
\left(
h_\ell^J(x)
-
\mu_{\ell,\text{clean}}^J
\right)
\]

Higher scores mean the example lies farther in the learned clean-to-injection direction.

The goal is not for every attack example to point exactly along \(d_\ell\). The goal is for the attack and matched-control score distributions to separate.

## 1.5 Measure detection quality

For each layer:

- calculate AUPRC separately for each BIPIA task;
- calculate AUROC separately for each BIPIA task;
- calculate macro-AUPRC across the five tasks.

\[
\operatorname{MacroAUPRC}(\ell)
=
\frac{1}{5}
\sum_{t=1}^{5}
\operatorname{AUPRC}_{\ell,t}
\]

Select:

\[
\ell^*
=
\arg\max_\ell
\operatorname{MacroAUPRC}(\ell)
\]

Freeze one primary layer for the rest of the experiment.

## Phase 1 output

Save:

- selected layer \(\ell^*\);
- selected-layer clean mean;
- selected-layer raw and normalized mean-difference directions;
- layer-wise per-task AUPRC/AUROC;
- macro-AUPRC curve;
- validation scores;
- pair manifest;
- model/lens/configuration provenance.

---

# Phase 2 — Freeze the Injection Detector

## Goal

> **Can the J-space direction selected during development serve as a reproducible injection detector?**

Use the frozen layer:

\[
\ell^*
\]

and the frozen direction learned in Phase 1:

\[
d_{\text{inj}} = d_{\ell^*}
\]

The primary detector score is:

\[
s(x)
=
\hat d_{\text{inj}}^\top
\left(
h_{\ell^*}^J(x)
-
\mu_{\text{clean}}^J
\right)
\]

Select a binary threshold:

\[
\tau
\]

using BIPIA validation data only.

After selecting \(\tau\), freeze:

- layer;
- direction;
- clean mean;
- threshold;
- J-space decomposition settings.

Do not add a more complex detector unless the simple mean-direction detector is clearly inadequate.

---

# Phase 3 — Held-Out and Cross-Benchmark Transfer

## Goal

> **Does the BIPIA-learned J-space signal transfer without retraining?**

Evaluate the frozen detector on:

1. BIPIA official test;
2. AgentDojo;
3. InjecAgent.

Do not retrain the detector or adjust \(\tau\).

For benchmarks with both attack and clean examples, report:

- AUPRC;
- AUROC;
- detection rate at the frozen threshold.

If a benchmark does not provide an appropriate clean condition, report:

- attack score distribution;
- true-positive rate at the BIPIA-frozen threshold.

This phase evaluates the **internal J-space signal**.

It does not yet require a causal claim.

---

# Phase 4 — Recognition vs Compliance

## Goal

> **Can a model have a high injection-related J-space score and still follow the attack?**

For every attacked example, record:

- frozen J-space detector score;
- detector decision;
- actual model output/action;
- benchmark attack-success result.

Analyze:

| Internal signal | Behavior | Interpretation |
|---|---|---|
| Low | Resists | resistance without measured J-space signal |
| High | Resists | high injection signal + resistance |
| Low | Follows | detector failure |
| High | Follows | recognition-compliance gap |

The important result is whether successful attacks occur despite a strong internal injection-related signal.

Use cautious language:

> "Injection-related information is readable from J-space."

Do not claim subjective awareness or consciousness.

---

# Phase 5 — Injection-Specific Causal Intervention

## Goal

> **Does changing the learned injection-associated J-space direction change attack behavior?**

At the frozen layer, intervene along the learned direction:

\[
h'_{\ell^*}
=
h_{\ell^*}
+
\alpha \hat d_{\text{inj}}
\]

and:

\[
h'_{\ell^*}
=
h_{\ell^*}
-
\alpha \hat d_{\text{inj}}
\]

Sweep a small set of positive and negative intervention strengths chosen on development data.

Do not assume beforehand which sign should improve safety.

## Required controls

Keep controls minimal:

1. no intervention;
2. same-norm random direction.

The purpose of the random control is to distinguish an injection-direction-specific effect from generic activation disruption.

## Measure

- Attack Success Rate;
- clean benchmark utility;
- malformed/refusal rate if relevant.

A useful causal result requires the injection-direction intervention to change attack behavior more than the matched random perturbation.

---

# Phase 6 — Read-Only Gate and Utility

## Goal

> **Can the frozen J-space detector be used as a practical pre-action monitor?**

Before the model acts:

\[
s(x)
=
\text{J-space injection score}
\]

If:

\[
s(x)>\tau
\]

route the example to a safe action such as block/refuse.

The gate does **not** modify the model's internal state.

Evaluate:

- attack block rate;
- Attack Success Rate after gating;
- false block rate on clean examples;
- benchmark-native clean-task utility.

The primary result is the tradeoff:

\[
\boxed{
\text{attack blocking}
\quad\text{vs}\quad
\text{clean utility}
}
\]

Primary utility evaluation should use the same benchmarks' clean/no-attack tasks.

Broad capability benchmarks such as GSM8K, MATH, or HumanEval are optional extensions and are not required for the core experiment.

---

# 7. Minimal Baseline

To determine whether J-space provides anything beyond an ordinary hidden-state signal, include one simple baseline after the primary experiment works:

> **Mean-difference detector on the full residual-stream activation \(h_{\ell^*}\) at the same frozen layer.**

Use the same:

- BIPIA train/validation split;
- scoring method;
- threshold-selection procedure;
- held-out benchmarks.

This baseline answers:

> Is the J-space representation competitive with simply reading the full hidden state?

Do not add a large collection of probe architectures unless needed.

---

# 8. Interpretability Diagnostics

Decoded J-lens concepts are useful for understanding the learned direction, but they are not the detector.

For selected examples, optionally report:

- top decoded J-space concepts;
- sparse coefficients;
- clean vs attack token ranks;
- percentage rank improvement.

For a token rank changing from \(r_c\) to \(r_a\):

\[
\text{Rank Improvement}
=
\frac{r_c-r_a}{r_c}\times100
\]

These diagnostics help explain the learned signal but do not determine whether an example is classified as an attack.

---

# 9. Experimental Outputs

The complete experiment should produce:

## Phase 1

- layer-wise macro-AUPRC curve;
- per-task AUPRC/AUROC;
- selected-layer artifact.

## Phase 3

- BIPIA-test detection metrics;
- AgentDojo transfer metrics;
- InjecAgent transfer metrics.

## Phase 4

- recognition-vs-compliance table/plot.

## Phase 5

- intervention strength vs Attack Success Rate;
- random-direction control.

## Phase 6

- attack-block-rate vs clean-utility curve.

## Baseline

- J-space detector vs full-residual detector at the same frozen layer.

---

# 10. Experimental Order

Implement incrementally.

```text
1. Layer selection
   BIPIA → h_l^J → d_l → validation scores → macro-AUPRC → l*

2. Freeze detector
   l* + d_inj + validation threshold

3. Held-out / transfer
   BIPIA test → AgentDojo → InjecAgent

4. Recognition vs compliance
   internal score + actual attack outcome

5. Causal intervention
   ± d_inj vs matched random direction

6. Read-only gate
   threshold detector before model action

7. Utility
   clean benchmark performance

8. Minimal baseline
   full residual-stream mean-difference detector
```

Do not build later phases until the earlier phase works.

---

# 11. Core Hypotheses

## H1 — Localization

Prompt injection is distinguishable from matched benign content using J-space representations at one or more fitted layers.

## H2 — Generalization

The frozen BIPIA-learned J-space detector transfers to held-out BIPIA data and external prompt-injection benchmarks without retraining.

## H3 — Recognition/compliance dissociation

Some successful attacks occur despite high frozen J-space injection scores.

## H4 — Causality

Manipulating the learned J-space injection direction changes Attack Success Rate more than a same-norm random-direction intervention.

## H5 — Practical utility

A read-only J-space gate reduces attack success while retaining useful clean-task performance.

---

# 12. What the Experiment Does Not Claim

The experiment does not assume or attempt to prove that:

- J-space is the model's entire reasoning process;
- a decoded concept proves the model "knows" something consciously;
- detection implies causality;
- a useful detector is automatically adversarially robust;
- J-space must outperform every possible hidden-state probe;
- one benchmark establishes universal prompt-injection defense.

Claims should remain tied to the tested models, fitted lenses, benchmarks, detector, and intervention.

---

# 13. Optional Follow-Up Only If Core Results Are Positive

These are **not part of the core experiment**:

- adaptive monitor evasion;
- additional model families;
- refitting a new J-lens;
- multiple probe architectures;
- full-residual and reconstruction-residual probe suites;
- large general-capability benchmark suites;
- alternative sparse-decomposition algorithms;
- multi-layer detectors.

Only pursue these after the focused experiment establishes a useful signal.

---

# 14. Final Experimental Thesis

The experiment tests one coherent chain:

\[
\boxed{
\text{Can we locate a prompt-injection signal in J-space?}
}
\]

\[
\downarrow
\]

\[
\boxed{
\text{Does the same signal generalize?}
}
\]

\[
\downarrow
\]

\[
\boxed{
\text{Can the signal be present even when the attack succeeds?}
}
\]

\[
\downarrow
\]

\[
\boxed{
\text{Does manipulating the signal change behavior?}
}
\]

\[
\downarrow
\]

\[
\boxed{
\text{Can we use the signal to block attacks without breaking clean tasks?}
}
\]

This progression keeps the project focused on the original research question while moving from **readability → generalization → behavior → causality → practical defense**.
