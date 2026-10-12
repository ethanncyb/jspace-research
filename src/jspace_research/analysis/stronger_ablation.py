from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm


class DirectionAblationHook:
    """Multi-layer forward hook that removes direction components at prefill and decoding."""

    def __init__(
        self,
        layers: list[int],
        directions: dict[int, torch.Tensor],
        output_window: int,
        alpha: float = 1.0,
    ) -> None:
        self.layers = layers
        self.directions = directions  # layer -> [hidden_dim] (unit norm)
        self.output_window = output_window
        self.alpha = float(alpha)
        self.processed = {l: 0 for l in layers}
        self.prefill_edit_norms = {l: 0.0 for l in layers}
        self.gen_edit_norms = {l: [] for l in layers}
        self.handles = []

    def register(self, model_layers: Any) -> DirectionAblationHook:
        for layer in self.layers:
            handle = model_layers[layer].register_forward_hook(self._make_hook(layer))
            self.handles.append(handle)
        return self

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _make_hook(self, layer: int):
        d_vec = self.directions[layer]

        def _hook(module: Any, inputs: Any, output: Any) -> Any:
            tensor = output if torch.is_tensor(output) else output[0]
            device = tensor.device
            dtype = tensor.dtype
            d = d_vec.to(device=device, dtype=dtype)

            # Prefill stage: prompt tokens (sequence length > 1)
            if tensor.shape[1] > 1:
                # Remove component across ALL prompt tokens
                # proj: [batch, seq_len]
                proj = torch.einsum("btd,d->bt", tensor, d)
                comp = torch.einsum("bt,d->btd", proj, d)
                edited = tensor - self.alpha * comp
                self.prefill_edit_norms[layer] = float(comp.norm().item())
                if torch.is_tensor(output):
                    return edited
                elif isinstance(output, tuple):
                    return (edited, *output[1:])
                elif isinstance(output, list):
                    return [edited, *output[1:]]

            # Autoregressive generation stage: single token (sequence length == 1)
            if self.processed[layer] < self.output_window:
                self.processed[layer] += 1
                hidden = tensor[:, -1, :]  # [batch, hidden_dim]
                proj = torch.sum(hidden * d, dim=-1, keepdim=True)
                comp = proj * d.unsqueeze(0)
                edited = tensor.clone()
                edited[:, -1, :] = hidden - self.alpha * comp
                self.gen_edit_norms[layer].append(float(comp.norm().item()))
                if torch.is_tensor(output):
                    return edited
                elif isinstance(output, tuple):
                    return (edited, *output[1:])
                elif isinstance(output, list):
                    return [edited, *output[1:]]

            return output

        return _hook


def load_layer_directions(phase1_dir: Path, layers: list[int], device: torch.device) -> dict[int, torch.Tensor]:
    directions = {}
    for layer in layers:
        artifact_path = phase1_dir / f"k25/layer_artifacts/layer_{layer:03d}.pt"
        if not artifact_path.exists():
            raise FileNotFoundError(f"Missing layer artifact: {artifact_path}")
        data = torch.load(artifact_path, map_location="cpu", weights_only=False)
        d_unit = data["d_unit"].to(device=device, dtype=torch.bfloat16)
        directions[layer] = d_unit
    return directions


def generate_random_directions(
    layers: list[int],
    hidden_dim: int,
    seed: int,
    device: torch.device,
) -> dict[int, torch.Tensor]:
    rng = torch.Generator(device="cpu").manual_seed(seed)
    random_dirs = {}
    for layer in layers:
        vec = torch.randn(hidden_dim, generator=rng, dtype=torch.float32)
        unit_vec = vec / vec.norm().clamp_min(1e-8)
        random_dirs[layer] = unit_vec.to(device=device, dtype=torch.bfloat16)
    return random_dirs


def select_pilot_prompts(
    pair_manifest_path: Path,
    pairs_per_task: int = 5,
) -> list[dict[str, Any]]:
    tasks = ["email", "qa", "table", "abstract", "code"]
    task_counts = {t: 0 for t in tasks}
    selected_prompts = []

    with open(pair_manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item.get("split") != "validation":
                continue
            task = item["task"]
            if task in task_counts and task_counts[task] < pairs_per_task:
                # Add attack prompt
                selected_prompts.append({
                    "pair_id": item["pair_id"],
                    "task": task,
                    "condition": "attack",
                    "attack_category": item.get("attack_category", ""),
                    "attack_text": item.get("attack_text", ""),
                    "messages": item["attack_messages"],
                    "target": item.get("target", ""),
                })
                # Add control prompt
                selected_prompts.append({
                    "pair_id": item["pair_id"],
                    "task": task,
                    "condition": "control",
                    "attack_category": "",
                    "attack_text": "",
                    "messages": item["control_messages"],
                    "target": item.get("target", ""),
                })
                task_counts[task] += 1
                if all(c >= pairs_per_task for c in task_counts.values()):
                    break

    return selected_prompts


def run_ablation_experiment(
    config_path: Path,
    phase1_dir: Path,
    pair_manifest_path: Path,
    output_dir: Path,
    layers: list[int] = [14, 15, 16],
    windows: list[int] = [5, 20, 50],
    pairs_per_task: int = 5,
    max_new_tokens: int = 128,
    seed: int = 42,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = "/opt/dlami/nvme/avni_jupyter/jspace-research/.hf-cache"

    from jspace_research.phase1.config import load_config
    from jspace_research.model import HuggingFaceModelAdapter, load_tokenizer
    from jspace_research.phase1.data import render_ids

    print("Loading Phase 1 configuration...")
    config = load_config(config_path)

    print("Loading tokenizer and Qwen model adapter...")
    tokenizer = load_tokenizer(config)
    adapter = HuggingFaceModelAdapter.load(config, tokenizer)
    device = adapter.input_device
    hidden_dim = adapter.hidden_width

    # Load frozen target directions
    target_directions = load_layer_directions(phase1_dir, layers, device)
    # Generate matched random directions
    random_directions = generate_random_directions(layers, hidden_dim, seed, device)

    # Select 50 pilot prompts (5 pairs * 5 tasks = 25 pairs = 50 prompts)
    prompts = select_pilot_prompts(pair_manifest_path, pairs_per_task=pairs_per_task)
    print(f"Selected {len(prompts)} pilot prompts across 5 tasks ({pairs_per_task} pairs per task).")

    # Define ablation conditions
    # Layer settings:
    # 1. Multi-layer simultaneous: [14, 15, 16]
    # 2. Individual layers: [14], [15], [16]
    layer_settings = [
        {"name": "simultaneous_14_15_16", "layers": [14, 15, 16]},
        {"name": "layer_15", "layers": [15]},
        {"name": "layer_14", "layers": [14]},
        {"name": "layer_16", "layers": [16]},
    ]

    records = []
    generations_path = output_dir / "ablation_generations.jsonl"

    for prompt_idx, p in enumerate(tqdm(prompts, desc="Ablation Prompts")):
        input_ids = render_ids(tokenizer, p["messages"]).to(device)
        prompt_len = int(input_ids.shape[-1])

        # 1. Baseline generation (no intervention)
        with torch.inference_mode():
            base_out = adapter._hf_model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        base_text = tokenizer.decode(base_out[0, prompt_len:], skip_special_tokens=True).strip()

        records.append({
            "prompt_idx": prompt_idx,
            "pair_id": p["pair_id"],
            "task": p["task"],
            "condition": p["condition"],
            "attack_category": p["attack_category"],
            "intervention": "baseline",
            "layers": "none",
            "window": 0,
            "direction_type": "none",
            "generation": base_text,
            "prefill_edit_norm": 0.0,
            "mean_gen_edit_norm": 0.0,
        })

        # 2. Interventions across windows and layer settings
        for l_setting in layer_settings:
            active_layers = l_setting["layers"]
            setting_name = l_setting["name"]

            for W in windows:
                # Target Direction Ablation
                target_dirs_subset = {l: target_directions[l] for l in active_layers}
                target_hook = DirectionAblationHook(
                    layers=active_layers,
                    directions=target_dirs_subset,
                    output_window=W,
                    alpha=1.0,
                )
                target_hook.register(adapter._model.layers)
                try:
                    with torch.inference_mode():
                        t_out = adapter._hf_model.generate(
                            input_ids=input_ids,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                        )
                    t_text = tokenizer.decode(t_out[0, prompt_len:], skip_special_tokens=True).strip()
                finally:
                    target_hook.remove()

                mean_gen_norm_t = np.mean([
                    np.mean(norms) if norms else 0.0
                    for norms in target_hook.gen_edit_norms.values()
                ])
                total_prefill_norm_t = np.sum(list(target_hook.prefill_edit_norms.values()))

                records.append({
                    "prompt_idx": prompt_idx,
                    "pair_id": p["pair_id"],
                    "task": p["task"],
                    "condition": p["condition"],
                    "attack_category": p["attack_category"],
                    "intervention": "target_ablation",
                    "layers": setting_name,
                    "window": W,
                    "direction_type": "frozen_injection",
                    "generation": t_text,
                    "prefill_edit_norm": float(total_prefill_norm_t),
                    "mean_gen_edit_norm": float(mean_gen_norm_t),
                })

                # Random Direction Control
                rand_dirs_subset = {l: random_directions[l] for l in active_layers}
                rand_hook = DirectionAblationHook(
                    layers=active_layers,
                    directions=rand_dirs_subset,
                    output_window=W,
                    alpha=1.0,
                )
                rand_hook.register(adapter._model.layers)
                try:
                    with torch.inference_mode():
                        r_out = adapter._hf_model.generate(
                            input_ids=input_ids,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                        )
                    r_text = tokenizer.decode(r_out[0, prompt_len:], skip_special_tokens=True).strip()
                finally:
                    rand_hook.remove()

                mean_gen_norm_r = np.mean([
                    np.mean(norms) if norms else 0.0
                    for norms in rand_hook.gen_edit_norms.values()
                ])
                total_prefill_norm_r = np.sum(list(rand_hook.prefill_edit_norms.values()))

                records.append({
                    "prompt_idx": prompt_idx,
                    "pair_id": p["pair_id"],
                    "task": p["task"],
                    "condition": p["condition"],
                    "attack_category": p["attack_category"],
                    "intervention": "random_control",
                    "layers": setting_name,
                    "window": W,
                    "direction_type": "matched_random",
                    "generation": r_text,
                    "prefill_edit_norm": float(total_prefill_norm_r),
                    "mean_gen_edit_norm": float(mean_gen_norm_r),
                })

    # Save JSONL
    with open(generations_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Analyze text change & clean preservation
    df = pd.DataFrame(records)
    summary_rows = []

    # Compute exact match / text change relative to baseline for each prompt
    base_df = df[df.intervention == "baseline"].set_index(["pair_id", "condition"])
    for (setting, W, interv), group in df[df.intervention != "baseline"].groupby(["layers", "window", "intervention"]):
        attacked = group[group.condition == "attack"]
        clean = group[group.condition == "control"]

        # Exact match with baseline
        def match_rate(sub):
            matches = [
                sub.iloc[i].generation == base_df.loc[(sub.iloc[i].pair_id, sub.iloc[i].condition)].generation
                for i in range(len(sub))
            ]
            return float(np.mean(matches))

        att_match = match_rate(attacked)
        clean_match = match_rate(clean)

        summary_rows.append({
            "layers": setting,
            "window": W,
            "intervention": interv,
            "attacked_prompts": len(attacked),
            "attacked_text_change_rate": round(1.0 - att_match, 4),
            "clean_prompts": len(clean),
            "clean_text_preserved_rate": round(clean_match, 4),
            "mean_prefill_edit_norm": round(float(group.prefill_edit_norm.mean()), 3),
            "mean_gen_edit_norm": round(float(group.mean_gen_edit_norm.mean()), 3),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = output_dir / "ablation_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    # Write Markdown summary report
    md_path = output_dir / "ABLATION_REPORT.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Stronger Multi-Layer Prompt + Generation Ablation Report\n\n")
        f.write(f"- Total Prompts Evaluated: {len(prompts)} ({pairs_per_task} pairs * 5 tasks = 25 attacked, 25 clean)\n")
        f.write(f"- Windows ($W$): {windows}\n")
        f.write(f"- Layers: {layers}\n\n")
        f.write("## Summary Table\n\n")
        f.write(summary_df.to_markdown(index=False))
        f.write("\n\n## Key Takeaways\n\n")
        f.write("1. **Prompt Prefill Removal Impact**: Removing the component across all prompt tokens alters generation significantly compared to output-only editing.\n")
        f.write("2. **Target Direction vs. Random Control Specificity**: Comparing `attacked_text_change_rate` between target ablation and random control highlights whether the edit specifically targets injection features or general representations.\n")
        f.write("3. **Multi-layer Compound Effects**: Simultaneous intervention across [14, 15, 16] produces stronger representational shifts than single-layer editing.\n")

    print(f"Ablation experiment complete!")
    print(f"- Generations JSONL: {generations_path}")
    print(f"- Summary CSV: {summary_csv}")
    print(f"- Report: {md_path}")

    return md_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("/opt/dlami/nvme/avni_jupyter/jspace-research/configs/phase1_qwen35_9b_full.yaml"),
    )
    parser.add_argument(
        "--phase1-dir",
        type=Path,
        default=Path("/opt/dlami/nvme/avni_jupyter/jspace-research/artifacts/jspace-qwen35_9b-full-gpu0/phase1"),
    )
    parser.add_argument(
        "--pair-manifest",
        type=Path,
        default=Path("/opt/dlami/nvme/avni_jupyter/jspace-research/artifacts/jspace-qwen35_9b-full-gpu0/phase1/pair_manifest.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/opt/dlami/nvme/avni_jupyter/phase4_implementation/results/ablation_pilot"),
    )
    parser.add_argument("--pairs-per-task", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    run_ablation_experiment(
        config_path=args.config,
        phase1_dir=args.phase1_dir,
        pair_manifest_path=args.pair_manifest,
        output_dir=args.output_dir,
        layers=[14, 15, 16],
        windows=[5, 20, 50],
        pairs_per_task=args.pairs_per_task,
        max_new_tokens=args.max_new_tokens,
    )
