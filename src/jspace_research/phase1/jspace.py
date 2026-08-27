from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm.auto import tqdm


def read_bfloat16_bits(bits: np.ndarray) -> torch.Tensor:
    array = np.asarray(bits, dtype=np.uint16).copy()
    return torch.from_numpy(array).view(torch.bfloat16).float()


def tensor_to_bfloat16_bits(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().to("cpu", dtype=torch.bfloat16).contiguous().view(torch.uint16).numpy()


@torch.no_grad()
def build_normalized_dictionary(
    *,
    jacobian: torch.Tensor,
    unembedding: torch.Tensor,
    layer: int,
    device: torch.device,
    chunk_size: int,
) -> torch.Tensor:
    jacobian = jacobian.to(device=device, dtype=torch.float32)
    vocabulary, width = unembedding.shape
    storage_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    dictionary = torch.empty((vocabulary, width), dtype=storage_dtype, device=device)
    for start in tqdm(
        range(0, vocabulary, chunk_size),
        desc=f"Build dictionary L{layer}",
        leave=False,
    ):
        end = min(vocabulary, start + chunk_size)
        weights = unembedding[start:end].to(device=device, dtype=torch.float32)
        atoms = weights @ jacobian
        atoms = atoms / atoms.norm(dim=1, keepdim=True).clamp_min(1e-8)
        dictionary[start:end] = atoms.to(storage_dtype)
    return dictionary


@torch.no_grad()
def screened_nonnegative_pursuit(
    activations: torch.Tensor,
    dictionary: torch.Tensor,
    *,
    sparsity_k: int = 25,
    screen_candidates: int = 512,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Approximate activations with sparse nonnegative dictionary combinations.

    This is a screened greedy approximation. It is not an exact orthogonal
    projection and does not reproduce Anthropic's gradient-pursuit solver.
    """

    if activations.ndim != 2 or dictionary.ndim != 2:
        raise ValueError("activations and dictionary must both be rank-2 tensors")
    if activations.shape[1] != dictionary.shape[1]:
        raise ValueError("activation and dictionary widths must match")
    if sparsity_k <= 0 or screen_candidates < sparsity_k:
        raise ValueError("Require 0 < sparsity_k <= screen_candidates")

    device = dictionary.device
    hidden = activations.to(device=device, dtype=torch.float32)
    batch_size = hidden.shape[0]
    candidate_count = min(screen_candidates, dictionary.shape[0])
    screen_scores = hidden.to(dictionary.dtype) @ dictionary.T
    top_values, top_ids = torch.topk(screen_scores, k=candidate_count, dim=1)
    candidate_valid = top_values > 0
    candidate_atoms = dictionary[top_ids].float()
    del screen_scores, top_values

    residual = hidden.clone()
    chosen = torch.zeros((batch_size, candidate_count), dtype=torch.bool, device=device)
    supports: list[list[int]] = [[] for _ in range(batch_size)]

    for _ in range(sparsity_k):
        correlations = torch.einsum("bmd,bd->bm", candidate_atoms, residual)
        correlations.masked_fill_(chosen | ~candidate_valid, -torch.inf)
        values, positions = correlations.max(dim=1)
        active = values > 0
        if not bool(active.any()):
            break
        for batch_index in torch.where(active)[0].tolist():
            position = int(positions[batch_index])
            supports[batch_index].append(position)
            chosen[batch_index, position] = True

            support = torch.tensor(supports[batch_index], device=device, dtype=torch.long)
            while support.numel() > 0:
                atoms = candidate_atoms[batch_index, support].T
                coefficients = torch.linalg.lstsq(atoms, hidden[batch_index]).solution
                positive = coefficients > 0
                if bool(positive.all()):
                    break
                support = support[positive]
            supports[batch_index] = support.tolist()
            if support.numel() == 0:
                residual[batch_index] = hidden[batch_index]
            else:
                atoms = candidate_atoms[batch_index, support].T
                coefficients = torch.linalg.lstsq(atoms, hidden[batch_index]).solution.clamp_min(0)
                residual[batch_index] = hidden[batch_index] - atoms @ coefficients

    reconstructions = torch.zeros_like(hidden)
    token_ids_out = torch.full((batch_size, sparsity_k), -1, dtype=torch.long)
    coefficients_out = torch.zeros((batch_size, sparsity_k), dtype=torch.float32)

    for batch_index, support_values in enumerate(supports):
        if not support_values:
            continue
        support = torch.tensor(support_values, device=device, dtype=torch.long)
        atoms = candidate_atoms[batch_index, support].T
        coefficients = torch.linalg.lstsq(atoms, hidden[batch_index]).solution.clamp_min(0)
        reconstructions[batch_index] = atoms @ coefficients
        token_ids = top_ids[batch_index, support].cpu()
        count = min(sparsity_k, token_ids.numel())
        token_ids_out[batch_index, :count] = token_ids[:count]
        coefficients_out[batch_index, :count] = coefficients[:count].cpu()

    return reconstructions.cpu(), token_ids_out, coefficients_out


def batched(indices: Sequence[int] | np.ndarray, batch_size: int) -> Iterable[np.ndarray]:
    values = np.asarray(indices, dtype=np.int64)
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def learn_task_balanced_direction(
    representations: torch.Tensor,
    labels: Sequence[int] | np.ndarray,
    tasks: Sequence[str],
) -> dict[str, torch.Tensor | float]:
    if representations.ndim != 2:
        raise ValueError("representations must be a matrix")
    label_values = np.asarray(labels, dtype=np.int64)
    task_values = np.asarray(tasks, dtype=object)
    if len(representations) != len(label_values) or len(label_values) != len(task_values):
        raise ValueError("representations, labels, and tasks must have equal length")

    attack_means: list[torch.Tensor] = []
    clean_means: list[torch.Tensor] = []
    for task in sorted(set(task_values.tolist())):
        task_mask = task_values == task
        attack_mask = torch.from_numpy(task_mask & (label_values == 1))
        clean_mask = torch.from_numpy(task_mask & (label_values == 0))
        if not bool(attack_mask.any()) or not bool(clean_mask.any()):
            raise ValueError(f"Task {task} must contain both attack and control examples")
        attack_means.append(representations[attack_mask].double().mean(dim=0))
        clean_means.append(representations[clean_mask].double().mean(dim=0))

    mu_attack = torch.stack(attack_means).mean(dim=0).float()
    mu_clean = torch.stack(clean_means).mean(dim=0).float()
    direction_raw = mu_attack - mu_clean
    direction_norm = float(direction_raw.norm())
    if direction_norm <= 1e-10:
        raise RuntimeError("Mean-difference direction norm is approximately zero")
    return {
        "mu_attack": mu_attack,
        "mu_clean": mu_clean,
        "d_raw": direction_raw,
        "d_norm": direction_norm,
        "d_unit": direction_raw / direction_norm,
    }


def direction_scores(
    representations: torch.Tensor,
    clean_mean: torch.Tensor,
    unit_direction: torch.Tensor,
) -> torch.Tensor:
    return (representations.float() - clean_mean.float()) @ unit_direction.float()


def compute_layer_metrics(
    validation_scores: pd.DataFrame,
    layers: Sequence[int],
    tasks: Sequence[str],
    task_display: dict[str, str],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for layer in layers:
        layer_scores = validation_scores[validation_scores.layer == layer]
        task_auprc: list[float] = []
        task_auroc: list[float] = []
        for task in tasks:
            subset = layer_scores[layer_scores.task == task]
            labels = subset.label.to_numpy()
            scores = subset.score.to_numpy()
            if len(np.unique(labels)) != 2:
                raise ValueError(f"Layer {layer}, task {task} lacks both labels")
            auprc = float(average_precision_score(labels, scores))
            auroc = float(roc_auc_score(labels, scores))
            task_auprc.append(auprc)
            task_auroc.append(auroc)
            records.append(
                {
                    "layer": layer,
                    "scope": "task",
                    "task": task,
                    "task_display": task_display[task],
                    "auprc": auprc,
                    "auroc": auroc,
                }
            )
        records.append(
            {
                "layer": layer,
                "scope": "macro",
                "task": None,
                "task_display": "Macro",
                "auprc": float(np.mean(task_auprc)),
                "auroc": float(np.mean(task_auroc)),
            }
        )
    return pd.DataFrame(records)


def select_layer(metrics: pd.DataFrame) -> pd.Series:
    macro = metrics[metrics.scope == "macro"]
    if macro.empty:
        raise ValueError("No macro metrics are available for layer selection")
    return macro.sort_values(["auprc", "layer"], ascending=[False, True], kind="mergesort").iloc[0]
