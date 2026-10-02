from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
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


@torch.no_grad()
def nonnegative_gradient_pursuit(
    activations: torch.Tensor,
    dictionary: torch.Tensor,
    *,
    sparsity_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Approximate activations with a nonnegative gradient-pursuit update.

    Each iteration selects the unused atom with the largest positive residual
    correlation, then updates every active coefficient along the gradient with
    the exact line-search step, clipped to keep coefficients nonnegative.  The
    implementation intentionally searches the full dictionary so it provides a
    useful robustness comparison to the screened least-squares pursuit used by
    the primary experiment.
    """

    if activations.ndim != 2 or dictionary.ndim != 2:
        raise ValueError("activations and dictionary must both be rank-2 tensors")
    if activations.shape[1] != dictionary.shape[1]:
        raise ValueError("activation and dictionary widths must match")
    if not 0 < sparsity_k <= dictionary.shape[0]:
        raise ValueError("sparsity_k must be positive and no larger than the dictionary")

    device = dictionary.device
    hidden = activations.to(device=device, dtype=torch.float32)
    batch_size = hidden.shape[0]
    supports = torch.full(
        (batch_size, sparsity_k), -1, dtype=torch.long, device=device
    )
    coefficients = torch.zeros(
        (batch_size, sparsity_k), dtype=torch.float32, device=device
    )
    residual = hidden.clone()

    for step in range(sparsity_k):
        correlations = residual.to(dictionary.dtype) @ dictionary.T
        if step:
            correlations.scatter_(1, supports[:, :step].clamp_min(0), -torch.inf)
        values, token_ids = correlations.max(dim=1)
        active = values > 0
        if not bool(active.any()):
            break
        supports[active, step] = token_ids[active]

        active_supports = supports[:, : step + 1]
        valid = active_supports >= 0
        atoms = dictionary[active_supports.clamp_min(0)].float()
        gradient = torch.einsum("bkd,bd->bk", atoms, residual)
        gradient.masked_fill_(~valid, 0.0)
        update_in_activation = torch.einsum("bkd,bk->bd", atoms, gradient)
        numerator = torch.einsum("bd,bd->b", residual, update_in_activation)
        denominator = torch.einsum(
            "bd,bd->b", update_in_activation, update_in_activation
        ).clamp_min(1e-12)
        step_size = (numerator / denominator).clamp_min(0.0)

        current = coefficients[:, : step + 1]
        negative = gradient < 0
        feasible = torch.where(
            negative,
            current / (-gradient).clamp_min(1e-12),
            torch.full_like(gradient, torch.inf),
        )
        max_step = feasible.min(dim=1).values
        step_size = torch.minimum(step_size, max_step)
        step_size.masked_fill_(~active, 0.0)
        coefficients[:, : step + 1] = (
            current + step_size[:, None] * gradient
        ).clamp_min(0.0)

        reconstruction = torch.einsum(
            "bkd,bk->bd", atoms, coefficients[:, : step + 1]
        )
        residual = hidden - reconstruction

    final_atoms = dictionary[supports.clamp_min(0)].float()
    valid = supports >= 0
    final_coefficients = coefficients.masked_fill(~valid, 0.0)
    reconstructions = torch.einsum("bkd,bk->bd", final_atoms, final_coefficients)
    return reconstructions.cpu(), supports.cpu(), final_coefficients.cpu()


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


def select_task_macro_balanced_threshold(
    labels: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    tasks: Sequence[str],
) -> tuple[float, float]:
    """Choose one threshold by training task-macro balanced accuracy.

    Exact ties use the higher threshold. Scores equal to the threshold are
    classified as attacks.
    """

    label_values = np.asarray(labels, dtype=np.int64)
    score_values = np.asarray(scores, dtype=np.float64)
    task_values = np.asarray(tasks, dtype=object)
    if not (len(label_values) == len(score_values) == len(task_values)):
        raise ValueError("labels, scores, and tasks must have equal length")
    if not np.isfinite(score_values).all():
        raise ValueError("scores must be finite")

    task_names = sorted(set(task_values.tolist()))
    if not task_names:
        raise ValueError("At least one task is required")
    for task in task_names:
        if set(label_values[task_values == task].tolist()) != {0, 1}:
            raise ValueError(f"Task {task} must contain both labels")

    unique_scores = np.unique(score_values)[::-1]
    thresholds = np.concatenate(
        ([np.nextafter(unique_scores[0], np.inf)], unique_scores)
    )
    macro_values = np.zeros(len(thresholds), dtype=np.float64)
    for task in task_names:
        mask = task_values == task
        task_labels = label_values[mask]
        task_scores = score_values[mask]
        positives = task_scores[task_labels == 1]
        negatives = task_scores[task_labels == 0]
        tpr = (positives[:, None] >= thresholds[None, :]).mean(axis=0)
        tnr = (negatives[:, None] < thresholds[None, :]).mean(axis=0)
        macro_values += 0.5 * (tpr + tnr)
    macro_values /= len(task_names)
    best_value = float(macro_values.max())
    best_indices = np.flatnonzero(np.isclose(macro_values, best_value, rtol=0, atol=1e-12))
    best_threshold = float(thresholds[best_indices[0]])
    return best_threshold, best_value


def threshold_metrics(
    labels: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    threshold: float,
) -> dict[str, float]:
    label_values = np.asarray(labels, dtype=np.int64)
    score_values = np.asarray(scores, dtype=np.float64)
    if set(label_values.tolist()) != {0, 1}:
        raise ValueError("Threshold metrics require both labels")
    predictions = (score_values >= threshold).astype(np.int64)
    positives = label_values == 1
    negatives = label_values == 0
    return {
        "auprc": float(average_precision_score(label_values, score_values)),
        "auroc": float(roc_auc_score(label_values, score_values)),
        "balanced_accuracy": float(balanced_accuracy_score(label_values, predictions)),
        "tpr": float(predictions[positives].mean()),
        "fpr": float(predictions[negatives].mean()),
    }
