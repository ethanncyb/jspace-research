from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer


def run_token_analysis(
    detector_path: Path,
    output_dir: Path,
    hf_cache_dir: Path,
    phase1_dir: Path,
    validation_scores_path: Path,
    pair_manifest_path: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    import os
    if "HF_HOME" not in os.environ:
        os.environ["HF_HOME"] = str(hf_cache_dir)
    
    # 1. Load detector
    det = torch.load(detector_path, weights_only=False, map_location="cpu")
    feat_ids = det["feature_token_ids"].numpy().astype(np.int64)
    weights = det["weights"].numpy().astype(np.float64)
    threshold = float(det["threshold"])
    intercept = float(det["intercept"])
    
    # 2. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3.5-9B",
        revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        local_files_only=True,
    )
    
    # 3. Top positive and negative tokens
    pos_order = np.argsort(weights)[-20:][::-1]
    neg_order = np.argsort(weights)[:20]
    
    top_pos_list = []
    for rank, idx in enumerate(pos_order, 1):
        tid = int(feat_ids[idx])
        w = float(weights[idx])
        word = tokenizer.decode([tid])
        top_pos_list.append({
            "rank": rank,
            "token_id": tid,
            "word": word,
            "weight": w,
            "direction": "positive_attack",
        })
        
    top_neg_list = []
    for rank, idx in enumerate(neg_order, 1):
        tid = int(feat_ids[idx])
        w = float(weights[idx])
        word = tokenizer.decode([tid])
        top_neg_list.append({
            "rank": rank,
            "token_id": tid,
            "word": word,
            "weight": w,
            "direction": "negative_clean",
        })
        
    top_tokens_df = pd.DataFrame(top_pos_list + top_neg_list)
    top_tokens_csv = output_dir / "top_tokens.csv"
    top_tokens_df.to_csv(top_tokens_csv, index=False)
    
    # 4. Load decomposition memmaps
    decomp_meta = json.loads((phase1_dir / "k25/cache/decompositions/layer_015.json").read_text())
    sparse_shape = tuple(decomp_meta["sparse_shape"])
    supp_path = phase1_dir / "k25/cache/decompositions/layer_015_support_ids_i32.dat"
    coeff_path = phase1_dir / "k25/cache/decompositions/layer_015_coefficients_f32.dat"
    supports = np.memmap(supp_path, dtype=np.int32, mode="r", shape=sparse_shape)
    coeffs = np.memmap(coeff_path, dtype=np.float32, mode="r", shape=sparse_shape)
    
    # Load validation scores & pair manifest
    val_df = pd.read_parquet(validation_scores_path)
    
    # Manifest lookup
    manifest_lookup = {}
    if pair_manifest_path.exists():
        with open(pair_manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                manifest_lookup[item["pair_id"]] = item
                
    def analyze_example(idx: int, case_type: str) -> dict[str, Any]:
        row = val_df.loc[val_df.example_index == idx].iloc[0]
        supp = supports[idx]
        cf = coeffs[idx]
        
        pos = np.searchsorted(feat_ids, supp)
        known = (supp >= 0) & (pos < len(feat_ids))
        known_idx = np.flatnonzero(known)
        known[known_idx] &= (feat_ids[pos[known_idx]] == supp[known_idx])
        
        contribs = []
        for s_id, c, k, p in zip(supp, cf, known, pos):
            if s_id < 0 or c == 0:
                continue
            w = float(weights[p]) if k else 0.0
            word = tokenizer.decode([int(s_id)])
            contribs.append({
                "token_id": int(s_id),
                "word": word,
                "coeff": float(c),
                "weight": float(w),
                "contribution": float(c * w),
            })
            
        contribs.sort(key=lambda x: x["contribution"], reverse=True)
        manifest_item = manifest_lookup.get(row.pair_id, {})
        attack_text = manifest_item.get("attack_text", "")
        
        return {
            "case_type": case_type,
            "example_index": int(idx),
            "example_id": str(row.example_id),
            "pair_id": str(row.pair_id),
            "task": str(row.task),
            "condition": str(row.condition),
            "label": int(row.label),
            "logistic_score": float(row.logistic_score),
            "threshold": threshold,
            "logistic_prediction": bool(row.logistic_prediction),
            "attack_text": attack_text,
            "top_positive_contributions": contribs[:5],
            "top_negative_contributions": contribs[-5:][::-1],
        }

    # Sample Caught Attacks (TP: label=1, pred=True)
    tp_indices = val_df[(val_df.label == 1) & (val_df.logistic_prediction == True)].example_index.values[:3]
    tp_cases = [analyze_example(idx, "Caught Attack (True Positive)") for idx in tp_indices]
    
    # Sample Missed Attacks (FN: label=1, pred=False)
    fn_indices = val_df[(val_df.label == 1) & (val_df.logistic_prediction == False)].example_index.values[:3]
    fn_cases = [analyze_example(idx, "Missed Attack (False Negative)") for idx in fn_indices]
    
    # Sample False Alarms (FP: label=0, pred=True)
    fp_indices = val_df[(val_df.label == 0) & (val_df.logistic_prediction == True)].example_index.values[:3]
    fp_cases = [analyze_example(idx, "False Alarm (False Positive)") for idx in fp_indices]
    
    case_studies = {
        "caught_attacks": tp_cases,
        "missed_attacks": fn_cases,
        "false_alarms": fp_cases,
    }
    
    case_studies_json = output_dir / "case_studies.json"
    with open(case_studies_json, "w", encoding="utf-8") as f:
        json.dump(case_studies, f, indent=2, ensure_ascii=False)
        
    # Write Markdown summary report
    md_report_path = output_dir / "TOKEN_ANALYSIS_REPORT.md"
    with open(md_report_path, "w", encoding="utf-8") as f:
        f.write("# Phase 3/4 Logistic Detector Top-Token Analysis Report\n\n")
        f.write(f"- Total active features: {len(feat_ids)}\n")
        f.write(f"- Intercept: {intercept:+.4f}\n")
        f.write(f"- Classification threshold: {threshold:.4f}\n\n")
        f.write("## 1. Top 20 Positive Logistic Weights (Attack Indicators)\n\n")
        f.write("| Rank | Token ID | Decoded String | Weight |\n|:---:|:---:|:---|:---:|\n")
        for item in top_pos_list:
            f.write(f"| {item['rank']} | `{item['token_id']}` | {repr(item['word'])} | {item['weight']:+.4f} |\n")
        f.write("\n## 2. Top 20 Negative Logistic Weights (Benign / Clean Indicators)\n\n")
        f.write("| Rank | Token ID | Decoded String | Weight |\n|:---:|:---:|:---|:---:|\n")
        for item in top_neg_list:
            f.write(f"| {item['rank']} | `{item['token_id']}` | {repr(item['word'])} | {item['weight']:+.4f} |\n")
        f.write("\n## 3. Case Studies: Words Behind Specific Predictions\n\n")
        for category, cases in case_studies.items():
            f.write(f"### {category.replace('_', ' ').title()}\n\n")
            for c in cases:
                f.write(f"#### {c['case_type']}: Example {c['example_index']} (`{c['example_id']}`, Task: {c['task']})\n")
                f.write(f"- **Label**: {c['label']} | **Score**: {c['logistic_score']:.4f} (Threshold: {threshold:.4f}) | **Pred**: {c['logistic_prediction']}\n")
                if c.get("attack_text"):
                    f.write(f"- **Attack text snippet**: `{repr(c['attack_text'][:80])}`\n")
                f.write("- **Top positive drivers**:\n")
                for p in c["top_positive_contributions"]:
                    f.write(f"  - `{repr(p['word'])}` (ID {p['token_id']}): coeff={p['coeff']:.3f} * weight={p['weight']:+.3f} = **{p['contribution']:+.4f}**\n")
                f.write("- **Top negative drivers**:\n")
                for n in c["top_negative_contributions"]:
                    f.write(f"  - `{repr(n['word'])}` (ID {n['token_id']}): coeff={n['coeff']:.3f} * weight={n['weight']:+.3f} = **{n['contribution']:+.4f}**\n")
                f.write("\n")
                
    print(f"Top-token analysis saved to: {output_dir}")
    print(f"- Report: {md_report_path}")
    print(f"- CSV: {top_tokens_csv}")
    print(f"- Case studies: {case_studies_json}")
    
    return {
        "top_positive": top_pos_list,
        "top_negative": top_neg_list,
        "case_studies": case_studies,
        "report_path": str(md_report_path),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--detector",
        type=Path,
        default=Path("/opt/dlami/nvme/avni_jupyter/phase3_implementation/results/k25/logistic_detector.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/opt/dlami/nvme/avni_jupyter/phase4_implementation/results/token_analysis"),
    )
    parser.add_argument(
        "--hf-cache",
        type=Path,
        default=Path("/opt/dlami/nvme/avni_jupyter/jspace-research/.hf-cache"),
    )
    parser.add_argument(
        "--phase1-dir",
        type=Path,
        default=Path("/opt/dlami/nvme/avni_jupyter/jspace-research/artifacts/jspace-qwen35_9b-full-gpu0/phase1"),
    )
    parser.add_argument(
        "--validation-scores",
        type=Path,
        default=Path("/opt/dlami/nvme/avni_jupyter/phase3_implementation/results/k25/phase3_validation_scores.parquet"),
    )
    parser.add_argument(
        "--pair-manifest",
        type=Path,
        default=Path("/opt/dlami/nvme/avni_jupyter/jspace-research/artifacts/jspace-qwen35_9b-full-gpu0/phase1/pair_manifest.jsonl"),
    )
    args = parser.parse_args()
    run_token_analysis(
        detector_path=args.detector,
        output_dir=args.output_dir,
        hf_cache_dir=args.hf_cache,
        phase1_dir=args.phase1_dir,
        validation_scores_path=args.validation_scores,
        pair_manifest_path=args.pair_manifest,
    )
