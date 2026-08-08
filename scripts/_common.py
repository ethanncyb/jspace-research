# Shared loader for the Qwen J-space demos (scripts/steering_demo.py,
# scripts/probe_swap.py). Mirrors the walkthrough.ipynb setup.
from __future__ import annotations

import torch
import transformers

import jlens

# Swap models by commenting/uncommenting MODEL_NAME; LENS_FILE follows via the
# dict below. Note: the 9B lens was fit on the *base* model, not the instruct.
# MODEL_NAME = "Qwen/Qwen3.5-4B"
MODEL_NAME = "Qwen/Qwen3.5-9B-Base"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_REVISION = "qwen-n1000"
LENS_FILE = {
    "Qwen/Qwen3.5-4B": "qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt",
    "Qwen/Qwen3.5-9B-Base": "qwen3.5-9b-pt/jlens/Salesforce-wikitext/Qwen3.5-9B-Base_jacobian_lens.pt",
}[MODEL_NAME]
# Filesystem-safe tag used in results/ artifact names.
MODEL_TAG = {
    "Qwen/Qwen3.5-4B": "qwen3.5-4b",
    "Qwen/Qwen3.5-9B-Base": "qwen3.5-9b-pt",
}[MODEL_NAME]


def load_model_and_lens():
    """Load MODEL_NAME + its Neuronpedia Jacobian lens.

    Returns (hf_model, model, lens, tokenizer, device).
    """
    if torch.cuda.is_available():
        device, dtype = torch.device("cuda"), torch.bfloat16
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device, dtype = torch.device("mps"), torch.float16
    else:
        device, dtype = torch.device("cpu"), torch.float32

    hf = transformers.AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=dtype).to(device)
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.from_pretrained(
        LENS_REPO, filename=LENS_FILE, revision=LENS_REVISION
    )
    print(f"loaded {model} on {device} / {dtype}")
    print(f"loaded {lens}")
    return hf, model, lens, tok, device
