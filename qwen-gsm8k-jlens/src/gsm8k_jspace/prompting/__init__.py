from gsm8k_jspace.benchmarks import get_benchmark
from gsm8k_jspace.config import AppConfig
from gsm8k_jspace.prompting.format import format_model_prompt
from gsm8k_jspace.prompting.gsm8k import DEFAULT_FEW_SHOT
from gsm8k_jspace.prompting.gsm8k import render_prompt as render_gsm8k_prompt
from gsm8k_jspace.types import GSM8KExample


def render_prompt(example: GSM8KExample, cfg: AppConfig) -> str:
    return get_benchmark(cfg).render_prompt(example, cfg)


__all__ = ["DEFAULT_FEW_SHOT", "format_model_prompt", "render_gsm8k_prompt", "render_prompt"]
