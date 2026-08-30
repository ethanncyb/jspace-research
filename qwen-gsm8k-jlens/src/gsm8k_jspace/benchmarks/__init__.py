"""Benchmark plugin protocol and registry."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from gsm8k_jspace.config import AppConfig, ConfigError
from gsm8k_jspace.types import GSM8KExample


class Benchmark(Protocol):
    name: str

    def load_examples(
        self,
        cfg: AppConfig,
        *,
        rows: Any = None,
        **kwargs: Any,
    ) -> list[GSM8KExample]: ...

    def render_prompt(self, example: GSM8KExample, cfg: AppConfig) -> str: ...

    def evaluate_run(
        self,
        run_dir: str | Path,
        cfg: AppConfig,
        completions: list[dict[str, Any]],
        selection: list[dict[str, Any]],
    ) -> dict[str, Any]: ...


class GSM8KBenchmark:
    name = "gsm8k"

    def load_examples(self, cfg: AppConfig, *, rows=None, **kwargs):
        from gsm8k_jspace.datasets.gsm8k import load_gsm8k_examples

        return load_gsm8k_examples(cfg, rows=rows)

    def render_prompt(self, example: GSM8KExample, cfg: AppConfig) -> str:
        from gsm8k_jspace.prompting.gsm8k import render_prompt

        return render_prompt(example, cfg)

    def evaluate_run(self, run_dir, cfg, completions, selection):
        from gsm8k_jspace.evaluation.evaluator import evaluate_gsm8k_run

        return evaluate_gsm8k_run(run_dir, cfg, completions, selection)


class BIPIABenchmark:
    name = "bipia"

    def load_examples(self, cfg: AppConfig, *, rows=None, **kwargs):
        from gsm8k_jspace.datasets.bipia import load_bipia_examples

        return load_bipia_examples(
            cfg,
            contexts=kwargs.get("contexts"),
            attacks=kwargs.get("attacks"),
        )

    def render_prompt(self, example: GSM8KExample, cfg: AppConfig) -> str:
        from gsm8k_jspace.prompting.bipia import render_prompt

        return render_prompt(example, cfg)

    def evaluate_run(self, run_dir, cfg, completions, selection):
        from gsm8k_jspace.evaluation.bipia import evaluate_bipia_run

        return evaluate_bipia_run(run_dir, cfg, completions, selection)


class AgentDojoBenchmark:
    name = "agentdojo"

    def load_examples(self, cfg: AppConfig, *, rows=None, **kwargs):
        from gsm8k_jspace.datasets.agentdojo import load_agentdojo_examples

        return load_agentdojo_examples(cfg, payload=kwargs.get("payload"), rows=rows)

    def render_prompt(self, example: GSM8KExample, cfg: AppConfig) -> str:
        from gsm8k_jspace.prompting.agentdojo import render_prompt

        return render_prompt(example, cfg)

    def evaluate_run(self, run_dir, cfg, completions, selection):
        from gsm8k_jspace.evaluation.agentdojo import evaluate_agentdojo_run

        return evaluate_agentdojo_run(run_dir, cfg, completions, selection)


class InjecAgentBenchmark:
    name = "injecagent"

    def load_examples(self, cfg: AppConfig, *, rows=None, **kwargs):
        from gsm8k_jspace.datasets.injecagent import load_injecagent_examples

        return load_injecagent_examples(
            cfg,
            user_cases=kwargs.get("user_cases"),
            attacker_cases=kwargs.get("attacker_cases"),
            rows=rows,
        )

    def render_prompt(self, example: GSM8KExample, cfg: AppConfig) -> str:
        from gsm8k_jspace.prompting.injecagent import render_prompt

        return render_prompt(example, cfg)

    def evaluate_run(self, run_dir, cfg, completions, selection):
        from gsm8k_jspace.evaluation.injecagent import evaluate_injecagent_run

        return evaluate_injecagent_run(run_dir, cfg, completions, selection)


_REGISTRY: dict[str, Benchmark] = {
    "gsm8k": GSM8KBenchmark(),
    "bipia": BIPIABenchmark(),
    "agentdojo": AgentDojoBenchmark(),
    "injecagent": InjecAgentBenchmark(),
}


def get_benchmark(cfg: AppConfig | str) -> Benchmark:
    name = cfg if isinstance(cfg, str) else cfg.benchmark.name
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise ConfigError(f"unknown benchmark {name!r}") from exc
