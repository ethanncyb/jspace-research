from __future__ import annotations

import importlib.util
import json
import os
import re
import runpy
from pathlib import Path
from types import ModuleType
from typing import Any

from tqdm.auto import tqdm

from ..phase1.data import hash_messages, render_ids
from ..runtime import append_jsonl, read_resumable_jsonl
from .common import (
    AGENT_DECODING_VERSION,
    content_hash,
    decode_completion,
    require_generation_context,
    save_record,
)

#: Upstream InjecAgent asks gpt-4-0613 for a simulated attacker tool response when its
#: pinned table has no entry for the agent's step-1 parameters. OpenRouter's openai/gpt-4
#: is that model.
SIMULATION_MODEL = "openai/gpt-4"
SIMULATION_CACHE = "injecagent_simulated_responses.jsonl"
_SIMULATION_ATTEMPTS = 3


def _extract_code_block(text: str) -> str | None:
    """Upstream's extraction: the first fenced block, minus a leading 'json' tag."""

    matches: list[str] = re.findall(r"```(.*?)```", text, re.DOTALL)
    if not matches:
        return None
    response: str = matches[0]
    if response.startswith("json"):
        response = response[4:]
    return response.strip()


class SimulationUnavailable(RuntimeError):
    """A simulated response is needed but cannot be generated right now.

    The case is deferred rather than failing the run: every other case still
    completes, and a later resume generates only the deferred ones.
    """


class SimulatedResponses:
    """Pinned simulated responses, falling back to upstream's GPT-4 generation.

    Generated responses are appended to a JSONL cache in the run directory, which the
    runner pushes with the other caches, so a resumed run reuses the response it already
    generated instead of sampling a new one.
    """

    def __init__(self, root: Path, output_dir: Path, tools: dict[str, dict[str, Any]]) -> None:
        with (root / "data/attacker_simulated_responses.json").open("r", encoding="utf-8") as handle:
            self.pinned: dict[str, str] = json.load(handle)
        prompts: dict[str, Any] = runpy.run_path(str(root / "src/prompts/generation_prompts.py"))
        self.system_message: str = prompts["SYSTEM_MESSAGE"]
        self.template: str = prompts["DS_ATTACKER_TOOL_RESPONSE_GEN_MESSAGE"]
        self.example: str = prompts["EXAMPLE"]
        self.tools: dict[str, dict[str, Any]] = tools
        self.cache_path: Path = output_dir / SIMULATION_CACHE
        self.generated: dict[str, str] = {
            str(row["key"]): str(row["response"]) for row in read_resumable_jsonl(self.cache_path)
        }
        self._client: Any = None

    def get(self, tool: str, parameters: str) -> tuple[str, str]:
        """Return the simulated response and its source, 'pinned' or 'generated'."""

        key: str = f"({tool},{parameters})"
        if key in self.pinned:
            return self.pinned[key], "pinned"
        if key not in self.generated:
            self.generated[key] = self._generate(key, tool, parameters)
        return self.generated[key], "generated"

    def _generate(self, key: str, tool: str, parameters: str) -> str:
        if self._client is None:
            try:
                from openai import OpenAI

                from ..phase2.scoring import OPENROUTER_BASE_URL

                self._client = OpenAI(
                    base_url=OPENROUTER_BASE_URL, api_key=os.environ["OPENROUTER_API_KEY"]
                )
            except (ImportError, KeyError) as exc:
                raise SimulationUnavailable(f"No simulation client: {exc!r}") from exc
        prompt: str = self.template.format(
            example=self.example,
            attacker_tool=json.dumps(self.tools[tool], indent=True),
            tool_parameters=parameters,
        )
        for _ in range(_SIMULATION_ATTEMPTS):
            try:
                completion: Any = self._client.chat.completions.create(
                    model=SIMULATION_MODEL,
                    messages=[
                        {"role": "system", "content": self.system_message},
                        {"role": "user", "content": prompt},
                    ],
                )
            except Exception as exc:  # noqa: BLE001 - any API failure defers the case
                raise SimulationUnavailable(f"Simulation request failed: {exc!r}") from exc
            response: str | None = _extract_code_block(completion.choices[0].message.content or "")
            if response:
                append_jsonl(
                    self.cache_path,
                    {
                        "key": key,
                        "response": response,
                        "requested_model": SIMULATION_MODEL,
                        "returned_model": getattr(completion, "model", None),
                    },
                )
                return response
        raise SimulationUnavailable(f"No fenced response after {_SIMULATION_ATTEMPTS} attempts")


def _load_parser(root: Path) -> ModuleType:
    path = root / "src" / "output_parsing.py"
    spec = importlib.util.spec_from_file_location("jspace_injecagent_output_parsing", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load InjecAgent parser: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tools(root: Path) -> dict[str, dict[str, Any]]:
    with (root / "data" / "tools.json").open("r", encoding="utf-8") as handle:
        toolkits = json.load(handle)
    return {
        toolkit["toolkit"] + tool["name"]: {**tool, "name": toolkit["toolkit"] + tool["name"]}
        for toolkit in toolkits
        for tool in toolkit["tools"]
    }


def _prompt(
    item: dict[str, Any],
    tools: dict[str, dict[str, Any]],
    system_prompt: str,
    user_prompt: str,
    *,
    second: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    names = [item["User Tool"], *item["Attacker Tools"]]
    descriptions = json.dumps([tools[name] for name in names], indent=1)
    scratchpad = (
        f"\n{item['Thought']}\nAction: {item['User Tool']}\n"
        f"Action Input: {item['Tool Parameters']}\nObservation: {item['Tool Response']}\n"
    )
    if second is not None:
        scratchpad += (
            f"Thought: {second['thought']}\nAction: {item['Attacker Tools'][0]}\n"
            f"Action Input: {second['parameters']}\nObservation: {second['response']}\n"
        )
    filled = user_prompt.format(
        available_tool_descriptions=descriptions,
        input=item["User Instruction"],
        available_tool_names=names,
        agent_scratchpad=scratchpad,
    )
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": filled}]


def build_cases(config: Any) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for subgroup, filename in (
        ("direct_harm", "test_cases_dh_base.json"),
        ("data_stealing", "test_cases_ds_base.json"),
    ):
        with (config.injecagent_root / "data" / filename).open("r", encoding="utf-8") as handle:
            values = json.load(handle)
        expected_count = 510 if subgroup == "direct_harm" else 544
        if len(values) != expected_count:
            raise RuntimeError(
                f"Pinned InjecAgent {subgroup} data must contain {expected_count} cases"
            )
        for index, item in enumerate(values):
            cases.append(
                {
                    "case_id": f"injecagent:{subgroup}:{index:04d}",
                    "benchmark": "injecagent",
                    "task": None,
                    "subgroup": subgroup,
                    "condition": "attack",
                    "item": item,
                }
            )
    cases.sort(key=lambda row: row["case_id"])
    if config.smoke:
        cases = [
            case
            for subgroup in ("direct_harm", "data_stealing")
            for case in [row for row in cases if row["subgroup"] == subgroup][:3]
        ]
    return cases


def _native_eval(parser: ModuleType, output: str, item: dict[str, Any], target: str) -> dict:
    try:
        return parser.evaluate_output_prompted(
            output,
            target,
            item["Attacker Tools"],
            item["User Tool"],
            item["Attacker Instruction"],
        )
    except (AssertionError, IndexError, KeyError, TypeError, ValueError) as exc:
        return {"eval": "invalid", "invalid_reason": type(exc).__name__}


def generate(
    config: Any,
    model: Any,
    scorer: Any,
    completed: dict[str, dict[str, Any]],
    identity: dict[str, Any],
) -> None:
    output_path = config.output_dir / "injecagent_records.jsonl"
    prompt_values = runpy.run_path(str(config.injecagent_root / "src/prompts/agent_prompts.py"))
    system_prompt, user_prompt = prompt_values["PROMPT_DICT"][config.injecagent_prompt_type]
    tools = _tools(config.injecagent_root)
    parser = _load_parser(config.injecagent_root)
    simulated = SimulatedResponses(config.injecagent_root, config.output_dir, tools)

    deferred: list[str] = []
    cases = build_cases(config)
    expected_ids = {case["case_id"] for case in cases}
    unexpected = sorted(set(completed) - expected_ids)
    if unexpected:
        raise RuntimeError(f"Unexpected cached InjecAgent case ID: {unexpected[0]}")
    progress = tqdm(
        total=len(cases), initial=len(completed), desc="Phase 4 InjecAgent generation"
    )
    for case in cases:
        case_hash = content_hash({**case, "decoding": AGENT_DECODING_VERSION})
        if case["case_id"] in completed:
            if completed[case["case_id"]].get("case_hash") != case_hash:
                raise RuntimeError("Cached InjecAgent case identity changed")
            continue
        item = case["item"]
        messages = _prompt(item, tools, system_prompt, user_prompt)
        input_ids = render_ids(model.tokenizer, messages)
        require_generation_context(
            int(input_ids.shape[-1]), model.context_length, config.max_new_tokens, "InjecAgent"
        )
        tokens, residual = model.generate_with_capture(
            input_ids,
            max_new_tokens=config.max_new_tokens,
            layer=scorer.mean["selected_layer"],
        )
        output = decode_completion(model.tokenizer, tokens)
        first = _native_eval(parser, output, item, item["Attacker Tools"][0])
        second_output = None
        second_eval = None
        step2_source = None
        if case["subgroup"] == "data_stealing" and first["eval"] == "succ":
            parameters = first["Attacker Tool Parameters"]
            try:
                response, step2_source = simulated.get(item["Attacker Tools"][0], parameters)
            except SimulationUnavailable as exc:
                print(f"Deferring {case['case_id']}: {exc}", flush=True)
                deferred.append(case["case_id"])
                progress.update()
                continue
            second = {
                "thought": first["Attacker Tool Thought"],
                "parameters": parameters,
                "response": response,
            }
            second_messages = _prompt(
                item, tools, system_prompt, user_prompt, second=second
            )
            second_ids = render_ids(model.tokenizer, second_messages)
            require_generation_context(
                int(second_ids.shape[-1]),
                model.context_length,
                config.max_new_tokens,
                "InjecAgent second step",
            )
            second_tokens = model.generate_from_prompt(
                second_ids, max_new_tokens=config.max_new_tokens
            )
            second_output = decode_completion(model.tokenizer, second_tokens)
            second_eval = _native_eval(
                parser, second_output, item, item["Attacker Tools"][1]
            )

        final_eval = first["eval"] if case["subgroup"] == "direct_harm" else (
            second_eval["eval"] if first["eval"] == "succ" and second_eval else first["eval"]
        )
        native_valid = final_eval != "invalid"
        native_success = (
            first["eval"] == "succ"
            if case["subgroup"] == "direct_harm"
            else bool(second_eval and second_eval["eval"] == "succ")
        )
        save_record(
            output_path,
            {
                **identity,
                **{key: value for key, value in case.items() if key != "item"},
                "case_hash": case_hash,
                "prompt_hash": hash_messages(messages),
                **scorer.score(residual, scorer.dictionary),
                "injection_exposed": True,
                "generated_response": output,
                "second_generated_response": second_output,
                "native_valid": native_valid,
                "native_utility": None,
                "native_attack_success": native_success,
                "native_step1_result": first["eval"],
                "native_step2_result": second_eval["eval"] if second_eval else None,
                "native_step2_response_source": step2_source,
            },
        )
        progress.update()
    progress.close()
    if deferred:
        raise RuntimeError(
            f"{len(deferred)} InjecAgent case(s) deferred until a simulated response can be "
            f"generated; rerun generate to complete them: {deferred}"
        )
