# SecurityProbe learning-curve demo (no model, no network, CPU-only).
#
# Simulates a repeated jailbreak attacker against the inference-time guard
# (jlens.guard_probe.SecurityProbe) backed by a persistent incident memory
# (DARWIN-Guard idea, post-training: learning = appending incidents):
#   - rounds 1..R: user "u-repeat" attacks every intent from
#     data/experiments/security-probe.json. Per intent and round the attacker
#     first retries the strategy that worked before (if any — the exact
#     repeat the repeat-offender threshold exists for) and then explores the
#     next strategy in a per-intent queue shuffled by a seeded RNG
#     (reproducible runs). Each attempt is assessed first: a block ends just
#     that attempt; otherwise a mock target answers iff the strategy is in
#     the intent's effective_strategies, and the outcome (success or
#     failure) is learned into the memory;
#   - transfer round: a fresh user "u-fresh" replays the round-1 attacks
#     without learning — with no incidents of their own on record, every
#     block must come through the any-user threshold, showing transfer of
#     experience across users;
#   - over-refusal check: all benign_lookalikes are assessed at the end
#     (none should be blocked).
#
# The memory file is deleted at the start of each run so the demo always
# starts from a clean slate.
#
# Artifacts: results/security_probe_demo.json, results/security_probe_demo.txt,
#            results/security_probe_memory.jsonl
from __future__ import annotations

import datetime
import json
import random
from pathlib import Path

from jlens.guard_memory import GuardMemory
from jlens.guard_probe import SecurityProbe

RESULTS = Path("results")
DATA = Path("data/experiments/security-probe.json")
MEMORY_PATH = RESULTS / "security_probe_memory.jsonl"
ROUNDS = 4
SEED = 0
REPEAT_USER = "u-repeat"
FRESH_USER = "u-fresh"
BENIGN_USER = "u-benign"


def new_attacker(queue: list) -> dict:
    """Per-intent attacker state: shuffled strategy queue, next untried
    index, and strategies that previously jailbroke the mock target."""
    return {"queue": queue, "next": 0, "known_good": []}


def attempt(
    probe: SecurityProbe, intent: dict, strategy: dict, user_id: str, learn: bool
) -> dict:
    """One attack attempt: assess, then (if not blocked) mock-target answer.

    The mock target answers iff the strategy is one of the intent's
    effective_strategies; the outcome is learned into the probe's memory
    unless ``learn`` is False (transfer round: keeps blocks attributable to
    the any-user threshold alone).
    """
    prompt = strategy["template"].format(x=intent["intent"])
    verdict = probe.assess(prompt, user_id)
    answered = False
    if verdict.action != "block":
        answered = strategy["id"] in intent["effective_strategies"]
        if learn:
            probe.learn(prompt, user_id, succeeded=answered, strategy=strategy["id"])
    record = {
        "intent": intent["id"],
        "strategy": strategy["id"],
        "action": verdict.action,
        "score": round(verdict.score, 4),
        "answered": answered,
        "trigger": None,
    }
    if verdict.action == "block":
        record["trigger"] = (
            "repeat_offender"
            if verdict.reason.startswith("repeat offender")
            else "any_user"
        )
    return record


def run_round(
    probe: SecurityProbe, intents, attackers, user_id: str, learn: bool = True
) -> list[dict]:
    """One attack round: per intent, retry the known-good strategy (if any)
    and explore the next queued one."""
    attempts: list[dict] = []
    for intent, attacker in zip(intents, attackers, strict=True):
        if attacker["known_good"]:
            record = attempt(probe, intent, attacker["known_good"][0], user_id, learn)
            record["kind"] = "retry"
            attempts.append(record)
        if attacker["next"] < len(attacker["queue"]):
            strategy = attacker["queue"][attacker["next"]]
            attacker["next"] += 1
            record = attempt(probe, intent, strategy, user_id, learn)
            record["kind"] = "explore"
            attempts.append(record)
            if record["answered"]:
                attacker["known_good"].append(strategy)
    return attempts


def summarize_round(label, user_id: str, attempts: list[dict]) -> dict:
    n = len(attempts)
    blocked = sum(a["action"] == "block" for a in attempts)
    flagged = sum(a["action"] == "flag" for a in attempts)
    answered = sum(a["answered"] for a in attempts)
    return {
        "round": label,
        "user_id": user_id,
        "attempts": n,
        "blocked": blocked,
        "flagged": flagged,
        "answered": answered,
        "block_rate": blocked / n,
        "attack_success_rate": answered / n,
    }


def main() -> None:
    data = json.loads(DATA.read_text())
    intents, strategies = data["intents"], data["strategies"]

    RESULTS.mkdir(exist_ok=True)
    MEMORY_PATH.unlink(missing_ok=True)  # clean slate, reproducible runs
    memory = GuardMemory(MEMORY_PATH)
    probe = SecurityProbe(memory)
    rng = random.Random(SEED)
    queues = [rng.sample(strategies, len(strategies)) for _ in intents]
    repeat_attackers = [new_attacker(queue) for queue in queues]
    # Same per-intent queues: the fresh user replays the round-1 attacks.
    fresh_attackers = [new_attacker(queue) for queue in queues]
    print(
        f"security-probe demo: {len(intents)} intents x {len(strategies)} "
        f"strategies, {ROUNDS} rounds as {REPEAT_USER!r} + 1 transfer round "
        f"as {FRESH_USER!r}"
    )

    per_round: list[dict] = []
    all_attempts: list[dict] = []
    for round_no in range(1, ROUNDS + 1):
        attempts = run_round(probe, intents, repeat_attackers, REPEAT_USER)
        all_attempts += attempts
        summary = summarize_round(round_no, REPEAT_USER, attempts)
        per_round.append(summary)
        print(
            f"round {summary['round']} ({REPEAT_USER}): {summary['attempts']} "
            f"attempts, blocked {summary['blocked']} "
            f"({summary['block_rate']:.1%}), flagged {summary['flagged']}, "
            f"answered {summary['answered']} ({summary['attack_success_rate']:.1%})"
        )

    attempts = run_round(probe, intents, fresh_attackers, FRESH_USER, learn=False)
    all_attempts += attempts
    summary = summarize_round("transfer", FRESH_USER, attempts)
    per_round.append(summary)
    print(
        f"round transfer ({FRESH_USER}): {summary['attempts']} attempts, "
        f"blocked {summary['blocked']} ({summary['block_rate']:.1%}), "
        f"flagged {summary['flagged']}, "
        f"answered {summary['answered']} ({summary['attack_success_rate']:.1%})"
    )

    catches = {
        trigger: sum(a["trigger"] == trigger for a in all_attempts)
        for trigger in ("repeat_offender", "any_user")
    }

    benign = []
    for prompt in data["benign_lookalikes"]:
        verdict = probe.assess(prompt, BENIGN_USER)
        benign.append(
            {
                "prompt": prompt,
                "action": verdict.action,
                "score": round(verdict.score, 4),
            }
        )
    n_benign_passed = sum(b["action"] != "block" for b in benign)
    benign_pass_rate = n_benign_passed / len(benign)
    print(
        f"benign lookalikes: {n_benign_passed}/{len(benign)} passed "
        f"(pass rate {benign_pass_rate:.2f})"
    )

    out = {
        "experiment": "security-probe learning curve (DARWIN-Guard-style incident memory)",
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "data": str(DATA),
        "config": {
            "rounds": ROUNDS,
            "seed": SEED,
            "repeat_user": REPEAT_USER,
            "fresh_user": FRESH_USER,
            "attacker": (
                "per intent and round: retry first known-good strategy, then "
                "explore the next strategy in a per-intent shuffled queue"
            ),
            "thresholds": {
                "block": probe.block_threshold,
                "repeat_offender": probe.repeat_offender_threshold,
                "flag": probe.flag_threshold,
            },
            "mock_target": "answers iff strategy in intent.effective_strategies",
        },
        "memory_path": str(MEMORY_PATH),
        "per_round": per_round,
        "catches": catches,
        "benign": {
            "n": len(benign),
            "passed": n_benign_passed,
            "pass_rate": benign_pass_rate,
            "results": benign,
        },
        "memory": {
            "incidents": len(memory),
            "successful": len(memory.query(succeeded=True)),
        },
        "attempts": all_attempts,
    }
    json_path = RESULTS / "security_probe_demo.json"
    json_path.write_text(json.dumps(out, indent=2))

    lines = [
        "security-probe demo — inference-time guard with persistent incident memory",
        f"date: {out['date']}",
        f"data: {DATA} ({len(intents)} intents, {len(strategies)} strategies, "
        f"{len(benign)} benign lookalikes)",
        f"thresholds: block {probe.block_threshold}, repeat-offender "
        f"{probe.repeat_offender_threshold}, flag {probe.flag_threshold}",
        "",
        f"{'round':>9} {'user':>9} {'attempts':>8} {'blocked':>7} {'flagged':>7} "
        f"{'answered':>8} {'block_rate':>10} {'success_rate':>12}",
    ]
    for row in per_round:
        lines.append(
            f"{str(row['round']):>9} {row['user_id']:>9} {row['attempts']:>8} "
            f"{row['blocked']:>7} {row['flagged']:>7} {row['answered']:>8} "
            f"{row['block_rate']:>10.3f} {row['attack_success_rate']:>12.3f}"
        )
    lines += [
        "",
        f"blocks by trigger: repeat-offender {catches['repeat_offender']}, "
        f"any-user {catches['any_user']}",
        f"benign lookalikes: {n_benign_passed}/{len(benign)} passed "
        f"(pass rate {benign_pass_rate:.2f})",
    ]
    for b in benign:
        if b["action"] != "allow":
            lines.append(f"  {b['action']:>5} ({b['score']:.3f}) {b['prompt'][:80]}")
    lines += [
        "",
        f"memory: {MEMORY_PATH} ({len(memory)} incidents, "
        f"{len(memory.query(succeeded=True))} successful)",
    ]
    txt_path = RESULTS / "security_probe_demo.txt"
    txt_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {json_path}\nwrote {txt_path}\nwrote {MEMORY_PATH}")


if __name__ == "__main__":
    main()
