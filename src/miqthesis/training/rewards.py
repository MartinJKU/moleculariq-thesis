from __future__ import annotations

import json
import math
import re
from typing import Any

ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)


def extract_answer_block(text: str) -> str | None:
    match = ANSWER_RE.search(text or "")
    return match.group(1).strip() if match else None


def parse_json_answer(text: str) -> dict | list | str | int | float | bool | None:
    block = extract_answer_block(text)
    candidate = block if block is not None else text
    if not isinstance(candidate, str) or not candidate.strip():
        return None
    try:
        return json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        return None


def normalize_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(key): normalize_json(value) for key, value in sorted(obj.items())}
    if isinstance(obj, list):
        normalized = [normalize_json(value) for value in obj]
        if all(isinstance(value, (str, int, float, bool, type(None))) for value in normalized):
            return sorted(normalized, key=lambda value: (type(value).__name__, str(value)))
        return normalized
    if isinstance(obj, float) and math.isfinite(obj) and obj.is_integer():
        return int(obj)
    return obj


def compare_json(pred: Any, target: Any) -> dict[str, Any]:
    if isinstance(target, str):
        try:
            target = json.loads(target)
        except json.JSONDecodeError:
            pass
    pred_norm = normalize_json(pred)
    target_norm = normalize_json(target)
    result = {
        "exact": int(pred_norm == target_norm),
        "field_accuracy": float(pred_norm == target_norm),
        "missing_keys": [],
        "extra_keys": [],
        "wrong_keys": [],
    }
    if isinstance(pred_norm, dict) and isinstance(target_norm, dict):
        pred_keys, target_keys = set(pred_norm), set(target_norm)
        common = pred_keys & target_keys
        result["missing_keys"] = sorted(target_keys - pred_keys)
        result["extra_keys"] = sorted(pred_keys - target_keys)
        result["wrong_keys"] = sorted(
            key for key in common if pred_norm[key] != target_norm[key]
        )
        correct = sum(pred_norm[key] == target_norm[key] for key in common)
        result["field_accuracy"] = correct / max(len(target_keys), 1)
    return result


def _basic_json_types_valid(obj: Any) -> bool:
    if isinstance(obj, dict):
        return all(
            isinstance(key, str) and _basic_json_types_valid(value)
            for key, value in obj.items()
        )
    if isinstance(obj, list):
        return all(_basic_json_types_valid(value) for value in obj)
    return obj is None or isinstance(obj, (str, int, float, bool))


def format_reward(text: str) -> float:
    block = extract_answer_block(text)
    parsed = parse_json_answer(text)
    reward = 0.25 * (block is not None)
    reward += 0.25 * (parsed is not None)
    reward += 0.25 * (parsed is not None and _basic_json_types_valid(parsed))
    trailing_ok = bool(
        block is not None
        and not re.sub(
            r"^\s*</answer>", "", text[ANSWER_RE.search(text).end(1):], flags=re.IGNORECASE
        ).strip()
    )
    reward += 0.25 * trailing_ok
    return float(reward)


def _completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
    return str(completion or "")


def _generation_details(predicted: Any, constraints: Any) -> tuple[float, float, float]:
    if not isinstance(predicted, dict):
        return 0.0, 0.0, 0.0
    smiles = predicted.get("smiles")
    if not isinstance(smiles, str):
        return 0.0, 0.0, 0.0
    try:
        from rdkit import Chem

        valid = float(Chem.MolFromSmiles(smiles) is not None)
    except ImportError:
        valid = 0.0
    if not valid:
        return 0.0, 0.0, 0.0
    try:
        from moleculariq_core import evaluate_answer

        constraint_list = constraints
        if isinstance(constraint_list, str):
            constraint_list = json.loads(constraint_list)
        if not constraint_list:
            score = evaluate_answer(
                task_type="constraint_generation",
                predicted=predicted,
                constraints=constraint_list,
                return_details=False,
            )
            score = float(score.get("reward", 0.0) if isinstance(score, dict) else score)
            return valid, score, float(score > 0)
        per_constraint = []
        for constraint in constraint_list:
            score = evaluate_answer(
                task_type="constraint_generation",
                predicted=predicted,
                constraints=[constraint],
                return_details=False,
            )
            score = float(score.get("reward", 0.0) if isinstance(score, dict) else score)
            per_constraint.append(float(score > 0))
        fraction = sum(per_constraint) / len(per_constraint)
        return valid, fraction, float(all(per_constraint))
    except (ImportError, TypeError, ValueError):
        return valid, 0.0, 0.0


def verifier_reward_value(
    text: str,
    target: Any,
    task_type: str,
    constraints: Any = None,
) -> float:
    parsed = parse_json_answer(text)
    has_tags = float(extract_answer_block(text) is not None)
    valid_json = float(parsed is not None)
    if "generation" in task_type or "constraint" in task_type:
        valid_smiles, fraction, success = _generation_details(parsed, constraints)
        return (
            0.10 * has_tags
            + 0.15 * valid_json
            + 0.25 * valid_smiles
            + 0.30 * fraction
            + 0.20 * success
        )
    comparison = compare_json(parsed, target) if parsed is not None else compare_json(None, target)
    return (
        0.10 * has_tags
        + 0.15 * valid_json
        + 0.25 * comparison["field_accuracy"]
        + 0.50 * comparison["exact"]
    )


def format_reward_function(completions: list[Any], **_: Any) -> list[float]:
    return [format_reward(_completion_text(completion)) for completion in completions]


def verifier_reward_function(
    completions: list[Any],
    target: list[Any],
    task_type: list[str],
    constraints: list[Any] | None = None,
    **_: Any,
) -> list[float]:
    constraints = constraints or [None] * len(completions)
    return [
        verifier_reward_value(_completion_text(completion), tgt, task, constraint)
        for completion, tgt, task, constraint in zip(
            completions, target, task_type, constraints, strict=True
        )
    ]
