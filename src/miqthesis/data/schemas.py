from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from miqthesis.constants import SYSTEM_PROMPT


def first_present(row: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        value = row.get(name)
        if value is not None and value != "":
            return value
    return default


def json_text(value: Any) -> str:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
        return json.dumps(parsed, sort_keys=True, ensure_ascii=False)
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def normalize_task_type(value: Any) -> str:
    text = str(value or "").lower()
    if "constraint" in text or "generation" in text:
        return "generation"
    if "index" in text or "indices" in text or "attribution" in text:
        return "index"
    if "count" in text:
        return "count"
    return text or "unknown"


def stable_uid(row: Mapping[str, Any]) -> str:
    explicit = first_present(row, "uid", "id", "question_id")
    if explicit is not None:
        return str(explicit)
    material = "|".join(
        str(first_present(row, name, default=""))
        for name in ("question", "prompt", "target", "original_smiles", "smiles")
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]


@dataclass(frozen=True)
class NormalizedExample:
    uid: str
    question: str
    target: str
    task_type: str
    features: str
    constraints: Any
    original_smiles: str | None
    complexity_bin: str
    multi_task_load: int
    is_randomized: bool
    is_kekulized: bool
    metadata: dict[str, Any]

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "NormalizedExample":
        known = {
            "uid", "id", "question_id", "question", "prompt", "target", "answer",
            "reference_answer", "task_type", "task", "category", "features", "feature",
            "supercategory", "constraints", "constraint", "original_smiles", "smiles",
            "canonical_smiles", "complexity_bin", "complexity", "multi_task_load",
            "multitask_load", "num_tasks", "is_randomized", "randomized",
            "is_kekulized", "kekulized",
        }
        return cls(
            uid=stable_uid(row),
            question=str(first_present(row, "question", "prompt", default="")),
            target=json_text(first_present(row, "target", "answer", "reference_answer", default={})),
            task_type=normalize_task_type(
                first_present(row, "task_type", "task", "category", default="")
            ),
            features=str(
                first_present(row, "features", "feature", "supercategory", default="unknown")
            ),
            constraints=first_present(row, "constraints", "constraint"),
            original_smiles=first_present(
                row, "original_smiles", "smiles", "canonical_smiles"
            ),
            complexity_bin=str(
                first_present(row, "complexity_bin", "complexity", default="unknown")
            ),
            multi_task_load=int(
                first_present(row, "multi_task_load", "multitask_load", "num_tasks", default=1)
                or 1
            ),
            is_randomized=bool(
                first_present(row, "is_randomized", "randomized", default=False)
            ),
            is_kekulized=bool(
                first_present(row, "is_kekulized", "kekulized", default=False)
            ),
            metadata={key: value for key, value in row.items() if key not in known},
        )

    def to_sft(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["messages"] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self.question},
            {"role": "assistant", "content": f"<answer>{self.target}</answer>"},
        ]
        return payload

    def to_grpo(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["prompt"] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self.question},
        ]
        return payload

