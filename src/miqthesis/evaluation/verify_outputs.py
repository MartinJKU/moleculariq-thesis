from __future__ import annotations

from typing import Any

from miqthesis.training.rewards import compare_json, parse_json_answer


def verify_prediction(
    prediction: str,
    target: Any,
    task_type: str,
    constraints: Any = None,
) -> dict[str, Any]:
    parsed = parse_json_answer(prediction)
    output = {
        "prediction_json": parsed,
        "valid_json": int(parsed is not None),
        "exact": 0.0,
        "field_accuracy": 0.0,
        "valid_smiles": 0.0,
        "constraint_satisfaction": 0.0,
    }
    if "generation" in task_type or "constraint" in task_type:
        smiles = parsed.get("smiles") if isinstance(parsed, dict) else None
        try:
            from rdkit import Chem

            output["valid_smiles"] = float(
                isinstance(smiles, str) and Chem.MolFromSmiles(smiles) is not None
            )
        except ImportError:
            output["valid_smiles"] = 0.0
        if output["valid_smiles"]:
            try:
                from moleculariq_core import evaluate_answer

                details = evaluate_answer(
                    task_type="constraint_generation",
                    predicted=parsed,
                    constraints=constraints,
                    return_details=True,
                )
                if isinstance(details, dict):
                    score = float(details.get("reward", details.get("score", 0.0)))
                    satisfied = details.get("details", [])
                    if satisfied:
                        fraction = sum(
                            bool(item.get("satisfied")) for item in satisfied
                        ) / len(satisfied)
                    else:
                        fraction = score
                else:
                    score = float(details)
                    fraction = score
                output["exact"] = score
                output["field_accuracy"] = fraction
                output["constraint_satisfaction"] = fraction
            except (ImportError, TypeError, ValueError):
                pass
        return output

    comparison = compare_json(parsed, target) if parsed is not None else compare_json(None, target)
    output.update(
        {
            "exact": float(comparison["exact"]),
            "field_accuracy": float(comparison["field_accuracy"]),
            "comparison": comparison,
        }
    )
    return output

