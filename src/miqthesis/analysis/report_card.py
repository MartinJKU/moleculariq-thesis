from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from miqthesis.constants import SYSTEM_PROMPT
from miqthesis.training.utils import load_yaml


def _training_config(model_id: str) -> tuple[Path | None, dict]:
    candidates = {
        "sft_count": Path("configs/sft/qwen05_count.yaml"),
        "sft_index": Path("configs/sft/qwen05_index.yaml"),
        "sft_generation": Path("configs/sft/qwen05_generation.yaml"),
        "sft_multitask": Path("configs/sft/qwen05_multitask.yaml"),
        "grpo_format": Path("configs/grpo/qwen05_format_reward.yaml"),
        "grpo_verifier": Path("configs/grpo/qwen05_verifier_reward.yaml"),
        "grpo_verifier_ablation": Path(
            "configs/grpo/qwen05_verifier_curriculum.yaml"
        ),
    }
    path = candidates.get(model_id)
    return (path, load_yaml(path)) if path and path.exists() else (None, {})


def _gpu_hours(model_id: str) -> float | None:
    paths = [
        Path("results/raw/training_logs") / f"{model_id}.jsonl",
        Path("results/raw/grpo_logs") / f"{model_id}.jsonl",
    ]
    for path in paths:
        if path.exists():
            log = pd.read_json(path, lines=True)
            if "wallclock_seconds" in log and not log.empty:
                return float(log["wallclock_seconds"].max()) / 3600
    return None


def generate(
    model_id: str,
    tables_dir: Path,
    output_dir: Path,
    models_path: Path = Path("configs/models.yaml"),
) -> Path:
    leaderboard = pd.read_csv(tables_dir / "main_leaderboard.csv")
    rows = leaderboard[leaderboard["model_id"] == model_id]
    if rows.empty:
        raise ValueError(f"No aggregate metrics for {model_id}")
    task_table = pd.read_csv(tables_dir / "task_type_breakdown.csv")
    task_rows = task_table[task_table["model_id"] == model_id]
    model_config = load_yaml(models_path)["models"][model_id]
    training_path, training = _training_config(model_id)
    decision_path = Path("results/validation/model_size_decision.json")
    decision = (
        json.loads(decision_path.read_text(encoding="utf-8"))
        if decision_path.exists()
        else {}
    )
    lines = [
        f"# Model report card: {model_id}",
        "",
        f"- Model path: `{model_config['model_path']}`",
        f"- Base model: `{training.get('model_name_or_path', model_config['model_path'])}`",
        f"- Model type: {model_config.get('type', 'unknown')}",
        f"- Full-parameter training: {training.get('full_parameter_finetuning', 'n/a')}",
        f"- Training config: `{training_path}`" if training_path else "- Training config: baseline",
        f"- Training data: `{training.get('train_file', 'n/a')}`",
        f"- Reward function: {training.get('reward_type', 'n/a')}",
        f"- Selected size: `{decision.get('next_model', 'not decided')}`",
        f"- Escalation triggered: {decision.get('escalate', 'not decided')}",
        f"- System prompt: `{SYSTEM_PROMPT}`",
        "- Extraction function: pinned MolecularIQ evaluator extractor",
        "",
    ]
    for _, row in rows.iterrows():
        lines.extend(
            [
                f"## Protocol {row['protocol']}",
                "",
                f"- avg_accuracy: {row['avg_accuracy']:.4f}",
                f"- pass@1: {row['pass_at_1']:.4f}",
                f"- pass@3: {row['pass_at_3']:.4f}",
                f"- valid JSON rate: {row['valid_json_rate']:.4f}",
                "",
            ]
        )
    lines.extend(["## Task breakdown", "", task_rows.to_markdown(index=False), ""])
    if not task_rows.empty:
        worst = task_rows.sort_values("avg_accuracy").iloc[0]
        lines.extend(
            [
                "## Known failure modes",
                "",
                f"- Lowest-scoring task family: `{worst['task_type']}` "
                f"(avg_accuracy={worst['avg_accuracy']:.4f}).",
                "",
            ]
        )
    checkpoint = Path("checkpoints/eval") / model_id
    generation = checkpoint / "generation_config.json"
    if generation.exists():
        lines.extend(
            [
                "## Frozen decoding configuration",
                "",
                "```json",
                json.dumps(json.loads(generation.read_text()), indent=2),
                "```",
                "",
            ]
        )
    manifests = sorted(
        Path("results/raw/lm_eval").glob(f"protocol_*/{model_id}/run_manifest.json")
    )
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        label = manifest["evaluation"].get(
            "protocol_label", manifest["evaluation"]["protocol"]
        )
        lines.extend(
            [
                f"## Evaluation command: {label}",
                "",
                "```text",
                " ".join(manifest["command"]),
                "```",
                "",
            ]
        )
    gpu_hours = _gpu_hours(model_id)
    lines.extend(
        [
            "## Efficiency",
            "",
            f"- GPU hours: {gpu_hours:.3f}" if gpu_hours is not None else "- GPU hours: unavailable",
            "",
        ]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{model_id}.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--tables_dir", default="results/tables")
    parser.add_argument("--output_dir", default="results/report_cards")
    parser.add_argument("--models", default="configs/models.yaml")
    args = parser.parse_args()
    print(
        generate(
            args.model_id,
            Path(args.tables_dir),
            Path(args.output_dir),
            Path(args.models),
        )
    )


if __name__ == "__main__":
    main()
