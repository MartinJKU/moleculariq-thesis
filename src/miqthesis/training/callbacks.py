from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

try:
    from transformers import TrainerCallback
except ImportError:
    class TrainerCallback:  # type: ignore[no-redef]
        """Lightweight fallback so utility imports do not require training extras."""


class ResourceLoggingCallback(TrainerCallback):
    """Transformers-compatible callback for reproducibility and efficiency logs."""

    def __init__(self, output_path: str | Path):
        self.output_path = Path(output_path)
        self.started = time.perf_counter()

    def on_log(self, args: Any, state: Any, control: Any, logs: dict | None = None, **_: Any):
        elapsed = time.perf_counter() - self.started
        tokens_seen = getattr(state, "num_input_tokens_seen", None)
        samples_seen = (
            state.global_step
            * getattr(args, "train_batch_size", 0)
            * getattr(args, "gradient_accumulation_steps", 1)
        )
        payload = dict(logs or {})
        payload.update(
            {
                "step": state.global_step,
                "wallclock_seconds": elapsed,
                "tokens_seen": tokens_seen,
                "samples_seen": samples_seen,
                "tokens_per_second": (
                    tokens_seen / elapsed if tokens_seen is not None and elapsed else None
                ),
                "samples_per_second": samples_seen / elapsed if elapsed else None,
                "gpu_count": getattr(args, "n_gpu", 1),
            }
        )
        if "reward" in payload and "mean_reward" not in payload:
            payload["mean_reward"] = payload["reward"]
        if "reward_std" in payload and "std_reward" not in payload:
            payload["std_reward"] = payload["reward_std"]
        if "completion_length" in payload and "completion_length_mean" not in payload:
            payload["completion_length_mean"] = payload["completion_length"]
        try:
            import torch

            payload["gpu_memory_allocated_bytes"] = (
                torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            )
        except ImportError:
            payload["gpu_memory_allocated_bytes"] = 0
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        return control
