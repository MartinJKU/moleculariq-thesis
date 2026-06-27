# Poster figures — training architecture

Two renderings of the MolecularIQ training pipeline implemented in this repo,
derived directly from the code (not idealized). Each is provided as **SVG**
(edit in Inkscape/Illustrator), **PDF** (drop into a LaTeX/`tikzposter` poster),
and **PNG @ 200 dpi** (quick preview / slides). The Graphviz `.dot` sources are
the editable originals — change text/colors there and re-render.

| File | Use |
|------|-----|
| `training_architecture.{svg,pdf,png}` | Full, detailed figure (data → SFT → checkpoint gate → GRPO → eval, with the reward breakdown and invariants). Good as a main thesis figure or a large poster panel. |
| `training_architecture_compact.{svg,pdf,png}` | Landscape "hero" banner of the two-stage SFT → GRPO spine with the GRPO reward loop. Good as a poster header strip. |

Re-render after editing a `.dot`:

```bash
dot -Tsvg training_architecture.dot -o training_architecture.svg
dot -Tpdf training_architecture.dot -o training_architecture.pdf
dot -Tpng -Gdpi=300 training_architecture.dot -o training_architecture.png   # 300 dpi for print
```

## What the architecture is (source of truth)

Two-stage, **full-parameter** training of Qwen2.5 (size ladder 0.5B → 1.5B → 3B),
followed by a controlled evaluation.

- **Data construction** — `src/miqthesis/data/`
  - `download.py`: load MolecularIQ benchmark + train pool; RDKit-canonical
    leakage filter removes any benchmark molecule from the training pool.
  - `prepare_sft.py`: molecule-level deterministic train/val split (4% val,
    disjoint pools); symbolic task generation with `moleculariq-core` over three
    families — `count`, `index`, `generation`/constraint — with SMILES
    augmentation (randomized 25%, kekulized 25%) and 1–5 properties per question;
    balanced multitask interleave → 150k train / 5k val.
  - `prepare_grpo.py`: systematic prompt-only sample → 50k train / 2k val.
  - Chat format (`schemas.py`): `system / user / <answer>JSON</answer>`.

- **Stage 1 — SFT** — `src/miqthesis/training/train_sft.py`, `configs/sft/`
  - HF `Trainer`, full-parameter (PEFT/LoRA/quant/frozen rejected by
    `assert_full_parameter_config` + a runtime `requires_grad` check).
  - `CausalChatCollator`: applies the Qwen chat template; labels = `input_ids`
    with **only padding masked to -100** → causal-LM cross-entropy over the full
    chat sequence (not completion-only).
  - lr 1e-5 cosine, 2 epochs, bf16, gradient checkpointing, eff. batch
    2 × 32 = 64, max_seq 1024. Variants: count / index / generation / multitask.

- **Checkpoint gate** — `src/miqthesis/training/prepare_checkpoint.py`
  - Writes a canonical `generation_config.json` whose `eos_token_id` includes
    `<|im_end|>` (151645) so vLLM stops the assistant turn; rejects adapter
    checkpoints; validates load; enforces one identical generation config + chat
    template across all Protocol A checkpoints.

- **Stage 2 — GRPO** — `src/miqthesis/training/train_grpo.py`, `rewards.py`, `configs/grpo/`
  - TRL `GRPOTrainer`, initialized from the SFT checkpoint, full-parameter.
  - Per prompt: G = 4 completions (T 0.7, top_p 0.95, max 256 tok) →
    group-relative advantage with KL penalty β 0.02 to the frozen reference;
    lr 2e-6, 1 epoch, eff. batch 1 × 16 = 16.
  - Two reward variants (`rewards.py`):
    - **format**: answer tags / valid JSON / valid JSON types / clean trailing
      (4 × 0.25).
    - **verifier** (symbolic): count & index → tags + valid JSON + field
      accuracy + exact match; generation → tags + valid JSON + RDKit-valid SMILES
      + constraint-satisfaction fraction + full success, scored with
      `moleculariq_core.evaluate_answer`.
  - Reward stats logged per step to `results/raw/grpo_logs/`.

- **Stage 3 — Evaluation** — `src/miqthesis/evaluation/`, `configs/eval/`
  - Separate vLLM environment. Protocol A = controlled internal comparison
    (fixed decoding / prompt / extractor / repeats, all variants); Protocol B =
    leaderboard-comparable run of the best model with native settings. Headline
    metric `avg_accuracy` plus pass@1 / pass@3; size escalation decided from
    validation metrics only (`analysis/model_selection.py`).

- **Cross-cutting** — full-parameter invariant (config + runtime + checkpoint),
  molecule-level no-leakage, fixed seeds + `run_manifest.json` +
  `ResourceLoggingCallback` (tokens/s, GPU memory), offline mode on compute nodes.
