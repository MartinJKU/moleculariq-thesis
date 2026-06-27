# MolecularIQ thesis poster — hypotheses, experiments, and plots

A 2-day, single-A100 (Leonardo) plan for a poster on **GRPO / RLVR for chemistry
reasoning** that has verifiable scientific value beyond "RL helped a small model."

> The headline runs and the new `pass@k` tooling live in the sibling repo
> **`multi-task-chemistry-rl`** (fast GRPO directly from the instruct base — no
> SFT / data-prep critical path). This thesis repo holds the strict SFT-vs-GRPO
> comparison and is the fallback / extension.

---

## 1. The framing (why this is science, not a demo)

A 0.5B model plus a **perfect symbolic verifier** (`moleculariq_core.evaluate_answer`)
is not a weakness — it is the ideal testbed for the most-debated open question in
RL-for-reasoning, which big RLVR models (e.g. ether0, 24B) cannot probe cheaply:

> **Does RLVR (GRPO) *teach* a model new capability, or only *elicit and sharpen*
> capability the base model already had — and does either form *transfer* to
> tasks the model never trained on?**

The verifier makes correctness exact and free, so we can sample hundreds of
completions per item and compute `pass@k` out to large `k` across many tasks —
the exact instrument needed to answer the question.

---

## 2. Hypotheses (one coherent story: H1 × H4, with H2 as a control)

**H4 — what multitask GRPO buys (transfer / compositional generalization)**
- **H4a** A single pooled multitask GRPO model matches or beats per-task
  specialists in-distribution (positive transfer, no catastrophic interference).
- **H4b** It generalizes to **held-out task types** measured by pass@1:
  - *composition*: trained on single-property counts/indices → held-out
    `multi_count` / `multi_index` (compose primitives it learned separately);
  - *analysis → synthesis*: trained to **count** rings/carbons → held-out
    `constraint_generation` (**generate** molecules hitting a target count);
  - *unseen property*: trained on ring/aromatic/… counts → held-out `hba_count`.

**H1 — the lens that makes H4 rigorous (elicitation vs. expansion via pass@k)**
- **H1a (in-distribution)** GRPO raises pass@1 but the base model **catches up at
  large k** — reproducing Yue et al. (2025) in a new domain (chemistry perception).
- **H1b (held-out — the headline)** On held-out tasks, **if** the GRPO model's
  pass@k stays above base even at large k → transfer is **genuine expansion**;
  **if** base catches up → the apparent transfer was only **elicitation**.
  *This applies the elicitation/expansion lens to **transfer**, which the
  literature has not — that is the novel contribution.*

**H2 — control (optional, isolates the active ingredient)**
- base → **format-only** GRPO → **verifier** GRPO. If format-only ≈ base and only
  the verifier model jumps, the gain is causally the **verifiable reward**, not
  extra on-policy compute or format pressure.

Each sub-claim is falsifiable and **interesting whichever way it falls.**

---

## 3. The train / held-out split (already wired up)

Train ONE pooled GRPO model on single-property **count + index** only:

| Trained (single-property)                                   | Held out (never trained)                          | Probes which claim |
|-------------------------------------------------------------|---------------------------------------------------|--------------------|
| `sc_ring_count`, `sc_aromatic_ring`, `sc_fused_ring`        | `mc_topology` (= compose those three)             | H4b composition    |
| `sc_carbon_atom`, `sc_hetero_atom`                          | `cg_carbon_atom` (count → generate)               | H4b synthesis      |
| `si_ring`, `si_aromatic_ring`                               | `mi_ring_aromatic` (= compose those two)          | H4b composition    |
| (counting in general)                                       | `sc_hba` (hba_count never trained)                | H4b unseen prop    |

Configs (in `multi-task-chemistry-rl/configs/multitask/`):
- `miq_transfer.yaml` — pooled **training** dataset (train tasks only)
- `miq_transfer_train.yaml` — GRPO **training** (warm-start from instruct base)
- `miq_transfer_format_train.yaml` — **H2** format-only control (optional)
- `miq_transfer_probe.yaml` — **pass@k probe** set (in-dist + all held-out)

Eval molecules come from the held-out tail of the pool (`split=test`), disjoint
from training molecules — so even in-distribution probes are on unseen molecules.

---

## 4. Commands (run in `multi-task-chemistry-rl`)

```bash
# --- login node (needs internet): build the training dataset once ---
grpo-preprocess-multitask --config configs/multitask/miq_transfer.yaml --overwrite

# --- compute node: train + pass@k + plot (one SLURM job) ---
sbatch slurm/transfer_passk.slurm
#   1) grpo-train  --config configs/multitask/miq_transfer_train.yaml
#   2) grpo-eval-passk --config .../miq_transfer_probe.yaml --model outputs/miq-transfer-grpo
#         (auto-evaluates the base model as the 'base' reference too)
#   3) grpo-plot-passk --passk-dir outputs/passk_eval --out-dir outputs/passk_report

# --- OPTIONAL H2 control: a second, format-only model, then re-probe + re-plot ---
grpo-train --config configs/multitask/miq_transfer_format_train.yaml
grpo-eval-passk --config configs/multitask/miq_transfer_probe.yaml \
    --model outputs/miq-transfer-grpo-format --model-label grpo_format --no-baseline
grpo-plot-passk --passk-dir outputs/passk_eval --out-dir outputs/passk_report

# --- OPTIONAL H4a specialists (drop if short on time): train 2-3 single tasks ---
#   grpo-train --config configs/single_task/miq_sc_ring_count.yaml   # etc.
```

`grpo-eval-passk` knobs: `--n-completions` (256 → larger k, tighter CI; drop to
128/64 to save wall time), `--num-samples` (items/task), `--temperature` (0.8).

---

## 5. Poster panels (what to actually show)

1. **Headline — pass@k curves, base vs GRPO, faceted by task** (in-distribution
   vs held-out). `outputs/passk_report/pass_at_k.{png,pdf}`. Crossover ⇒
   elicitation; sustained gap ⇒ expansion. This single figure carries H1+H4b.
2. **Crossover table** `pass_at_k_crossover.csv`: per task, pass@1(base/GRPO) and
   the crossover k with the elicitation/expansion verdict.
3. **Transfer bar** — pass@1: base vs GRPO across in-dist + held-out tasks
   (the H4b "it generalizes" panel). From the same summaries.
4. **(H4a, optional)** specialists vs multitask pass@1 bars.
5. **(H2, optional)** base vs format-GRPO vs verifier-GRPO pass@1 bars.
6. **(Diagnostics, free)** `distinct_answer_rate` / `valid_smiles_rate` from eval
   — guards against reward-hacking (a flat-answer collapse on set/index tasks).

---

## 6. 2-day timeline (de-risked — minimum viable result = base + 1 model)

**Day 1**
- AM: build the train dataset on a login node; `sbatch slurm/transfer_passk.slurm`.
  Smoke-test first with tiny `--max-steps 20` if unsure of the env.
- PM: while the model trains, draft poster text/figure stubs. If time, kick off
  the H2 format-only model and/or 2 specialists on the same or a second node.

**Day 2**
- AM: pass@k probe (base + trained) → `pass_at_k.{png,pdf}` + crossover CSV.
- PM: assemble panels; write the interpretation; add optional H2/H4a panels if
  their runs finished.

**Fallbacks (each still a complete poster):**
- Only base + 1 GRPO model finishes → H1a + H1b + H4b stand on their own.
- Training slow → cut `--max-steps`, fewer probe tasks, `--n-completions 64`.
- pass@k too slow on HF `generate` → fewer items (`--num-samples 60`) or fewer k.

---

## 7. How to state the result (interpretation guide)

- **pass@1 up, in-dist crossover at small/medium k** → "GRPO sharpens the base
  model's existing chemistry knowledge (elicitation), consistent with Yue et al."
- **held-out pass@k stays above base at all k** → "multitask GRPO induced a
  reusable, *transferable* primitive — evidence of expansion, not just
  reweighting." (Strongest possible poster claim.)
- **held-out crossover** → "transfer measured by pass@1 is largely elicitation;
  pass@k reveals the base already covered these solutions." (Also a real,
  publishable-flavored finding — and a caution about pass@1 transfer claims.)
- If **H2** included and format-only ≈ base → "the verifiable reward is the
  causal ingredient."

---

## 8. References (for the poster)

- Yue et al., *Does Reinforcement Learning Really Incentivize Reasoning Capacity
  in LLMs Beyond the Base Model?*, 2025 — arXiv:2504.13837 (the pass@k crossover).
- Wen et al., *RLVR Implicitly Incentivizes Correct Reasoning in Base LLMs*
  (CoT-Pass@K), 2025 — arXiv:2506.14245 (the counter-argument).
- Narayanan et al. (FutureHouse), *ether0: a scientific reasoning model for
  chemistry*, 2025 — arXiv:2506.17238 (your seminar; RLVR for chemistry at 24B).
- Chen et al., *Evaluating Large Language Models Trained on Code*, 2021 —
  arXiv:2107.03374 (the unbiased pass@k estimator used here).
- DeepSeek-AI, *DeepSeek-R1* / GRPO, 2025 (the RL algorithm).
- MolecularIQ: `ml-jku/moleculariq-trainPool`, `ml-jku/moleculariq-core`.
