# Poster defense — Q&A prep

## ⚠️ Read first: TWO experiments, two settings — keep them straight

You ran two complementary experiments with different rewards and warm-starts.
Examiners will probe the seams, so be explicit about which is which.

**Experiment A — accuracy comparison (Fig 1), separate repo [REAL]:**
- Three independent models from Qwen2.5-0.5B: **base**, **SFT (from base)**, **GRPO (from base)**.
- GRPO reward = **binary exact-match only** (RDKit / MolecularIQ). Eval metric = exact-match accuracy.
- **Relabel "SFT+GRPO" → "GRPO".** GRPO is trained **from base, not from the SFT checkpoint**, so
  the conditions are base / (prompted) / SFT / GRPO — three *parallel* models, not a chain.
  The poster tape "binary correctness" is correct *for this experiment*.

**Experiment B — pass@k mechanism study (Fig 6 / H4), `multi-task-chemistry-rl` [REAL]:**
- GRPO from **`Qwen2.5-0.5B-Instruct`**, **shaped** reward (exact + Jaccard on index / numeric on
  counts + SMILES-validity), `miq_transfer_train.yaml`.
- pass@k to 256, n=256 samples/item, ~80–100 items/task, T=0.8, bootstrap CIs; plus a greedy
  diagnostic (accuracy / distinct-answer-rate / json-valid). One seed (42), ~400-step checkpoint.
- This is where the **over-coverage index reward-hacking** appears (a shaped-reward artifact).

**⚠️ Consistency caveat (be ready for this):** the Fig 1 GRPO (from *base*, *binary* reward) and the
pass@k GRPO (from *instruct*, *shaped* reward) are **different models**. So the pass@k mechanism
story characterizes Experiment B, not literally the Fig 1 bar. Safest: present them as **two studies**
("headline accuracy comparison" + "mechanism analysis"), or unify by running pass@k on the
Experiment-A GRPO. Also confirm "base" = the same Qwen2.5-0.5B variant everywhere and matches the
poster tape (base vs -Instruct).

**Still to confirm (likely placeholder):** Fig 3 (SMILES robustness) and Fig 5
(validity-vs-correctness) — defend only if you actually ran them; otherwise relabel "planned".

**Two index failure modes, one per experiment (a strength — use it):**
- *Experiment A (binary):* index exact-match ≈ 0 for all G → **zero group advantage → no gradient**
  (no learning signal at all).
- *Experiment B (shaped):* the Jaccard proxy gives signal but rewards **over-coverage** → the model
  learns wide ranges that overlap but never exact-match, dropping **below** base.
- Either way it's **not cleanly "0.5B can't index"**: the base samples correct sets at a nonzero rate
  (si_ring base pass@k ≈ 0.31 at k=256) and GRPO's outputs are valid JSON (json_valid=1.0). Lead with
  reward/initialization, not raw capacity.

---

## Core method (GRPO + verifier)

**Q: Why GRPO not PPO/DPO? What does group-relative advantage buy you? What happens when all G samples get the same reward?**
- **vs PPO:** PPO needs a learned value/critic network — a second model to train, extra
  memory, and notoriously unstable to fit on a 0.5B with sparse, spiky rewards. GRPO is
  **critic-free**: the baseline is the *group mean* over G completions of the same prompt,
  so advantage = (r − mean)/std within the group. Cheaper and more stable for verifiable rewards.
- **vs DPO:** DPO is offline preference learning — it needs pairwise preferences and never
  explores. We have a *programmatic scalar reward*, not preferences, and we specifically need
  **on-policy exploration** to discover correct trajectories. That's online RLVR → GRPO.
- **All-equal reward → zero gradient.** If all G completions score the same, advantage = 0 for
  every sample, so that prompt contributes **no policy-gradient signal** (only the KL term
  remains). On indexing the binary term is 0 for all G almost every step → no learning from it.
  **That is exactly why we added the shaped reward** — to keep within-group variance non-zero.
  The catch (see below) is that the shaped proxy we used is misaligned with exact-match.

**Q: Why binary exact-match and not shaped/partial (e.g., Jaccard)? Wouldn't partial credit give signal where indexing gives none?**
- **We tested both — and that's the finding.** Experiment A's GRPO used **binary only**, so on
  indexing all G completions score 0 → **no gradient** (your intuition is right: binary gives no
  signal there). Experiment B **added shaped Jaccard** precisely to inject signal — and it
  backfired: Jaccard rewards overlap, which is maximized by **over-covering** (gold {5–10} ⊂ pred
  {3–12} → Jaccard ≈ 0.6), so the model emitted wide ranges that overlap but never exact-match,
  dropping **below** base. So partial credit *did* give signal where binary gave none — but the
  **wrong** signal.
- **Takeaway:** the choice isn't binary-vs-shaped, it's that *naïve* shaping is misaligned with
  exact-match. **What we'd do now:** precision/size penalty (penalize |pred|>|gold|), anneal the
  shaped weight to 0, or gate shaped credit behind a competence threshold. Predicted effect:
  removes the over-coverage optimum; whether exact-match then rises depends on cold-start (below).

**Q: KL penalty / reference model? Temperature, group size G, sensitivity?**
- **KL:** `beta = 0.005` (TRL GRPO), penalizing divergence from a **frozen reference = the
  initial policy (the instruct base)**. Low beta = weak anchor → permits the drift that, on
  index, moved the policy below base. **G = 4** completions/prompt.
- **Temperature:** training used the TRL default **T = 1.0** (not separately tuned); evaluation
  used **T = 0.8** (same for base and GRPO). **Sensitivity to G, beta, T was not swept — a
  limitation.** G=4 is small, so advantage estimates are noisy; larger G and a beta sweep are
  the obvious robustness follow-ups.

---

## Elicitation vs. expansion (the most attackable claim)

**Q: pass@k is temperature-sensitive. Did you tune T separately for base and GRPO? Could entropy collapse (not a real frontier limit) explain GRPO's worse high-k?**
- We used the **identical** sampling config (T=0.8, top_p=0.95, top_k=50) for **both** models —
  the standard apples-to-apples protocol. We did **not** sweep temperature per model (limitation).
- **Honest concession:** yes, if GRPO collapsed entropy, its high-k pass@k partly reflects
  reduced diversity. But (a) that *is* part of "sharpening" — RLVR narrowing the distribution is
  the mechanism, not a confound that breaks the claim; (b) crucially, on **multi_count GRPO is
  above base at *every* k including k=256** — diversity collapse can only *lower* high-k pass@k,
  so it cannot explain an *upward* shift. The confound therefore can't account for the full
  picture, which strengthens the task-dependent reading.
- **The clean check I'd add:** report pass@k at *matched generation entropy*, or sweep T per
  model and compare frontiers. Name this as future work.

**Q: What k, how many samples? Is the high-k crossover significant?**
- k = 1…256, **n = 256 samples/item**, ~80–100 items/task, unbiased Chen et al. estimator,
  **95% bootstrap CIs over items** (the shaded bands). Where bands separate (ring_count crossover,
  multi_count gap, index gap) the difference is meaningful; where they overlap
  (constraint_generation ≈ tie) **we do not claim a difference.** Caveat: CIs capture item-level
  noise, **not** training-seed variance (single seed).

**Q: "Expansion vs elicitation" is a strong dichotomy — could it be task-dependent rather than global?**
- **Agree, and that's the contribution.** It *is* task-dependent: **expansion** on multi_count
  (above base at all k), **elicitation** on ring_count / hba (crossover), **regression/reward-hack**
  on indexing, **null** on constraint-generation. The headline isn't a single global verdict —
  it's a **regime taxonomy governed by reward reachability.** Own this framing; don't defend a
  global claim.

---

## Indexing failure

**Q: How do you know the index floor is capacity, not prompt format / index tokenization / parser? Does the base ever produce correct index sets, and at what rate?**
- It's **not the parser/format**: GRPO's index outputs are **valid JSON (json_valid = 1.0)** and
  parse fine — they're *wrong in content* (over-covering ranges), not malformed.
- The base **does** produce correct sets sometimes: si_ring base pass@k ≈ **0.08 at k=1, 0.31 at
  k=256** (nonzero ⇒ the task is reachable, just rare). GRPO drove this to ≈ 0.
- So the honest diagnosis is **reward misalignment + rarity**, not pure capacity. Capacity makes
  exact-match rare enough that the misaligned proxy dominates the gradient. I would **not** claim
  "scale is the prerequisite" — the data point to initialization + reward design first.

**Q (sharp): Atom indices depend on canonicalization. Is the indexing task even well-posed if RDKit can reindex on parsing?**
- **Well-posed**, because the index is defined as the atom's position in the **given input SMILES
  string**, and both the question and the gold answer are computed by RDKit from that *same*
  string the model sees. RDKit assigns atom indices in input order on parse; we never
  recanonicalize. A different SMILES for the same molecule would give a different (equally valid)
  index set, but since Q and gold share the exact string, the mapping is deterministic and
  unambiguous. The task is "index within this SMILES as written," not "index a canonical form."

**Q: Did you try a few SFT indexing demonstrations to seed correct trajectories, then GRPO? If not, why claim scale is the prerequisite?**
- **We did not** — so we **do not** claim scale is the prerequisite (correct the poster's "scale
  helps" note to "untested"). Because the base already samples correct sets occasionally, the
  predicted fix is exactly **cold-start SFT on RDKit-generated traces → GRPO** (give RL something
  to reinforce) plus the reward de-hack. That's the headline future experiment.

---

## Evaluation & benchmark validity

**Q: MolecularIQ is from your own lab — same leakage concern you level at others? Is the test set provably disjoint from Qwen's pretraining?**
- Train/eval are disjoint *within our pipeline* (eval = held-out trainPool tail, fixed seed). We
  **cannot prove** molecule-level disjointness from Qwen's pretraining corpus — SMILES strings
  may appear in web text. Three-part defense: (1) the task is **compute-not-recall** — "how many
  fused rings / which atoms are aromatic" is not a memorized fact even if the SMILES was seen;
  (2) the central claim is **relative** (base vs GRPO on identical data) — any shared pretraining
  exposure cancels and cannot explain the GRPO delta or the pass@k crossover; (3) the verifier
  computes ground truth, it isn't a lookup. Concede the disjointness limitation openly.

**Q: "Verifiable ≠ memorization" — but counting oxygens in common molecules could be recall. How do you separate reasoning from frequency-based recall?**
- Fair for single counts on common scaffolds. Mitigations: held-out molecules; multi_count and
  index/generation are far less memorizable; the comparison is relative. **Not done:** stratify
  accuracy by molecule frequency/commonness — name it as the clean test.

**Q: Greedy vs sampled — which is "the" accuracy?**
- Be explicit and consistent. **pass@1** = one sample at T=0.8 (e.g., hba 0.17); **greedy** = T=0
  point estimate (hba 0.05). They differ. Recommendation: report **greedy as the headline single-
  shot accuracy** (deterministic, reproducible) and pass@1/pass@k for the distributional picture.
  (Fig 1 is a placeholder, so fix this when you make the real one.)

---

## Generalization (H2 / Fig 3) — currently PLACEHOLDER

**Q: Did GRPO train on randomized SMILES? If yes, robustness is expected; if no, what's the mechanism?**
- The transfer run trained on trainPool SMILES **as given (no randomized-SMILES augmentation)**,
  so a robustness test *would* be a genuine (not circular) generalization probe — **but we have
  not run it.** Don't claim Fig 3 until you do. If you run it: retention under randomized/kekulized
  SMILES with no augmentation would be real evidence; collapse would be the expected failure.

**Q: How is "complexity" quantified, and is the shrinkage significant?**
- Not yet measured. Quantify by heavy-atom count / ring count / answer-set size; test with
  stratified CIs. As-is this is an unsupported placeholder claim — measure it or drop it.

---

## Statistical rigor

**Q: Single seed or multiple? Error bars? Are SFT-vs-GRPO differences significant? How much should anyone generalize?**
- **Fig 1 (Experiment A) is real** — base/SFT/GRPO exact-match accuracy. To claim SFT-vs-GRPO
  significance, report **error bars** (bootstrap CI over the test items at minimum; ideally ≥3
  training seeds). Confirm Experiment A's seed/step count — if it's a single seed, say so.
- **pass@k (Experiment B)** has **item-level bootstrap CIs** but **no seed/run variance**, on one
  0.5B and a partial (~400-step) checkpoint.
- Net: frame the work as a **controlled case study / mechanism demonstration**, not a broad
  empirical law — one model family, one benchmark, limited seeds.

**Q: Why 0.5B at all — isn't the conclusion just "small models can't index"?**
- 0.5B is a deliberate **methodological choice**: a small model + an exact verifier makes
  **exhaustive pass@k** (256 samples × ~100 items × many tasks × 2 models) affordable and the
  elicitation/expansion question cleanly testable — infeasible at 24B. The finding isn't "can't
  index"; it's the **regime taxonomy + the reward-hacking mechanism**, which is *model-agnostic*:
  it predicts the same proxy-hacking at any scale wherever exact reward is unreachable.

---

## "So what" / novelty

**Q: "RLVR sharpens reachable skills, can't reach unreached ones" is already known. What's novel?**
- Concede the broad statement is known (Yue et al. 2025). The novelty is:
  1. **Elicitation/expansion applied to transfer/held-out composition**, not just in-distribution
     — and finding genuine **compositional expansion** (multi_count above base at all k), which
     the in-distribution-only literature doesn't show.
  2. A **clean, mechanistic reward-hacking case study** in a fully-verifiable domain: the
     over-coverage "receipt" linking dense-proxy misalignment to capability *regression below base*.
  3. A **predictive framing** — regimes governed by reward reachability (reachable→elicit/expand;
     unreachable→hack/regress).
  4. The **chemistry-verifier RLVR testbed** itself: exact, cheap, exhaustive pass@k.
- Honest one-liner: *the generic claim is incremental; the chemistry-grounded mechanism + the
  transfer/composition result + the reward-hacking demonstration are the contribution.*
