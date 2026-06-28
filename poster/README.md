# Poster

A0 landscape poster for the MolecularIQ GRPO pass@k study.

## Files
- `poster.html` — edit text directly (name, institute, supervisors, repo URL, contact).
  Keeps the original H1–H3 framing; **H4 (elicitation vs expansion)** + **Fig 6 (pass@k)**
  + the "how to fix" note are added in the right column.
- `render.py` — renders `poster_A0.pdf` (print) + `poster_preview.png`.
- `pass_at_k.png` / `.pdf` — **placeholder figure**: a faithful reproduction of the
  observed run so the template renders. **Replace with your real figure** from
  Leonardo: `outputs/passk_report/pass_at_k.png` (same plotting code,
  `grpo-plot-passk`, real data). The numbers in the poster text match this run —
  re-check them against your final checkpoint before printing.
- `pass_at_k_crossover.csv` — per-task crossover / verdict table.

## You supply these figures (referenced by name, not committed here)
`fig1_overall_accuracy.png`, `fig2_by_tasktype.png`, `fig5_smiles_representation.png`,
`fig6_validity_vs_correctness.png`, `mol_indexed.svg`, `mol_caffeine_light.svg`.
The committed `poster_preview.png` / `poster_A0.pdf` show labeled `[ your figure ]`
boxes in their place.

## Render
```bash
pip install playwright
# this environment ships Chromium; point Playwright at it instead of "playwright install":
python render.py        # if Playwright can't find the browser, set executable_path
                        # to your chromium, e.g. /opt/pw-browsers/chromium-*/chrome-linux/chrome
```

## Before printing
1. Drop in the real `pass_at_k.png` and confirm the cited numbers (pass@1 values,
   crossover k) match it.
2. Fill the `[bracketed]` author / repo / contact placeholders.
3. Optional: swap the inline ring drawing for a real RDKit depiction.
