# Poster

A0 landscape poster for the MolecularIQ GRPO pass@k study.

## Files
- `poster.html` — edit text directly (name, institute, supervisors, repo URL, contact).
- `render.py` — renders `poster_A0.pdf` (print) + `poster_preview.png`.
- `pass_at_k.png` / `.pdf` — **placeholder figure**: a faithful reproduction of the
  observed run so the template renders. **Replace with your real figure** from
  Leonardo: `outputs/passk_report/pass_at_k.png` (same plotting code,
  `grpo-plot-passk`, real data). The numbers in the poster text match this run —
  re-check them against your final checkpoint before printing.
- `pass_at_k_crossover.csv` — per-task crossover / verdict table.

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
