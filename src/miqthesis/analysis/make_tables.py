from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    input_dir, output_dir = Path(args.input_dir), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    leaderboard = pd.read_csv(input_dir / "main_leaderboard.csv")
    controlled = leaderboard[leaderboard["protocol"] == "A"].copy()
    columns = ["model_id", "avg_accuracy", "pass_at_1", "pass_at_3", "valid_json_rate"]
    controlled[columns].to_latex(
        output_dir / "main_leaderboard.tex",
        index=False,
        float_format="%.3f",
        caption=(
            "Controlled Protocol A results. avg\\_accuracy is the headline metric; "
            "pass@1/pass@3 are repeat-based metrics."
        ),
        label="tab:main-leaderboard",
    )


if __name__ == "__main__":
    main()

