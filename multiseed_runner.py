import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


def run_one_seed(
    train_script: str,
    seed: int,
    lang_config: str,
    run_tag_prefix: str,
    python_exec: str = sys.executable,
) -> str:
    run_tag = f"{run_tag_prefix}_seed{seed}"
    cmd = [
        python_exec,
        train_script,
        "--seed", str(seed),
        "--lang_config", lang_config,
        "--run_tag", run_tag,
    ]

    print("\n" + "=" * 80)
    print(f"Running seed {seed} | lang={lang_config} | run_tag={run_tag}")
    print("=" * 80)
    print("Command:", " ".join(cmd))

    subprocess.run(cmd, check=True)
    return run_tag


def aggregate_runs(base_dir: str, run_tags: list[str], out_csv: str) -> pd.DataFrame:
    all_rows = []

    for run_tag in run_tags:
        summary_path = Path(base_dir) / run_tag / "final_summary.csv"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing final_summary.csv: {summary_path}")

        df = pd.read_csv(summary_path)
        df["RunTag"] = run_tag
        all_rows.append(df)

    full_df = pd.concat(all_rows, ignore_index=True)

    metric_cols = [
        "Top-1",
        "Top-5",
        "MeanRank",
        "MedianRank",
        "MarginMean",
        "NeighborOverlap@5",
        "Uniformity",
    ]
    missing = [c for c in metric_cols if c not in full_df.columns]
    if missing:
        raise ValueError(f"Missing required metric columns: {missing}")

    group_cols = ["Experiment"]
    agg = (
        full_df.groupby(group_cols)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )

    agg.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in agg.columns.to_flat_index()
    ]

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out_path, index=False)

    print("\n" + "=" * 80)
    print("Aggregated multi-seed results")
    print("=" * 80)
    print(agg.to_string(index=False))
    print(f"\nSaved aggregated CSV to: {out_path}")

    return agg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_script",
        type=str,
        default="train_structure_preserving_multilingual.py",
        help="Path to the training script",
    )
    parser.add_argument(
        "--lang_config",
        type=str,
        default="en-fr",
        help='Language pair, e.g. "en-fr" or "en-ru"',
    )
    parser.add_argument(
        "--run_tag_prefix",
        type=str,
        default="multiseed_enfr",
        help="Prefix for per-seed run directories",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="tier1_results",
        help="Base results directory",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="tier1_results/multiseed_enfr_summary.csv",
        help="Output aggregated CSV",
    )
    parser.add_argument(
        "--start_seed",
        type=int,
        default=1,
        help="First seed value",
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=5,
        help="Number of seeds to run",
    )
    args = parser.parse_args()

    run_tags = []
    for seed in range(args.start_seed, args.start_seed + args.num_seeds):
        run_tag = run_one_seed(
            train_script=args.train_script,
            seed=seed,
            lang_config=args.lang_config,
            run_tag_prefix=args.run_tag_prefix,
        )
        run_tags.append(run_tag)

    aggregate_runs(
        base_dir=args.base_dir,
        run_tags=run_tags,
        out_csv=args.out_csv,
    )


if __name__ == "__main__":
    main()