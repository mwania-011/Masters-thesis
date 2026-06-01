import os
import pandas as pd
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, required=True, help="e.g. enfr or enru")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    return parser.parse_args()

def main():
    args = parse_args()

    runs = [f"{args.prefix}_seed{s}" for s in args.seeds]
    base_dir = "tier1_results"
    metrics = [
        "Top-1", "Top-5", "MeanRank", "MedianRank",
        "MarginMean", "NeighborOverlap@5", "Uniformity"
    ]

    all_rows = []

    for run in runs:
        path = os.path.join(base_dir, run, "final_summary.csv")
        df = pd.read_csv(path)
        df["Run"] = run
        all_rows.append(df)

    full_df = pd.concat(all_rows, ignore_index=True)
    agg_df = full_df.groupby("Experiment")[metrics].agg(["mean", "std"]).reset_index()

    agg_df.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in agg_df.columns.values
    ]

    out_path = os.path.join(base_dir, f"{args.prefix}_multiseed_summary.csv")
    agg_df.to_csv(out_path, index=False)

    print(f"\nSaved aggregated results to: {out_path}\n")
    print(agg_df.to_string(index=False))

if __name__ == "__main__":
    main()