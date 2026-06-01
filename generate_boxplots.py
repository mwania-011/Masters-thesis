import pandas as pd
import matplotlib.pyplot as plt
import os

def load_seed_values(base_path, metric):
    values = []

    for i in range(1, 6):
        path = os.path.join(base_path, f"multiseed_enfr_seed{i}", "final_summary.csv")
        df = pd.read_csv(path)

        baseline = df[df["Model"] == "baseline"][metric].values[0]
        geo = df[df["Model"] == "geo"][metric].values[0]
        full = df[df["Model"].str.contains("full")][metric].values[0]

        values.append([baseline, geo, full])

    return list(zip(*values))  # separate per model

# Load Top-1 values
baseline, geo, full = load_seed_values("tier1_results", "Top1")

data = [baseline, geo, full]

plt.figure(figsize=(6, 4))
plt.boxplot(data, labels=["Baseline", "Geo", "Full"])

plt.ylabel("Top-1 Accuracy")
plt.title("Distribution Across Seeds (en-fr)")

plt.grid(True, linestyle="--", alpha=0.5)

plt.savefig("Fig_Boxplot_Top1_enfr.png", dpi=300)
plt.close()