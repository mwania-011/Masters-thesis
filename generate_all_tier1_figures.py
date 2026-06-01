import os
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = "fig_panel_e_cdf"
os.makedirs(OUT_DIR, exist_ok=True)

# Example lambda_topo (x-axis)
lambda_topo = np.array([0.0001, 0.0002, 0.0003, 0.0004, 0.0005])

# Top-1 means for 4 languages
top1_enfr = np.array([0.9228, 0.9226, 0.9230, 0.9235, 0.9252])
top1_enru = np.array([0.9139, 0.9215, 0.9208, 0.9220, 0.9179])
top1_enit = np.array([0.9177, 0.9200, 0.9210, 0.9225, 0.9233])
top1_enes = np.array([0.7936, 0.7950, 0.7970, 0.7985, 0.8000])

# 5-seed std for shading
std_enfr = np.array([0.0025, 0.0020, 0.0022, 0.0020, 0.0025])
std_enru = np.array([0.0028, 0.0030, 0.0027, 0.0031, 0.0029])
std_enit = np.array([0.0030, 0.0028, 0.0025, 0.0029, 0.0027])
std_enes = np.array([0.0022, 0.0024, 0.0025, 0.0023, 0.0026])

colors = {
    "en-fr": "#1f77b4",
    "en-ru": "#ff7f0e",
    "en-it": "#2ca02c",
    "en-es": "#d62728"
}

plt.figure(figsize=(7, 4))

def plot_cdf_style(x, y, std, color, label):
    # Stepwise (horizontal then vertical)
    plt.step(x, y, where='post', color=color, linewidth=2, label=label)
    # Error shading
    y_lower = y - std
    y_upper = y + std
    plt.fill_between(x, y_lower, y_upper, step='post', color=color, alpha=0.2)

# Draw all languages
plot_cdf_style(lambda_topo, top1_enfr, std_enfr, colors["en-fr"], "en-fr")
plot_cdf_style(lambda_topo, top1_enru, std_enru, colors["en-ru"], "en-ru")
plot_cdf_style(lambda_topo, top1_enit, std_enit, colors["en-it"], "en-it")
plot_cdf_style(lambda_topo, top1_enes, std_enes, colors["en-es"], "en-es")

plt.xscale("log")
plt.xticks(lambda_topo, ["0.0001", "0.0002", "0.0003", "0.0004", "0.0005"])
plt.xlabel(r"$\lambda_{topo}$")
plt.ylabel("Top-1 Accuracy")
plt.title("Top-1 Accuracy vs Topology Weight (CDF-style)")
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend(frameon=False, loc="lower right")
plt.tight_layout()

save_path = os.path.join(OUT_DIR, "Panel_e_CDF_style.png")
plt.savefig(save_path, dpi=600, bbox_inches="tight")
plt.show()
print(f"Saved figure: {save_path}")