import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

# ============================================================
# Fig. 3 — Final IEEE-Ready High-Contrast Panel (1x4)
# ============================================================

lambda_vals = np.array([1e-4, 2e-4, 5e-4])

results = {
    "en-fr": {
        "Top1": [0.9228, 0.9225, 0.9250],
        "Top5": [0.9693, 0.9662, 0.9700],
        "Margin": [0.1775, 0.1783, 0.1814],
        "Uniformity": [-3.6000, -3.6260, -3.6270],
    },
    "en-ru": {
        "Top1": [0.9139, 0.9215, 0.9179],
        "Top5": [0.9599, 0.9648, 0.9652],
        "Margin": [0.1655, 0.1803, 0.1808],
        "Uniformity": [-3.6160, -3.6470, -3.6900],
    },
    "en-it": {
        "Top1": [0.9210, 0.9235, 0.9222],
        "Top5": [0.9638, 0.9647, 0.9640],
        "Margin": [0.1222, 0.1237, 0.1230],
        "Uniformity": [-1.9700, -1.9501, -1.9820],
    },
    "en-es": {
        "Top1": [0.7959, 0.7961, 0.7954],
        "Top5": [0.8848, 0.8834, 0.8829],
        "Margin": [0.0610, 0.0608, 0.0604],
        "Uniformity": [-1.9820, -1.9876, -1.9950],
    },
}

# -----------------------------
# Global style (IEEE-quality)
# -----------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.linewidth": 1.0,
    "savefig.bbox": "tight",
    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "axes.edgecolor": "black",
})

colors = {
    "en-fr": "#1f4ea8",
    "en-ru": "#6f6f6f",
    "en-it": "#2ca02c",
    "en-es": "#d62728",
}

linestyles = {
    "en-fr": "-",
    "en-ru": "--",
    "en-it": "-.",
    "en-es": ":",
}

markers = {
    "en-fr": "o",
    "en-ru": "o",
    "en-it": "s",
    "en-es": "D",
}
plt.rcParams.update({
    "font.family": "serif",

    # ---- FONT SIZES ----
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,

    # ---- FORCE BLACK ----
    "text.color": "black",
    "axes.labelcolor": "black",
    "axes.edgecolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",

    # ---- MAKE EVERYTHING BOLD ----
    "font.weight": "bold",
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
})
# -----------------------------
# Smooth curve in log space
# -----------------------------
def smooth_curve_logx(x, y, n=300):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_log = np.log10(x)
    coeffs = np.polyfit(x_log, y, 2)
    x_log_dense = np.linspace(x_log.min(), x_log.max(), n)
    y_dense = np.polyval(coeffs, x_log_dense)
    x_dense = 10 ** x_log_dense
    return x_dense, y_dense

# -----------------------------
# Lambda tick formatting
# -----------------------------
def lambda_formatter(val, pos):
    if np.isclose(val, 1e-4):
        return r"$10^{-4}$"
    elif np.isclose(val, 2e-4):
        return r"$2\times10^{-4}$"
    elif np.isclose(val, 5e-4):
        return r"$5\times10^{-4}$"
    return ""

# -----------------------------
# Plot
# -----------------------------
fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.2))  # was 3.9

metric_info = [
    ("Top1", "(a) Top-1", "Top-1"),
    ("Top5", "(b) Top-5", "Top-5"),
    ("Margin", "(c) Margin", "Margin"),
    ("Uniformity", "(d) Uniformity", "Uniformity (↓ better)"),
]

for ax, (metric, title, ylabel) in zip(axes, metric_info):
    for lang in ["en-fr", "en-ru", "en-it", "en-es"]:
        x_dense, y_dense = smooth_curve_logx(lambda_vals, np.array(results[lang][metric]))

        ax.plot(
            x_dense, y_dense,
            color=colors[lang],
            linestyle=linestyles[lang],
            linewidth=2.2,
            label=lang
        )

        ax.scatter(
            lambda_vals,
            results[lang][metric],
            color=colors[lang],
            marker=markers[lang],
            s=64,
            edgecolor="black",
            linewidth=0.7,
            zorder=3
        )

    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(lambda_vals))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda_formatter))
    ax.xaxis.set_minor_formatter(NullFormatter())

    ax.set_title(title, pad=8, fontweight="bold")
    ax.title.set_color("black")

    ax.set_xlabel(r"$\lambda_{topo}$", color="black")
    ax.set_ylabel(ylabel, color="black", labelpad=4)

    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.45, color="gray")

    ax.tick_params(axis="both", which="major", width=1.2, length=6, color="black")

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color("black")

    # Force tick labels black
    for label in ax.get_xticklabels():
        label.set_color("black")
    for label in ax.get_yticklabels():
        label.set_color("black")

    # Axis limits
    if metric == "Uniformity":
        ax.set_ylim(-3.8, -1.85)
    elif metric == "Top1":
        ax.set_ylim(0.78, 0.93)
    elif metric == "Top5":
        ax.set_ylim(0.88, 0.972)
    elif metric == "Margin":
        ax.set_ylim(0.055, 0.185)

# -----------------------------
# Legend (forced black text)
# -----------------------------
handles, labels = axes[0].get_legend_handles_labels()
legend = fig.legend(
    handles, labels,
    loc="upper center",
    ncol=4,
    frameon=False,
    bbox_to_anchor=(0.5, 1.06),
    columnspacing=1.0,
    handlelength=1.8,
    handletextpad=0.4,
    borderaxespad=0.2,
    fontsize=13 
)

for text in legend.get_texts():
    text.set_color("black")
    text.set_fontsize(14)
    text.set_fontweight("bold")  # optional

# -----------------------------
# Layout
# -----------------------------
plt.subplots_adjust(left=0.055, right=0.995, bottom=0.23, top=0.86, wspace=0.34)

# -----------------------------
# Save
# -----------------------------
plt.savefig("Fig3_final_ieee.png", dpi=700)
plt.savefig("Fig3_final_ieee.pdf")
plt.close()

print("Saved: Fig3_final_ieee.png")
print("Saved: Fig3_final_ieee.pdf")