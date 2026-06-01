import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Fig. 4 — Structure Comparison (IEEE Style — matched to Fig. 3)
# ============================================================

langs = ["en-fr", "en-ru", "en-it", "en-es"]
models = ["Baseline", "Geo", "Full"]

data = {
    "Margin": {
        "Baseline": [0.1705, 0.1892, 0.1210, 0.0617],
        "Geo":      [0.1754, 0.1799, 0.1233, 0.0612],
        "Full":     [0.1796, 0.1929, 0.1237, 0.0608],
    },
    "Margin_std": {
        "Baseline": [0.0052, 0.0138, 0.0031, 0.0016],
        "Geo":      [0.0063, 0.0181, 0.0050, 0.0015],
        "Full":     [0.0032, 0.0046, 0.0020, 0.0011],
    },
    "Uniformity": {
        "Baseline": [-3.5474, -3.6514, -1.9571, -1.9761],
        "Geo":      [-3.5878, -3.6107, -1.9934, -1.9947],
        "Full":     [-3.5983, -3.6522, -1.9501, -1.9876],
    },
    "Uniformity_std": {
        "Baseline": [0.0215, 0.0242, 0.0451, 0.0385],
        "Geo":      [0.0284, 0.0476, 0.0365, 0.0384],
        "Full":     [0.0267, 0.0379, 0.0156, 0.0336],
    },
}

# ============================================================
# GLOBAL STYLE — MATCH FIG. 3
# ============================================================
plt.rcParams.update({
    "font.family": "serif",

    # Font sizes (same as Fig. 3)
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,

    # Black styling
    "text.color": "black",
    "axes.labelcolor": "black",
    "axes.edgecolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",

    # Bold style (consistent with Fig. 3)
    "font.weight": "medium",
    "axes.titleweight": "medium",
    "axes.labelweight": "medium",

    "axes.linewidth": 1.0,
    "savefig.bbox": "tight",
})

colors = {
    "Baseline": "#000000",
    "Geo": "#6f6f6f",
    "Full": "#1f4ea8",
}

x = np.arange(len(langs))
w = 0.22

# ============================================================
# DRAW FUNCTION
# ============================================================
def draw_grouped_bars(ax, metric, metric_std=None, title="", ylabel=None, ylim=None):
    for i, m in enumerate(models):
        ax.bar(
            x + (i - 1) * w,
            data[metric][m],
            width=w,
            color=colors[m],
            edgecolor="black",
            linewidth=0.8,
            label=m,
            zorder=3,
            yerr=data[metric_std][m] if metric_std else None,
            capsize=2.5,
            error_kw=dict(
                lw=0.9,
                capthick=0.9,
                ecolor="black",
                zorder=4
            ) if metric_std else None
        )

    ax.set_xticks(x)
    ax.set_xticklabels(langs)

    ax.set_title(title, pad=8)
    ax.set_xlabel("")  # cleaner like Fig. 3

    if ylabel:
        ax.set_ylabel(ylabel, labelpad=5)

    # Grid (match Fig. 3)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.45, color="gray")

    # Spine styling (match Fig. 3)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color("black")

    ax.tick_params(axis="both", which="major", width=1.2, length=6, color="black")

    # Y-limits
    if ylim:
        ax.set_ylim(ylim)
    else:
        vals = []
        for m in models:
            vals.extend(data[metric][m])
        ymin, ymax = min(vals), max(vals)
        padding = 0.05 * (ymax - ymin)
        ax.set_ylim(ymin - padding, ymax + padding)


# ============================================================
# PLOT
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))

draw_grouped_bars(
    axes[0],
    metric="Margin",
    metric_std="Margin_std",
    title="(a) Margin",
    ylabel="Margin",
)

draw_grouped_bars(
    axes[1],
    metric="Uniformity",
    metric_std="Uniformity_std",
    title="(b) Uniformity",
    ylabel="Uniformity (↓ better)",
    ylim=(-3.8, -1.85)
)

# Align y-labels perfectly (same as Fig. 3)
fig.align_ylabels(axes)

# ============================================================
# LEGEND — MATCH FIG. 3
# ============================================================
handles, labels = axes[0].get_legend_handles_labels()

legend = fig.legend(
    handles, labels,
    loc="upper center",
    ncol=3,
    frameon=False,
    bbox_to_anchor=(0.5, 1.08),
    columnspacing=1.2,
    handlelength=1.8,
    handletextpad=0.4,
    borderaxespad=0.2,
    fontsize=12
)

for text in legend.get_texts():
    text.set_color("black")
    text.set_fontsize(13)
    text.set_fontweight("medium")

# ============================================================
# LAYOUT — MATCH FIG. 3
# ============================================================
plt.subplots_adjust(
    left=0.07,
    right=0.995,
    bottom=0.20,
    top=0.85,
    wspace=0.30
)

# ============================================================
# SAVE
# ============================================================
plt.savefig("Fig4_structure_ieee.png", dpi=700)
plt.savefig("Fig4_structure_ieee.pdf")
plt.close()

print("Saved: Fig4_structure_ieee.png")
print("Saved: Fig4_structure_ieee.pdf")