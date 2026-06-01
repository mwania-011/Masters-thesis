import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# =========================================================
# OUTPUT
# =========================================================
OUT_DIR = "final_composite_figures"
os.makedirs(OUT_DIR, exist_ok=True)

# =========================================================
# JOURNAL STYLE
# =========================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.4,
    "figure.dpi": 160,
    "savefig.dpi": 600,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
})

COLORS = {
    "baseline": "#4d4d4d",
    "geo": "#9a9a9a",
    "full1": "#b9c7e6",
    "full2": "#7f9bd1",
    "full5": "#2f5aa8",
}

# =========================================================
# DATA
# =========================================================
# English-French
models_enfr = ["Baseline", "Geo", "Full\n(0.0001)", "Full\n(0.0002)", "Full\n(0.0005)"]
top1_enfr = np.array([0.9224, 0.9154, 0.9228, 0.9226, 0.9252])
top5_enfr = np.array([0.9682, 0.9678, 0.9694, 0.9662, 0.9700])
meanrank_enfr = np.array([10.2972, 10.2320, 10.9678, 10.8696, 9.4522])
medianrank_enfr = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
margin_enfr = np.array([0.1705, 0.1754, 0.1776, 0.1786, 0.1816])
overlap_enfr = np.array([0.3127, 0.3071, 0.3105, 0.3040, 0.3118])
uniformity_enfr = np.array([-3.5474, -3.5878, -3.5983, -3.6261, -3.6272])

# English-Russian
models_enru = ["Baseline", "Geo", "Full\n(0.0001)", "Full\n(0.0002)", "Full\n(0.0005)"]
top1_enru = np.array([0.9167, 0.9183, 0.9139, 0.9215, 0.9179])
top5_enru = np.array([0.9667, 0.9627, 0.9599, 0.9647, 0.9651])
margin_enru = np.array([0.1763, 0.1781, 0.1655, 0.1804, 0.1808])
overlap_enru = np.array([0.3314, 0.3358, 0.3439, 0.3369, 0.3427])
uniformity_enru = np.array([-3.6723, -3.7105, -3.6158, -3.6474, -3.6909])

# Full sweep only
lambda_topo = np.array([0.0001, 0.0002, 0.0005])
sweep_top1 = np.array([0.9228, 0.9226, 0.9252])
sweep_top5 = np.array([0.9694, 0.9662, 0.9700])
sweep_margin = np.array([0.1776, 0.1786, 0.1816])
sweep_uniformity = np.array([-3.5983, -3.6261, -3.6272])

# Baseline references for sweep
baseline_top1_ref = 0.9224
baseline_top5_ref = 0.9682

# =========================================================
# HELPERS
# =========================================================
def style_axis(ax, grid_y=True):
    if grid_y:
        ax.grid(True, axis="y", linestyle="--", linewidth=0.45, alpha=0.35)
    else:
        ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

def panel_label(ax, txt):
    ax.text(
        0.02, 0.96, txt,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=11, fontweight="bold"
    )

def panel_header(ax, title):
    ax.text(
        0.5, 0.97, title,
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=8,
        fontweight="normal"
    )

def bar_colors():
    return [
        COLORS["baseline"],
        COLORS["geo"],
        COLORS["full1"],
        COLORS["full2"],
        COLORS["full5"],
    ]

def normalize_for_summary(values, higher_is_better=True):
    v = np.array(values, dtype=float)
    if not higher_is_better:
        v = -v
    vmin, vmax = v.min(), v.max()
    if np.isclose(vmax, vmin):
        return np.ones_like(v)
    return (v - vmin) / (vmax - vmin)

def validate_lengths(labels, *arrays):
    n = len(labels)
    for idx, arr in enumerate(arrays, start=1):
        if len(arr) != n:
            raise ValueError(
                f"Length mismatch: labels has {n} items but array {idx} has {len(arr)} items."
            )

def draw_grouped_bars(ax, labels, y1, y2, ylabel, legend1, legend2, ylim=None):
    validate_lengths(labels, y1, y2)
    x = np.arange(len(labels))
    width = 0.36
    colors = bar_colors()

    ax.bar(
        x - width / 2, y1, width,
        color=colors, edgecolor="black", linewidth=0.5, label=legend1
    )
    ax.bar(
        x + width / 2, y2, width,
        color=colors, edgecolor="black", linewidth=0.5, hatch="//", label=legend2
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(frameon=False, loc="upper left", ncol=1)
    style_axis(ax)

def draw_single_bar(ax, labels, values, ylabel, ylim=None):
    validate_lengths(labels, values)
    x = np.arange(len(labels))

    ax.bar(x, values, color=bar_colors(), edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    style_axis(ax)

def draw_curve(ax, xvals, yvals, ylabel, baseline=None):
    validate_lengths(xvals, yvals)

    ax.plot(xvals, yvals, marker="o", color="black")
    if baseline is not None:
        ax.axhline(baseline, linestyle="--", linewidth=0.9, color="gray")

    ax.set_xscale("log")
    ax.set_xticks(xvals)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel(r"$\lambda_{topo}$")
    ax.set_ylabel(ylabel)
    style_axis(ax)

# =========================================================
# FIGURE 1
# Main retrieval and structure comparison on English-French
# =========================================================
fig = plt.figure(figsize=(11.2, 6.5))
gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

ax1 = fig.add_subplot(gs[0, 0])
draw_grouped_bars(
    ax1, models_enfr, top1_enfr, top5_enfr,
    ylabel="Accuracy", legend1="Top-1", legend2="Top-5",
    ylim=(0.90, 0.975)
)
panel_label(ax1, "(a)")

ax2 = fig.add_subplot(gs[0, 1])
draw_grouped_bars(
    ax2, models_enfr, meanrank_enfr, medianrank_enfr,
    ylabel="Rank", legend1="Mean rank", legend2="Median rank",
    ylim=(0, 12)
)
panel_label(ax2, "(b)")

ax3 = fig.add_subplot(gs[0, 2])
draw_single_bar(
    ax3, models_enfr, margin_enfr,
    ylabel="Margin", ylim=(0.16, 0.185)
)
panel_label(ax3, "(c)")

ax4 = fig.add_subplot(gs[1, 0])
draw_single_bar(
    ax4, models_enfr, uniformity_enfr,
    ylabel="Uniformity", ylim=(-3.66, -3.52)
)
panel_label(ax4, "(d)")

ax5 = fig.add_subplot(gs[1, 1])
draw_single_bar(
    ax5, models_enfr, overlap_enfr,
    ylabel="Overlap@5", ylim=(0.30, 0.315)
)
panel_label(ax5, "(e)")

# Fixed structure summary panel
ax6 = fig.add_subplot(gs[1, 2])
norm_margin = normalize_for_summary(margin_enfr, higher_is_better=True)
norm_overlap = normalize_for_summary(overlap_enfr, higher_is_better=True)
norm_uniformity = normalize_for_summary(uniformity_enfr, higher_is_better=False)

validate_lengths(models_enfr, norm_margin, norm_overlap, norm_uniformity)

x_summary = np.arange(len(models_enfr))
w_summary = 0.25

ax6.bar(
    x_summary - w_summary, norm_margin, w_summary,
    color="#c7c7c7", edgecolor="black", linewidth=0.5, label="Margin"
)
ax6.bar(
    x_summary, norm_overlap, w_summary,
    color="#8f8f8f", edgecolor="black", linewidth=0.5, label="Overlap@5"
)
ax6.bar(
    x_summary + w_summary, norm_uniformity, w_summary,
    color="#4a6fb3", edgecolor="black", linewidth=0.5, label="Uniformity"
)
ax6.set_xticks(x_summary)
ax6.set_xticklabels(models_enfr)
ax6.set_ylabel("Normalized score")
ax6.set_ylim(0, 1.05)
ax6.legend(frameon=False, loc="upper left")
style_axis(ax6)
panel_label(ax6, "(f)")

fig.subplots_adjust(
    left=0.06,
    right=0.98,
    top=0.96,
    bottom=0.08,
    wspace=0.30,
    hspace=0.35
)
fig.savefig(
    os.path.join(OUT_DIR, "final_composite_fig1_enfr_main.png"),
    bbox_inches="tight"
)
plt.close(fig)

# =========================================================
# FIGURE 2
# Effect of topology weight
# =========================================================
fig = plt.figure(figsize=(8.8, 6.2))
gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.28)

ax1 = fig.add_subplot(gs[0, 0])
draw_curve(ax1, lambda_topo, sweep_top1, "Top-1", baseline=baseline_top1_ref)
panel_label(ax1, "(a)")

ax2 = fig.add_subplot(gs[0, 1])
draw_curve(ax2, lambda_topo, sweep_top5, "Top-5", baseline=baseline_top5_ref)
panel_label(ax2, "(b)")

ax3 = fig.add_subplot(gs[1, 0])
draw_curve(ax3, lambda_topo, sweep_margin, "Margin", baseline=None)
panel_label(ax3, "(c)")

ax4 = fig.add_subplot(gs[1, 1])
draw_curve(ax4, lambda_topo, sweep_uniformity, "Uniformity", baseline=None)
panel_label(ax4, "(d)")

fig.subplots_adjust(
    left=0.06,
    right=0.98,
    top=0.96,
    bottom=0.08,
    wspace=0.30,
    hspace=0.35
)
fig.savefig(
    os.path.join(OUT_DIR, "final_composite_fig2_topology_sweep.png"),
    bbox_inches="tight"
)
plt.close(fig)

# =========================================================
# FIGURE 3
# Cross-language comparison with 5-seed error bars
# =========================================================
fig = plt.figure(figsize=(7.6, 5.4))
gs = GridSpec(2, 2, figure=fig, hspace=0.28, wspace=0.30)

langs = ["en-fr", "en-ru"]
x_lang = np.arange(len(langs))
bar_w = 0.22

# ---------------------------------------------------------
# Data (5-seed means)
# ---------------------------------------------------------
baseline_top1 = np.array([0.9230, 0.9285714286])
geo_top1      = np.array([0.9179, 0.9243428571])
full_top1     = np.array([0.9206, 0.9268571429])

baseline_top5 = np.array([0.9694, 0.9683428571])
geo_top5      = np.array([0.9680, 0.9714285714])
full_top5     = np.array([0.9681, 0.9705142857])

baseline_margin = np.array([0.1705, 0.1892062962])
geo_margin      = np.array([0.1754, 0.1798823386])
full_margin     = np.array([0.1796, 0.1929468393])

baseline_uniformity = np.array([-3.5474, -3.6513824940])
geo_uniformity      = np.array([-3.5878, -3.6106677532])
full_uniformity     = np.array([-3.5983, -3.6521855354])

# ---------------------------------------------------------
# Error bars (5-seed std)
# ---------------------------------------------------------
baseline_top1_err = np.array([0.0022, 0.0065527685])
geo_top1_err      = np.array([0.0031, 0.0105567466])
full_top1_err     = np.array([0.0025, 0.0027701028])

baseline_top5_err = np.array([0.0012, 0.0033709443])
geo_top5_err      = np.array([0.0016, 0.0044629998])
full_top5_err     = np.array([0.0011, 0.0026372527])

baseline_margin_err = np.array([0.0052, 0.0138442177])
geo_margin_err      = np.array([0.0063, 0.0181228518])
full_margin_err     = np.array([0.0032, 0.0046475851])

baseline_uniformity_err = np.array([0.0215, 0.0241730772])
geo_uniformity_err      = np.array([0.0284, 0.0476285757])
full_uniformity_err     = np.array([0.0267, 0.0378940247])

validate_lengths(
    langs,
    baseline_top1, geo_top1, full_top1,
    baseline_top5, geo_top5, full_top5,
    baseline_margin, geo_margin, full_margin,
    baseline_uniformity, geo_uniformity, full_uniformity,
    baseline_top1_err, geo_top1_err, full_top1_err,
    baseline_top5_err, geo_top5_err, full_top5_err,
    baseline_margin_err, geo_margin_err, full_margin_err,
    baseline_uniformity_err, geo_uniformity_err, full_uniformity_err
)

# (a) Top-1
ax1 = fig.add_subplot(gs[0, 0])
ax1.bar(
    x_lang - bar_w, baseline_top1, bar_w,
    yerr=baseline_top1_err, capsize=3,
    color=COLORS["baseline"], edgecolor="black", linewidth=0.5,
    label="Baseline"
)
ax1.bar(
    x_lang, geo_top1, bar_w,
    yerr=geo_top1_err, capsize=3,
    color=COLORS["geo"], edgecolor="black", linewidth=0.5,
    label="Geo"
)
ax1.bar(
    x_lang + bar_w, full_top1, bar_w,
    yerr=full_top1_err, capsize=3,
    color=COLORS["full5"], edgecolor="black", linewidth=0.5,
    label="Full"
)
ax1.set_xticks(x_lang)
ax1.set_xticklabels(langs)
ax1.set_ylabel("Top-1")
ax1.set_ylim(0.90, 0.935)
style_axis(ax1)
panel_label(ax1, "(a)")
panel_header(ax1, "Top-1 Accuracy")
ax1.legend(frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.22))

# (b) Top-5
ax2 = fig.add_subplot(gs[0, 1])
ax2.bar(
    x_lang - bar_w, baseline_top5, bar_w,
    yerr=baseline_top5_err, capsize=3,
    color=COLORS["baseline"], edgecolor="black", linewidth=0.5
)
ax2.bar(
    x_lang, geo_top5, bar_w,
    yerr=geo_top5_err, capsize=3,
    color=COLORS["geo"], edgecolor="black", linewidth=0.5
)
ax2.bar(
    x_lang + bar_w, full_top5, bar_w,
    yerr=full_top5_err, capsize=3,
    color=COLORS["full5"], edgecolor="black", linewidth=0.5
)
ax2.set_xticks(x_lang)
ax2.set_xticklabels(langs)
ax2.set_ylabel("Top-5")
ax2.set_ylim(0.955, 0.973)
style_axis(ax2)
panel_label(ax2, "(b)")
panel_header(ax2, "Top-5 Accuracy")

# ---------------------------------------------------------
# Gains relative to baseline
# ---------------------------------------------------------
geo_margin_gain = geo_margin - baseline_margin
full_margin_gain = full_margin - baseline_margin
geo_margin_gain_err = np.sqrt(geo_margin_err**2 + baseline_margin_err**2)
full_margin_gain_err = np.sqrt(full_margin_err**2 + baseline_margin_err**2)

geo_uniformity_gain = np.abs(geo_uniformity) - np.abs(baseline_uniformity)
full_uniformity_gain = np.abs(full_uniformity) - np.abs(baseline_uniformity)
geo_uniformity_gain_err = np.sqrt(geo_uniformity_err**2 + baseline_uniformity_err**2)
full_uniformity_gain_err = np.sqrt(full_uniformity_err**2 + baseline_uniformity_err**2)

# (c) Margin gain
ax3 = fig.add_subplot(gs[1, 0])
ax3.axhline(0, color="black", linewidth=0.7)
ax3.bar(
    x_lang - bar_w / 2, geo_margin_gain, bar_w,
    yerr=geo_margin_gain_err, capsize=3,
    color=COLORS["geo"], edgecolor="black", linewidth=0.5,
    label="Geo - Baseline"
)
ax3.bar(
    x_lang + bar_w / 2, full_margin_gain, bar_w,
    yerr=full_margin_gain_err, capsize=3,
    color=COLORS["full5"], edgecolor="black", linewidth=0.5,
    label="Full - Baseline"
)
ax3.set_xticks(x_lang)
ax3.set_xticklabels(langs)
ax3.set_ylabel("Δ Margin")
style_axis(ax3)
panel_label(ax3, "(c)")
panel_header(ax3, "Margin Gain")
ax3.legend(frameon=False, loc="upper right", fontsize=8)

# (d) Uniformity gain
ax4 = fig.add_subplot(gs[1, 1])
ax4.axhline(0, color="black", linewidth=1.0)
ax4.bar(
    x_lang - bar_w / 2, geo_uniformity_gain, bar_w,
    yerr=geo_uniformity_gain_err, capsize=3,
    color=COLORS["geo"], edgecolor="black", linewidth=0.5,
    label="Geo - Baseline"
)
ax4.bar(
    x_lang + bar_w / 2, full_uniformity_gain, bar_w,
    yerr=full_uniformity_gain_err, capsize=3,
    color=COLORS["full5"], edgecolor="black", linewidth=0.5,
    label="Full - Baseline"
)
ax4.set_xticks(x_lang)
ax4.set_xticklabels(langs)
ax4.set_ylabel("Δ Uniformity")
style_axis(ax4)
panel_label(ax4, "(d)")
panel_header(ax4, "Uniformity Gain")

fig.subplots_adjust(
    left=0.08,
    right=0.98,
    top=0.90,
    bottom=0.10,
    wspace=0.30,
    hspace=0.28
)
fig.savefig(
    os.path.join(OUT_DIR, "Fig3_Cross_Language_Comparison_ErrorBars.png"),
    bbox_inches="tight"
)
plt.close(fig)

print(f"Done. Figures saved in: {OUT_DIR}")




import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy.interpolate import make_interp_spline

# =========================================================
# OUTPUT
# =========================================================
OUT_DIR = "final_composite_figures"
os.makedirs(OUT_DIR, exist_ok=True)

# =========================================================
# JOURNAL STYLE
# =========================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 0.85,
    "lines.linewidth": 1.8,
    "figure.dpi": 180,
    "savefig.dpi": 600,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "grid.color": "#b0b0b0",
})

# =========================================================
# COLORS
# =========================================================
BLUE = "#1f4ea8"
GRAY = "#9a9a9a"

# =========================================================
# DATA
# topology sweep values
# =========================================================
lambda_topo = np.array([0.0001, 0.0002, 0.0005], dtype=float)

# English-French (Full sweep)
top1_enfr = np.array([0.9228, 0.9226, 0.9252], dtype=float)
top5_enfr = np.array([0.9694, 0.9662, 0.9700], dtype=float)
margin_enfr = np.array([0.1776, 0.1786, 0.1816], dtype=float)
uniformity_enfr = np.array([-3.5983, -3.6261, -3.6272], dtype=float)

# English-Russian (Full sweep)
top1_enru = np.array([0.9139, 0.9215, 0.9179], dtype=float)
top5_enru = np.array([0.9599, 0.9647, 0.9651], dtype=float)
margin_enru = np.array([0.1655, 0.1804, 0.1808], dtype=float)
uniformity_enru = np.array([-3.6158, -3.6474, -3.6909], dtype=float)

# Baseline references
baseline_top1_ref = 0.9224
baseline_top5_ref = 0.9682

# =========================================================
# HELPERS
# =========================================================
def validate_lengths(xvals, *arrays):
    n = len(xvals)
    for i, arr in enumerate(arrays, start=1):
        if len(arr) != n:
            raise ValueError(
                f"Length mismatch: x has {n} points but array {i} has {len(arr)} points."
            )

def style_axis(ax, show_grid=True):
    if show_grid:
        ax.grid(True, axis="both", linestyle="--", linewidth=0.45, alpha=0.28)
    for spine in ax.spines.values():
        spine.set_linewidth(0.85)
    ax.tick_params(axis="both", which="major", width=0.85, length=4)

def add_panel_label(ax, label):
    ax.text(
        0.02, 0.97, label,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=14,
        fontweight="bold"
    )

def smooth_curve_logx(x, y, num=200):
    """
    Smooth in log10(x) space.
    With only 3 measured points, quadratic is the highest sensible order.
    """
    x_log = np.log10(np.asarray(x, dtype=float))
    y = np.asarray(y, dtype=float)

    if len(x) < 3:
        raise ValueError("Need at least 3 points for quadratic smoothing.")

    x_new_log = np.linspace(x_log.min(), x_log.max(), num)
    spline = make_interp_spline(x_log, y, k=2)   # quadratic, safer than cubic for 3 points
    y_new = spline(x_new_log)
    x_new = 10 ** x_new_log
    return x_new, y_new

def set_log_ticks(ax, xvals):
    ax.set_xscale("log")
    ax.set_xticks(xvals)
    ax.set_xticklabels(["0.0001", "0.0002", "0.0005"])

def plot_smooth_panel(
    ax,
    xvals,
    y_enfr,
    y_enru,
    ylabel,
    title,
    baseline=None,
    ylim=None
):
    validate_lengths(xvals, y_enfr, y_enru)

    # Smooth curves
    x_s, y_enfr_s = smooth_curve_logx(xvals, y_enfr, num=200)
    _,   y_enru_s = smooth_curve_logx(xvals, y_enru, num=200)

    # Curves
    ax.plot(
        x_s, y_enfr_s,
        color=BLUE,
        linewidth=2.1,
        linestyle="-",
        zorder=2
    )
    ax.plot(
        x_s, y_enru_s,
        color=GRAY,
        linewidth=2.1,
        linestyle="--",
        zorder=2
    )

    # Actual measured points
    ax.scatter(
        xvals, y_enfr,
        s=34,
        color=BLUE,
        edgecolor="black",
        linewidth=0.45,
        zorder=3
    )
    ax.scatter(
        xvals, y_enru,
        s=34,
        color=GRAY,
        edgecolor="black",
        linewidth=0.45,
        zorder=3
    )

    # Baseline reference
    if baseline is not None:
        ax.axhline(
            baseline,
            color="black",
            linestyle="--",
            linewidth=0.9,
            alpha=0.9,
            zorder=1
        )

    set_log_ticks(ax, xvals)
    ax.set_xlabel(r"$\lambda_{topo}$")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10, pad=6)

    if ylim is not None:
        ax.set_ylim(*ylim)

    style_axis(ax)

# =========================================================
# FIGURE: SMOOTH TOPOLOGY CURVES
# =========================================================
fig = plt.figure(figsize=(12.5, 3.2))
gs = GridSpec(1, 4, figure=fig, wspace=0.30)

# (a)
ax1 = fig.add_subplot(gs[0, 0])
plot_smooth_panel(
    ax1,
    lambda_topo,
    top1_enfr,
    top1_enru,
    ylabel="Top-1",
    title="Top-1",
    baseline=baseline_top1_ref,
    ylim=(0.9135, 0.9257)
)
add_panel_label(ax1, "(a)")

# (b)
ax2 = fig.add_subplot(gs[0, 1])
plot_smooth_panel(
    ax2,
    lambda_topo,
    top5_enfr,
    top5_enru,
    ylabel="",
    title="Top-5",
    baseline=baseline_top5_ref,
    ylim=(0.9594, 0.9705)
)
add_panel_label(ax2, "(b)")

# (c)
ax3 = fig.add_subplot(gs[0, 2])
plot_smooth_panel(
    ax3,
    lambda_topo,
    margin_enfr,
    margin_enru,
    ylabel="",
    title="Margin",
    baseline=None,
    ylim=(0.1645, 0.1840)
)
add_panel_label(ax3, "(c)")

# (d)
ax4 = fig.add_subplot(gs[0, 3])
plot_smooth_panel(
    ax4,
    lambda_topo,
    uniformity_enfr,
    uniformity_enru,
    ylabel="",
    title="Uniformity",
    baseline=None,
    ylim=(-3.695, -3.595)
)
add_panel_label(ax4,   "(d)")

# Remove individual x-labels
for ax in [ax1, ax2, ax3, ax4]:
    ax.set_xlabel("")

# Global x-label
fig.text(0.5, 0.02, r"$\lambda_{topo}$", ha="center", fontsize=11)

# Legend (top center)
fig.legend(
    handles=[
        Line2D([0], [0], color="#1f4ea8", lw=2, label="en-fr"),
        Line2D([0], [0], color="#9a9a9a", lw=2, linestyle="--", label="en-ru"),
    ],
    loc="upper center",
    ncol=2,
    frameon=False,
    bbox_to_anchor=(0.5, 1.05)
)

fig.subplots_adjust(
    left=0.06,
    right=0.99,
    top=0.80,
    bottom=0.18
)

fig.savefig(
    os.path.join(OUT_DIR, "Fig3_Horizontal_JournalReady.png"),
    bbox_inches="tight"
)
plt.close(fig)



import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

# =========================================================
# OUTPUT
# =========================================================
OUT_DIR = "final_composite_figures"
os.makedirs(OUT_DIR, exist_ok=True)

# =========================================================
# JOURNAL / CAMERA-READY STYLE
# =========================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "axes.linewidth": 0.85,
    "lines.linewidth": 1.3,
    "figure.dpi": 180,
    "savefig.dpi": 600,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

# =========================================================
# COLORS
# =========================================================
COLORS = {
    "baseline": "#4d4d4d",
    "geo": "#a6a6a6",
    "full_1": "#c7d4ee",
    "full_2": "#89a4d6",
    "full_5": "#2f5aa8",
}

# =========================================================
# DATA
# =========================================================
models_enfr = ["Baseline", "Geo", "Full\n(0.0001)", "Full\n(0.0002)", "Full\n(0.0005)"]

top1_enfr = np.array([0.9224, 0.9154, 0.9228, 0.9226, 0.9252])
top5_enfr = np.array([0.9682, 0.9678, 0.9694, 0.9662, 0.9700])
meanrank_enfr = np.array([10.2972, 10.2320, 10.9678, 10.8696, 9.4522])
margin_enfr = np.array([0.1705, 0.1754, 0.1776, 0.1786, 0.1816])
overlap_enfr = np.array([0.3127, 0.3071, 0.3105, 0.3040, 0.3118])
uniformity_enfr = np.array([-3.5474, -3.5878, -3.5983, -3.6261, -3.6272])

# =========================================================
# HELPERS
# =========================================================
def validate_lengths(labels, *arrays):
    n = len(labels)
    for idx, arr in enumerate(arrays, start=1):
        if len(arr) != n:
            raise ValueError(
                f"Length mismatch: labels has {n} items but array {idx} has {len(arr)} items."
            )

def panel_label(ax, txt):
    ax.text(
        0.02, 0.94, txt,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=13,
        fontweight="bold"
    )

def style_axis(ax, ygrid=True):
    if ygrid:
        ax.grid(True, axis="y", linestyle="--", linewidth=0.45, alpha=0.22)
    else:
        ax.grid(False)

    for spine in ax.spines.values():
        spine.set_linewidth(0.85)

    ax.tick_params(axis="both", width=0.85, length=4)

def model_bar_colors():
    return [
        COLORS["baseline"],
        COLORS["geo"],
        COLORS["full_1"],
        COLORS["full_2"],
        COLORS["full_5"],
    ]

def draw_metric_bars(ax, labels, values, ylabel, title, ylim=None, higher_is_better=True):
    validate_lengths(labels, values)
    x = np.arange(len(labels))
    colors = model_bar_colors()

    bars = ax.bar(
        x,
        values,
        width=0.62,
        color=colors,
        edgecolor="black",
        linewidth=0.6
    )

    if higher_is_better:
        best_idx = int(np.argmax(values))
    else:
        best_idx = int(np.argmin(values))

    bars[best_idx].set_edgecolor("black")
    bars[best_idx].set_linewidth(1.2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9.5, pad=6)

    if ylim is not None:
        ax.set_ylim(*ylim)

    style_axis(ax)

def normalize_for_summary(values, higher_is_better=True):
    v = np.array(values, dtype=float)
    if not higher_is_better:
        v = -v
    vmin, vmax = v.min(), v.max()
    if np.isclose(vmax, vmin):
        return np.ones_like(v)
    return (v - vmin) / (vmax - vmin)

# =========================================================
# FIGURE 1
# =========================================================
fig = plt.figure(figsize=(11.4, 6.3))
gs = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.34)

# (a) Top-1
ax1 = fig.add_subplot(gs[0, 0])
draw_metric_bars(
    ax1,
    models_enfr,
    top1_enfr,
    ylabel="Top-1",
    title="Top-1 Retrieval Accuracy",
    ylim=(0.910, 0.927),
    higher_is_better=True
)
panel_label(ax1, "(a)")

# (b) Top-5
ax2 = fig.add_subplot(gs[0, 1])
draw_metric_bars(
    ax2,
    models_enfr,
    top5_enfr,
    ylabel="Top-5",
    title="Top-5 Retrieval Accuracy",
    ylim=(0.965, 0.971),
    higher_is_better=True
)
panel_label(ax2, "(b)")

# (c) Mean Rank
ax3 = fig.add_subplot(gs[0, 2])
draw_metric_bars(
    ax3,
    models_enfr,
    meanrank_enfr,
    ylabel="Mean Rank",
    title="Mean Rank",
    ylim=(9.0, 11.2),
    higher_is_better=False
)
panel_label(ax3, "(c)")

# (d) Margin
ax4 = fig.add_subplot(gs[1, 0])
draw_metric_bars(
    ax4,
    models_enfr,
    margin_enfr,
    ylabel="Margin",
    title="Margin",
    ylim=(0.168, 0.183),
    higher_is_better=True
)
panel_label(ax4, "(d)")

# (e) Uniformity
ax5 = fig.add_subplot(gs[1, 1])
draw_metric_bars(
    ax5,
    models_enfr,
    uniformity_enfr,
    ylabel="Uniformity",
    title="Uniformity",
    ylim=(-3.635, -3.540),
    higher_is_better=False
)
panel_label(ax5, "(e)")

# (f) Normalized structural summary
ax6 = fig.add_subplot(gs[1, 2])

norm_margin = normalize_for_summary(margin_enfr, higher_is_better=True)
norm_overlap = normalize_for_summary(overlap_enfr, higher_is_better=True)
norm_uniformity = normalize_for_summary(uniformity_enfr, higher_is_better=False)

validate_lengths(models_enfr, norm_margin, norm_overlap, norm_uniformity)

x = np.arange(len(models_enfr))
w = 0.23

bars1 = ax6.bar(
    x - w, norm_margin, width=w,
    color="#d0d0d0", edgecolor="black", linewidth=0.6, label="Margin"
)
bars2 = ax6.bar(
    x, norm_overlap, width=w,
    color="#9f9f9f", edgecolor="black", linewidth=0.6, label="Overlap@5"
)
bars3 = ax6.bar(
    x + w, norm_uniformity, width=w,
    color="#4f79c7", edgecolor="black", linewidth=0.6, label="Uniformity"
)

# highlight strongest combined model
best_idx = int(np.argmax(norm_margin + norm_overlap + norm_uniformity))
for bar_group in (bars1, bars2, bars3):
    bar_group[best_idx].set_edgecolor("black")
    bar_group[best_idx].set_linewidth(1.0)

ax6.set_xticks(x)
ax6.set_xticklabels(models_enfr)
ax6.set_ylabel("Normalized Score")
ax6.set_title("Normalized Structural Metrics (Min–Max Across Models)", fontsize=9.5, pad=6)
ax6.set_ylim(0, 1.05)
style_axis(ax6)
panel_label(ax6, "(f)")
ax6.legend(frameon=False, loc="upper left", ncol=1)

# Shared family legend
family_legend = [
    Patch(facecolor=COLORS["baseline"], edgecolor="black", linewidth=0.6, label="Baseline"),
    Patch(facecolor=COLORS["geo"], edgecolor="black", linewidth=0.6, label="Geo"),
    Patch(facecolor=COLORS["full_1"], edgecolor="black", linewidth=0.6, label="Full variants"),
]

fig.legend(
    handles=family_legend,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=3,
    frameon=False
)

fig.subplots_adjust(
    left=0.06,
    right=0.985,
    top=0.90,
    bottom=0.10,
    wspace=0.34,
    hspace=0.42
)

save_path = os.path.join(OUT_DIR, "Figure1_CameraReady_Final.png")
fig.savefig(save_path, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {save_path}")