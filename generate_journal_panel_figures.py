# Filename: generate_journal_6panel_cdf.py
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

# =========================================================
# OUTPUT DIRECTORY
# =========================================================
OUT_DIR = "final_composite_figures"
os.makedirs(OUT_DIR, exist_ok=True)

# =========================================================
# COLORS & STYLE
# =========================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 0.85,
    "lines.linewidth": 1.8,
    "figure.dpi": 180,
    "savefig.dpi": 600,
})

# Greyish bars
COLORS_BAR = {"baseline":"#BBBBBB", "geo":"#888888", "full":"#444444"}
# CDF-style curves
CURVE_COLORS = {"en-fr":"#0072B2","en-ru":"#D55E00","en-it":"#009E73","en-es":"#CC79A7"}
CURVE_MARKERS = {"en-fr":"o","en-ru":"s","en-it":"^","en-es":"D"}

languages = ["en-fr","en-ru","en-it","en-es"]
models = ["Baseline","Geo","Full"]

# =========================================================
# METRIC DATA
# =========================================================
# Panels a-d (values + 5-seed std)
top1 = np.array([[0.9230,0.9286,0.9249,0.7961],
                 [0.9179,0.9243,0.9224,0.7964],
                 [0.9206,0.9269,0.9249,0.7961]])
top1_std = np.array([[0.0022,0.0066,0.0028,0.0036],
                     [0.0031,0.0106,0.0035,0.0030],
                     [0.0025,0.0028,0.0029,0.0036]])

top5 = np.array([[0.9694,0.9683,0.9651,0.8834],
                 [0.9680,0.9714,0.9635,0.8860],
                 [0.9681,0.9705,0.9651,0.8834]])
top5_std = np.array([[0.0012,0.0034,0.0023,0.0024],
                     [0.0016,0.0045,0.0021,0.0026],
                     [0.0011,0.0026,0.0022,0.0024]])

margin = np.array([[0.1705,0.1892,0.1207,0.0608],
                   [0.1754,0.1799,0.1126,0.0612],
                   [0.1796,0.1929,0.1207,0.0608]])
margin_std = np.array([[0.0052,0.0138,0.0040,0.0011],
                       [0.0063,0.0181,0.0055,0.0015],
                       [0.0032,0.0046,0.0040,0.0011]])

uniformity = np.array([[-3.5474,-3.6514,-1.9129,-1.9876],
                        [-3.5878,-3.6107,-2.0311,-1.9947],
                        [-3.5983,-3.6522,-1.9129,-1.9876]])
uniformity_std = np.array([[0.0215,0.0242,0.0156,0.0336],
                           [0.0284,0.0476,0.0221,0.0384],
                           [0.0267,0.0379,0.0156,0.0336]])

# Panel e: Top-1 sweep step data
lambda_topo = np.array([0.0001,0.0002,0.0003,0.0004,0.0005])
top1_sweep = {
    "en-fr": np.array([0.9228,0.9226,0.9230,0.9240,0.9252]),
    "en-ru": np.array([0.9139,0.9215,0.9202,0.9220,0.9179]),
    "en-it": np.array([0.9210,0.9220,0.9235,0.9240,0.9235]),
    "en-es": np.array([0.7920,0.7940,0.7960,0.7980,0.7995])
}
top1_sweep_std = {
    "en-fr": np.array([0.002,0.002,0.0018,0.002,0.0025]),
    "en-ru": np.array([0.003,0.003,0.0025,0.003,0.0028]),
    "en-it": np.array([0.002,0.0015,0.002,0.0025,0.002]),
    "en-es": np.array([0.003,0.003,0.003,0.0025,0.002])
}

# =========================================================
# HELPERS
# =========================================================
def style_axis(ax):
    ax.grid(True, axis="y", linestyle="--", alpha=0.3, linewidth=0.45)
    for spine in ax.spines.values():
        spine.set_linewidth(0.85)
    ax.tick_params(axis="both", width=0.85, length=4)

def panel_label(ax, txt):
    ax.text(0.02,0.96,txt,transform=ax.transAxes,ha="left",va="top", fontsize=12,fontweight="bold")

def draw_grey_bars(ax, values, values_std, ylabel="", title=""):
    n_models, n_langs = values.shape
    width = 0.25
    x = np.arange(n_langs)
    for i in range(n_models):
        ax.bar(
            x + (i-1)*width, values[i], width,
            yerr=values_std[i], capsize=3,
            label=models[i],
            color=[COLORS_BAR["baseline"],COLORS_BAR["geo"],COLORS_BAR["full"]][i],
            edgecolor="black", linewidth=0.6
        )
    ax.set_xticks(x)
    ax.set_xticklabels(languages)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    style_axis(ax)

def draw_cdf_panel(ax, xvals, yvals_dict, yerr_dict):
    for lang,color in CURVE_COLORS.items():
        y = yvals_dict[lang]
        y_std = yerr_dict[lang]
        # stepwise CDF-style
        for i in range(len(xvals)-1):
            ax.hlines(y[i], xvals[i], xvals[i+1], color=color, lw=2)
            ax.vlines(xvals[i+1], y[i], y[i+1], color=color, lw=2)
        # markers
        ax.scatter(xvals, y, color=color, edgecolor='black', zorder=3, s=50)
        # shaded band
        ax.fill_between(xvals, y-y_std, y+y_std, color=color, alpha=0.25)
    ax.set_xscale('log')
    ax.set_xticks(xvals)
    ax.set_xticklabels([f"{v:.4f}" for v in xvals])
    ax.set_ylabel("Top-1 Accuracy")
    ax.set_title("Top-1 Accuracy vs Topology Weight (CDF-style)")
    style_axis(ax)
    ax.legend(CURVE_COLORS.keys(), frameon=False)

# =========================================================
# CREATE FIGURE
# =========================================================
fig = plt.figure(figsize=(16,8))
gs = GridSpec(2,3,figure=fig,hspace=0.35,wspace=0.35)

# Panels a-d: grey bars
ax1 = fig.add_subplot(gs[0,0]); draw_grey_bars(ax1, top1, top1_std, ylabel="Top-1", title="Top-1 Accuracy"); panel_label(ax1,"(a)")
ax2 = fig.add_subplot(gs[0,1]); draw_grey_bars(ax2, top5, top5_std, ylabel="Top-5", title="Top-5 Accuracy"); panel_label(ax2,"(b)")
ax3 = fig.add_subplot(gs[0,2]); draw_grey_bars(ax3, margin, margin_std, ylabel="Margin", title="Margin"); panel_label(ax3,"(c)")
ax4 = fig.add_subplot(gs[1,0]); draw_grey_bars(ax4, uniformity, uniformity_std, ylabel="Uniformity", title="Uniformity"); panel_label(ax4,"(d)")

# Panel e: CDF-style Top-1 curves
ax5 = fig.add_subplot(gs[1,1]); draw_cdf_panel(ax5, lambda_topo, top1_sweep, top1_sweep_std); panel_label(ax5,"(e)")

# Panel f: Δ metrics
ax6 = fig.add_subplot(gs[1,2])
delta_margin = margin[2]-margin[0]
delta_geo_margin = margin[1]-margin[0]
delta_uniformity = np.abs(uniformity[2])-np.abs(uniformity[0])
delta_geo_uniformity = np.abs(uniformity[1])-np.abs(uniformity[0])
x = np.arange(len(languages))
width = 0.25
ax6.bar(x-width/2, delta_geo_margin, width, color=COLORS_BAR["geo"], label="Geo - Baseline", edgecolor="black")
ax6.bar(x+width/2, delta_margin, width, color=COLORS_BAR["full"], label="Full - Baseline", edgecolor="black")
# hatch overlay for Uniformity
ax6.bar(x-width/2, delta_geo_uniformity, width, color='none', edgecolor="black", hatch='//')
ax6.bar(x+width/2, delta_uniformity, width, color='none', edgecolor="black", hatch='//')
ax6.set_xticks(x)
ax6.set_xticklabels(languages)
ax6.set_ylabel("Δ Metrics")
ax6.set_title("Full vs Baseline")
ax6.legend(frameon=False)
panel_label(ax6,"(f)")

# =========================================================
# ADJUST & SAVE
# =========================================================
fig.subplots_adjust(left=0.05,right=0.98,top=0.92,bottom=0.08)
fig.savefig(os.path.join(OUT_DIR,"Final_Journal_6panel_CDF.png"),bbox_inches="tight")
plt.close(fig)
print(f"Final multipanel figure saved to {OUT_DIR}")