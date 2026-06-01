# Filename: generate_composite_journal_figures.py
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy.interpolate import make_interp_spline

# Output directory
OUT_DIR = "final_composite_figures"
os.makedirs(OUT_DIR, exist_ok=True)

# IEEE colors
COLORS = {
    "baseline": "#0072B2",
    "geo": "#D55E00",
    "full": "#009E73",
}

languages = ["en-fr", "en-ru", "en-it", "en-es"]
models = ["Baseline", "Geo", "Full"]

# Example data: replace with your mean ± std results
top1 = np.array([[0.9230, 0.9286, 0.9235, 0.7961],
                 [0.9179, 0.9243, 0.9219, 0.7964],
                 [0.9206, 0.9269, 0.9235, 0.7961]])
top1_std = np.array([[0.0022, 0.0066, 0.0012, 0.0036],
                     [0.0031, 0.0106, 0.0025, 0.0030],
                     [0.0025, 0.0028, 0.0012, 0.0036]])

top5 = np.array([[0.9694, 0.9683, 0.9647, 0.8834],
                 [0.9680, 0.9714, 0.9639, 0.8860],
                 [0.9681, 0.9705, 0.9647, 0.8834]])
top5_std = np.array([[0.0012, 0.0034, 0.0010, 0.0024],
                     [0.0016, 0.0045, 0.0021, 0.0026],
                     [0.0011, 0.0026, 0.0010, 0.0024]])

margin = np.array([[0.1705, 0.1892, 0.1237, 0.0608],
                   [0.1754, 0.1799, 0.1233, 0.0612],
                   [0.1796, 0.1929, 0.1237, 0.0608]])
margin_std = np.array([[0.0052, 0.0138, 0.0020, 0.0011],
                       [0.0063, 0.0181, 0.0050, 0.0015],
                       [0.0032, 0.0046, 0.0020, 0.0011]])

uniformity = np.array([[-3.5474, -3.6514, -1.9501, -1.9876],
                        [-3.5878, -3.6107, -1.9934, -1.9947],
                        [-3.5983, -3.6522, -1.9501, -1.9876]])
uniformity_std = np.array([[0.0215, 0.0242, 0.0156, 0.0336],
                           [0.0284, 0.0476, 0.0365, 0.0384],
                           [0.0267, 0.0379, 0.0156, 0.0336]])

lambda_topo = np.array([0.0001,0.0002,0.0005])
top1_sweep_enfr = np.array([0.9228, 0.9226, 0.9252])
top1_sweep_enru = np.array([0.9139, 0.9215, 0.9179])

# =========================================================
# Helper functions
# =========================================================
def style_axis(ax):
    ax.grid(True, axis="y", linestyle="--", linewidth=0.45, alpha=0.3)
    for spine in ax.spines.values():
        spine.set_linewidth(0.85)
    ax.tick_params(axis="both", width=0.85, length=4)

def panel_label(ax, txt):
    ax.text(0.02, 0.96, txt, transform=ax.transAxes, ha="left", va="top",
            fontsize=12, fontweight="bold")

def draw_bars(ax, yvals, yerr=None, ylabel="", title=""):
    n_models, n_langs = yvals.shape
    width = 0.25
    x = np.arange(n_langs)
    for i in range(n_models):
        ax.bar(x + (i-1)*width, yvals[i], width,
               yerr=yerr[i] if yerr is not None else None,
               label=models[i],
               color=[COLORS["baseline"], COLORS["geo"], COLORS["full"]][i],
               edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(languages)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    style_axis(ax)

def smooth_curve(x, y, num=200):
    x_log = np.log10(x)
    spline = make_interp_spline(x_log, y, k=2)
    x_smooth_log = np.linspace(x_log.min(), x_log.max(), num)
    y_smooth = spline(x_smooth_log)
    return 10 ** x_smooth_log, y_smooth

# =========================================================
# Composite panel figure
# =========================================================
fig = plt.figure(figsize=(15,8))
gs = GridSpec(2,3,figure=fig, hspace=0.35, wspace=0.35)

# (a) Top-1
ax1 = fig.add_subplot(gs[0,0])
draw_bars(ax1, top1, top1_std, ylabel="Top-1", title="Top-1 Accuracy")
panel_label(ax1,"(a)")

# (b) Top-5
ax2 = fig.add_subplot(gs[0,1])
draw_bars(ax2, top5, top5_std, ylabel="Top-5", title="Top-5 Accuracy")
panel_label(ax2,"(b)")

# (c) Margin
ax3 = fig.add_subplot(gs[0,2])
draw_bars(ax3, margin, margin_std, ylabel="Margin", title="Margin")
panel_label(ax3,"(c)")

# (d) Uniformity
ax4 = fig.add_subplot(gs[1,0])
draw_bars(ax4, uniformity, uniformity_std, ylabel="Uniformity", title="Uniformity")
panel_label(ax4,"(d)")

# (e) Smooth Top-1 sweep
ax5 = fig.add_subplot(gs[1,1])
x_s, y_s_enfr = smooth_curve(lambda_topo, top1_sweep_enfr)
_, y_s_enru = smooth_curve(lambda_topo, top1_sweep_enru)
ax5.plot(x_s, y_s_enfr, color=COLORS["baseline"], lw=2.1, label="en-fr")
ax5.plot(x_s, y_s_enru, color=COLORS["geo"], lw=2.1, linestyle="--", label="en-ru")
ax5.scatter(lambda_topo, top1_sweep_enfr, color=COLORS["baseline"], edgecolor="black")
ax5.scatter(lambda_topo, top1_sweep_enru, color=COLORS["geo"], edgecolor="black")
ax5.set_xscale("log")
ax5.set_xticks(lambda_topo)
ax5.set_xticklabels(["0.0001","0.0002","0.0005"])
ax5.set_xlabel(r"$\lambda_{topo}$")
ax5.set_ylabel("Top-1 Accuracy")
ax5.set_title("Top-1 vs Topology Weight")
ax5.legend(frameon=False)
style_axis(ax5)
panel_label(ax5,"(e)")

# (f) Δ Margin / Δ Uniformity
ax6 = fig.add_subplot(gs[1,2])
delta_margin = margin[2]-margin[0]
delta_uniformity = np.abs(uniformity[2])-np.abs(uniformity[0])
draw_bars(ax6, np.array([delta_margin, delta_uniformity]), np.zeros_like(delta_margin),
          ylabel="Δ Metrics", title="Full - Baseline")
panel_label(ax6,"(f)")

fig.subplots_adjust(left=0.05,right=0.98,top=0.92,bottom=0.08)
fig.savefig(os.path.join(OUT_DIR,"Composite_Panel_Figures.png"), bbox_inches="tight")
plt.close(fig)

print(f"Composite panel figure saved to {OUT_DIR}")