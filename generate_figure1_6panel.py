import matplotlib.pyplot as plt
import numpy as np

# Language pairs
langs = ['en-fr', 'en-ru', 'en-it', 'en-es']
x = np.arange(len(langs))
width = 0.25

# Example mean ± std for Top-1, Top-5, Margin, Uniformity (replace with your 10-seed data)
top1_mean = [0.9230, 0.9286, 0.9204, 0.7956]
top1_std  = [0.0022, 0.0066, 0.0031, 0.0029]
top1_geo  = [0.9179, 0.9243, 0.9219, 0.7964]
top1_geo_std = [0.0031, 0.0106, 0.0025, 0.0030]
top1_full = [0.9206, 0.9269, 0.9235, 0.7961]
top1_full_std = [0.0025, 0.0028, 0.0012, 0.0036]

top5_mean = [0.9694, 0.9683, 0.9632, 0.8862]
top5_std = [0.0012, 0.0034, 0.0029, 0.0019]
top5_geo = [0.9680, 0.9714, 0.9639, 0.8860]
top5_geo_std = [0.0016, 0.0045, 0.0021, 0.0026]
top5_full = [0.9681, 0.9705, 0.9647, 0.8834]
top5_full_std = [0.0011, 0.0026, 0.0010, 0.0024]

margin_mean = [0.1705, 0.1892, 0.1210, 0.0617]
margin_std = [0.0052, 0.0138, 0.0031, 0.0016]
margin_geo = [0.1754, 0.1799, 0.1233, 0.0612]
margin_geo_std = [0.0063, 0.0181, 0.0050, 0.0015]
margin_full = [0.1796, 0.1929, 0.1237, 0.0608]
margin_full_std = [0.0032, 0.0046, 0.0020, 0.0011]

uniformity_mean = [-3.5474, -3.6514, -1.9571, -1.9761]
uniformity_std = [0.0215, 0.0242, 0.0451, 0.0385]
uniformity_geo = [-3.5878, -3.6107, -1.9934, -1.9947]
uniformity_geo_std = [0.0284, 0.0476, 0.0365, 0.0384]
uniformity_full = [-3.5983, -3.6522, -1.9501, -1.9876]
uniformity_full_std = [0.0267, 0.0379, 0.0156, 0.0336]

# Δ metrics (Full - Baseline example)
delta_top1 = np.array(top1_full) - np.array(top1_mean)
delta_margin = np.array(margin_full) - np.array(margin_mean)

# Create Figure 1 layout
fig, axs = plt.subplots(2, 3, figsize=(18, 10))
plt.subplots_adjust(wspace=0.3, hspace=0.4)

# Panel (a): Top-1 Accuracy
axs[0,0].bar(x - width, top1_mean, width, yerr=top1_std, label='Baseline', color='lightgrey', capsize=3)
axs[0,0].bar(x, top1_geo, width, yerr=top1_geo_std, label='Geo', color='grey', capsize=3)
axs[0,0].bar(x + width, top1_full, width, yerr=top1_full_std, label='Full', color='dimgray', capsize=3)
axs[0,0].set_xticks(x)
axs[0,0].set_xticklabels(langs)
axs[0,0].set_ylabel('Top-1 Accuracy')
axs[0,0].set_title('(a) Top-1 Accuracy')
axs[0,0].legend(frameon=False)

# Panel (b): Top-5 Accuracy
axs[0,1].bar(x - width, top5_mean, width, yerr=top5_std, color='lightgrey', capsize=3)
axs[0,1].bar(x, top5_geo, width, yerr=top5_geo_std, color='grey', capsize=3)
axs[0,1].bar(x + width, top5_full, width, yerr=top5_full_std, color='dimgray', capsize=3)
axs[0,1].set_xticks(x)
axs[0,1].set_xticklabels(langs)
axs[0,1].set_ylabel('Top-5 Accuracy')
axs[0,1].set_title('(b) Top-5 Accuracy')

# Panel (c): Margin
axs[0,2].bar(x - width, margin_mean, width, yerr=margin_std, color='lightgrey', capsize=3)
axs[0,2].bar(x, margin_geo, width, yerr=margin_geo_std, color='grey', capsize=3)
axs[0,2].bar(x + width, margin_full, width, yerr=margin_full_std, color='dimgray', capsize=3)
axs[0,2].set_xticks(x)
axs[0,2].set_xticklabels(langs)
axs[0,2].set_ylabel('Margin')
axs[0,2].set_title('(c) Margin')

# Panel (d): Uniformity
axs[1,0].bar(x - width, uniformity_mean, width, yerr=uniformity_std, color='lightgrey', capsize=3)
axs[1,0].bar(x, uniformity_geo, width, yerr=uniformity_geo_std, color='grey', capsize=3)
axs[1,0].bar(x + width, uniformity_full, width, yerr=uniformity_full_std, color='dimgray', capsize=3)
axs[1,0].set_xticks(x)
axs[1,0].set_xticklabels(langs)
axs[1,0].set_ylabel('Uniformity')
axs[1,0].set_title('(d) Uniformity')

# Panel (e): Top-1 Accuracy vs Topology Weight (CDF-style smooth line)
topo_weights = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]
# Example Top-1 smooth curves (replace with real results)
en_fr_curve = [0.922, 0.9225, 0.923, 0.923, 0.923]
en_ru_curve = [0.914, 0.915, 0.916, 0.917, 0.918]
en_it_curve = [0.918, 0.919, 0.920, 0.921, 0.922]
en_es_curve = [0.792, 0.793, 0.794, 0.795, 0.796]

axs[1,1].plot(topo_weights, en_fr_curve, '-o', label='en-fr', color='tab:blue')
axs[1,1].plot(topo_weights, en_ru_curve, '-o', label='en-ru', color='tab:orange')
axs[1,1].plot(topo_weights, en_it_curve, '-o', label='en-it', color='tab:green')
axs[1,1].plot(topo_weights, en_es_curve, '-o', label='en-es', color='tab:red')
axs[1,1].set_xlabel(r'$\lambda_{topo}$')
axs[1,1].set_ylabel('Top-1 Accuracy')
axs[1,1].set_title('(e) Top-1 Accuracy vs Topology Weight')
axs[1,1].legend(frameon=False)
axs[1,1].grid(True, linestyle='--', alpha=0.3)

# Panel (f): Δ metrics (Full - Baseline)
axs[1,2].bar(x, delta_top1, width, color='dimgray', label='Δ Top-1')
axs[1,2].bar(x, delta_margin, width/2, color='grey', label='Δ Margin')
axs[1,2].set_xticks(x)
axs[1,2].set_xticklabels(langs)
axs[1,2].set_ylabel('Δ Metrics')
axs[1,2].set_title('(f) Δ Metrics')
axs[1,2].legend(frameon=False)

# Adjust layout
plt.tight_layout()
plt.savefig('Figure1_6panel_journal.png', dpi=300)
plt.show()
