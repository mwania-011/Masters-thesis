import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.manifold import TSNE
import seaborn as sns

# Import your model + dataset
from train_geo_tatoeba import GeoTransformer, ParallelDataset
from transformers import XLMRobertaTokenizer
from torch.utils.data import DataLoader

# -----------------------
# CONFIG
# -----------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "figures"
os.makedirs(SAVE_DIR, exist_ok=True)

# -----------------------
# LOAD DATA
# -----------------------
tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
dataset = ParallelDataset(tokenizer, split="train[50000:51000]")  # smaller for speed
dataloader = DataLoader(dataset, batch_size=128)

# -----------------------
# LOAD MODEL
# -----------------------
model = GeoTransformer().to(device)
model.load_state_dict(torch.load("full_model.pt"))
model.eval()

# -----------------------
# ENCODE
# -----------------------
def encode_dataset(model, dataloader):
    model.eval()
    z1_all, z2_all = [], []

    with torch.no_grad():
        for batch in dataloader:
            x1, x2 = batch
            x1 = {k: v.to(device) for k, v in x1.items()}
            x2 = {k: v.to(device) for k, v in x2.items()}

            z1 = model(**x1)
            z2 = model(**x2)

            z1_all.append(z1)
            z2_all.append(z2)

    return torch.cat(z1_all), torch.cat(z2_all)

z1, z2 = encode_dataset(model, dataloader)

# -----------------------
# FIGURE 1: t-SNE
# -----------------------
def plot_tsne(z1, z2):
    z = torch.cat([z1, z2], dim=0).cpu().numpy()
    labels = np.array([0]*len(z1) + [1]*len(z2))

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    z_2d = tsne.fit_transform(z)

    plt.figure(figsize=(6,6))
    plt.scatter(z_2d[labels==0,0], z_2d[labels==0,1], s=5, label="EN")
    plt.scatter(z_2d[labels==1,0], z_2d[labels==1,1], s=5, label="FR")

    plt.legend()
    plt.title("t-SNE Visualization")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/fig1_tsne.png", dpi=300)
    plt.close()

# -----------------------
# FIGURE 2: Similarity Distribution
# -----------------------
def plot_similarity(z):
    z = F.normalize(z, dim=-1)
    sim = torch.matmul(z, z.T).cpu().numpy()

    sim_vals = sim[np.triu_indices_from(sim, k=1)]

    plt.figure(figsize=(6,4))
    plt.hist(sim_vals, bins=50)
    plt.title("Similarity Distribution")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/fig2_similarity.png", dpi=300)
    plt.close()

# -----------------------
# FIGURE 3: Heatmap
# -----------------------
def plot_heatmap(z):
    z = F.normalize(z, dim=-1)
    sim = torch.matmul(z[:100], z[:100].T).cpu().numpy()

    plt.figure(figsize=(6,5))
    sns.heatmap(sim, cmap="viridis")
    plt.title("Similarity Heatmap")
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/fig3_heatmap.png", dpi=300)
    plt.close()

# -----------------------
# FIGURE 4: Ablation Bar Chart
# -----------------------
def plot_ablation():
    models = ["Baseline", "Geo", "Geo+Topo"]
    top1 = [0.9152, 0.9180, 0.9076]

    x = np.arange(len(models))

    plt.figure(figsize=(6,4))
    plt.bar(x, top1)
    plt.xticks(x, models)
    plt.title("Top-1 Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/fig4_ablation.png", dpi=300)
    plt.close()

# -----------------------
# RUN ALL
# -----------------------
print("Generating figures...")

plot_tsne(z1, z2)
plot_similarity(z1)
plot_heatmap(z1)
plot_ablation()

print(f"All figures saved in ./{SAVE_DIR}/")