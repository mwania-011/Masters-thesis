import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import XLMRobertaTokenizer, XLMRobertaModel
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

# -----------------------
# Device
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Model
# -----------------------
class GeoTransformer(nn.Module):
    def __init__(self, model_name="xlm-roberta-base", proj_dim=256):
        super().__init__()
        self.encoder = XLMRobertaModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, proj_dim)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0]
        z = self.projection(cls_emb)
        z = F.normalize(z, dim=-1)
        return z

# -----------------------
# Losses
# -----------------------
def alignment_loss(z1, z2):
    return F.mse_loss(z1, z2)

def compute_knn_graph(z, k=5):
    with torch.no_grad():
        sim = torch.matmul(z, z.T)
        _, indices = torch.topk(sim, k=k+1, dim=-1)
    return indices[:, 1:]

def topology_loss(z, k=5):
    knn_idx = compute_knn_graph(z, k)
    loss = 0.0
    for i in range(z.size(0)):
        neighbors = z[knn_idx[i]]
        loss += F.mse_loss(z[i].unsqueeze(0), neighbors.mean(dim=0, keepdim=True))
    return loss / z.size(0)

def geometry_loss(z):
    sim_matrix = torch.matmul(z, z.T)
    identity = torch.eye(z.size(0)).to(z.device)
    return (sim_matrix - identity).pow(2).mean()

# -----------------------
# Dummy Dataset (replace with real data later)
# -----------------------
class DummyParallelDataset(Dataset):
    def __init__(self, tokenizer, size=1000):
        self.tokenizer = tokenizer
        self.size = size
        self.data = [("Hello world", "Bonjour le monde")] * size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        en, fr = self.data[idx]

        en_enc = self.tokenizer(en, return_tensors="pt", padding="max_length", truncation=True, max_length=32)
        fr_enc = self.tokenizer(fr, return_tensors="pt", padding="max_length", truncation=True, max_length=32)

        return (
            {k: v.squeeze(0) for k, v in en_enc.items()},
            {k: v.squeeze(0) for k, v in fr_enc.items()},
        )

# -----------------------
# Training Step
# -----------------------
def training_step(model, batch, optimizer, lambda_topo=0.5, lambda_geo=0.1):
    model.train()

    en_input, fr_input = batch

    en_input = {k: v.to(device) for k, v in en_input.items()}
    fr_input = {k: v.to(device) for k, v in fr_input.items()}

    z_en = model(**en_input)
    z_fr = model(**fr_input)

    L_align = alignment_loss(z_en, z_fr)

    z_all = torch.cat([z_en, z_fr], dim=0)
    L_topo = topology_loss(z_all)
    L_geo = geometry_loss(z_all)

    L_total = L_align + lambda_topo * L_topo + lambda_geo * L_geo

    optimizer.zero_grad()
    L_total.backward()
    optimizer.step()

    return L_total.item(), L_align.item(), L_topo.item(), L_geo.item()

# -----------------------
# Main Training Loop
# -----------------------
def main():
    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

    dataset = DummyParallelDataset(tokenizer)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = GeoTransformer().to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    for epoch in range(3):
        total_loss = 0

        for batch in dataloader:
            loss, l_align, l_topo, l_geo = training_step(model, batch, optimizer)
            total_loss += loss

        print(f"Epoch {epoch+1}")
        print(f"Total Loss: {total_loss:.4f}")
        print(f"Align: {l_align:.4f} | Topo: {l_topo:.4f} | Geo: {l_geo:.4f}")
        print("-" * 50)

    torch.save(model.state_dict(), "geo_model.pt")
    print("Model saved as geo_model.pt")

# -----------------------
# Run
# -----------------------
if __name__ == "__main__":
    main()