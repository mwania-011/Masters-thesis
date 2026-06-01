import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

from transformers import XLMRobertaTokenizer, XLMRobertaModel
from datasets import load_dataset

# -----------------------
# Device
# -----------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
# XNLI Dataset
# -----------------------
class XNLIDataset(Dataset):
    def __init__(self, tokenizer, split="train[:50000]", max_length=64):
        self.dataset = load_dataset("xnli", "en", split=split)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        text1 = sample["premise"]
        text2 = sample["hypothesis"]

        # Safety check (VERY IMPORTANT)
        if not isinstance(text1, str):
            text1 = ""
        if not isinstance(text2, str):
            text2 = ""

        # Optional: skip empty samples (better)
        if text1.strip() == "" or text2.strip() == "":
            text1 = "empty"
            text2 = "empty"

        enc1 = self.tokenizer(
            text1,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        enc2 = self.tokenizer(
            text2,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return (
            {k: v.squeeze(0) for k, v in enc1.items()},
            {k: v.squeeze(0) for k, v in enc2.items()},
        )

# -----------------------
# Training Step
# -----------------------
def training_step(model, batch, optimizer, lambda_topo=0.5, lambda_geo=0.1):
    model.train()

    x1, x2 = batch
    x1 = {k: v.to(device) for k, v in x1.items()}
    x2 = {k: v.to(device) for k, v in x2.items()}

    z1 = model(**x1)
    z2 = model(**x2)

    L_align = alignment_loss(z1, z2)

    z_all = torch.cat([z1, z2], dim=0)
    L_topo = topology_loss(z_all)
    L_geo = geometry_loss(z_all)

    L_total = L_align + lambda_topo * L_topo + lambda_geo * L_geo

    optimizer.zero_grad()
    L_total.backward()
    optimizer.step()

    return L_total.item(), L_align.item(), L_topo.item(), L_geo.item()

# -----------------------
# Evaluation
# -----------------------
def evaluate(model, dataloader):
    model.eval()
    total_align = 0

    with torch.no_grad():
        for batch in dataloader:
            x1, x2 = batch
            x1 = {k: v.to(device) for k, v in x1.items()}
            x2 = {k: v.to(device) for k, v in x2.items()}

            z1 = model(**x1)
            z2 = model(**x2)

            total_align += alignment_loss(z1, z2).item()

    return total_align / len(dataloader)

# -----------------------
# Main
# -----------------------
def main():
    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

    train_dataset = XNLIDataset(tokenizer, split="train[:50000]")
    val_dataset = XNLIDataset(tokenizer, split="validation[:5000]")

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    model = GeoTransformer().to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    for epoch in range(5):
        total_loss = 0

        for batch in train_loader:
            loss, l_align, l_topo, l_geo = training_step(model, batch, optimizer)
            total_loss += loss

        val_score = evaluate(model, val_loader)

        print(f"Epoch {epoch+1}")
        print(f"Train Loss: {total_loss:.4f}")
        print(f"Validation Alignment: {val_score:.4f}")
        print("-" * 50)

    torch.save(model.state_dict(), "geo_xnli_model.pt")
    print("Model saved as geo_xnli_model.pt")

# -----------------------
# Run
# -----------------------
if __name__ == "__main__":
    main()