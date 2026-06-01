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
# Contrastive Loss
# -----------------------
def alignment_loss(z1, z2, temperature=0.05):
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    logits = torch.matmul(z1, z2.T) / temperature
    labels = torch.arange(z1.size(0)).to(z1.device)

    return F.cross_entropy(logits, labels)

# -----------------------
# Geometry Loss
# -----------------------
def geometry_loss(z):
    sim_matrix = torch.matmul(z, z.T)
    identity = torch.eye(z.size(0)).to(z.device)
    return (sim_matrix - identity).pow(2).mean()

# -----------------------
# Tatoeba Dataset (EN-FR)
# -----------------------
from datasets import load_dataset

class ParallelDataset(Dataset):
    def __init__(self, tokenizer, split="train[:50000]", max_length=64):
        dataset = load_dataset("opus_books", "en-fr", split=split)

        self.pairs = []
        for sample in dataset:
            en = sample["translation"]["en"]
            fr = sample["translation"]["fr"]

            if isinstance(en, str) and isinstance(fr, str):
                self.pairs.append((en, fr))

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        en, fr = self.pairs[idx]

        enc1 = self.tokenizer(
            en,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        enc2 = self.tokenizer(
            fr,
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
def training_step(model, batch, optimizer, lambda_geo=0.05):
    model.train()

    x1, x2 = batch
    x1 = {k: v.to(device) for k, v in x1.items()}
    x2 = {k: v.to(device) for k, v in x2.items()}

    z1 = model(**x1)
    z2 = model(**x2)

    L_align = alignment_loss(z1, z2)

    z_all = torch.cat([z1, z2], dim=0)
    L_geo = geometry_loss(z_all)

    L_total = L_align + lambda_geo * L_geo

    optimizer.zero_grad()
    L_total.backward()
    optimizer.step()

    return L_total.item(), L_align.item(), L_geo.item()

# -----------------------
# Evaluation
# -----------------------
def evaluate(model, dataloader):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            x1, x2 = batch
            x1 = {k: v.to(device) for k, v in x1.items()}
            x2 = {k: v.to(device) for k, v in x2.items()}

            z1 = model(**x1)
            z2 = model(**x2)

            total_loss += alignment_loss(z1, z2).item()

    return total_loss / len(dataloader)

# -----------------------
# Main
# -----------------------
def main():
    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

    train_dataset = ParallelDataset(tokenizer, split="train[:50000]")
    val_dataset = ParallelDataset(tokenizer, split="train[50000:55000]")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    model = GeoTransformer().to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    for epoch in range(5):
        total_loss = 0

        for batch in train_loader:
            loss, l_align, l_geo = training_step(model, batch, optimizer)
            total_loss += loss

        val_score = evaluate(model, val_loader)

        print(f"Epoch {epoch+1}")
        print(f"Train Loss: {total_loss:.4f}")
        print(f"Validation Contrastive Loss: {val_score:.4f}")
        print("-" * 50)

    torch.save(model.state_dict(), "geo_tatoeba_model.pt")
    print("Model saved as geo_tatoeba_model.pt")

if __name__ == "__main__":
    main()