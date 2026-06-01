import os
import random
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import XLMRobertaTokenizer, XLMRobertaModel
from datasets import load_dataset
import torch.nn as nn

# =========================================================
# CONFIG
# =========================================================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "xlm-roberta-base"
MAX_LENGTH = 64
BATCH_SIZE = 128
VAL_SPLIT = "train[50000:55000]"
LANG_CONFIG = "en-fr"
SEED = 42

MODEL_PATHS = {
    "baseline": "tier1_results/checkpoints/baseline_best.pt",
    "geo": "tier1_results/checkpoints/geo_best.pt",
    "full": "tier1_results/checkpoints/full_topo_0p0001_best.pt",
}

OUT_DIR = "robustness_results"
os.makedirs(OUT_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# =========================================================
# MODEL
# =========================================================
class GeoTransformer(nn.Module):
    def __init__(self, model_name=MODEL_NAME, proj_dim=256):
        super().__init__()
        self.encoder = XLMRobertaModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, proj_dim),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0]
        z = self.projection(cls_emb)
        z = F.normalize(z, dim=-1)
        return z


# =========================================================
# PERTURBATIONS
# =========================================================
def word_dropout(text: str, p: float = 0.1) -> str:
    words = text.split()
    if len(words) <= 2:
        return text
    kept = [w for w in words if random.random() > p]
    if len(kept) < 2:
        kept = words[:2]
    return " ".join(kept)

def local_swap(text: str, p: float = 0.1) -> str:
    words = text.split()
    if len(words) <= 2:
        return text
    i = 0
    while i < len(words) - 1:
        if random.random() < p:
            words[i], words[i+1] = words[i+1], words[i]
            i += 2
        else:
            i += 1
    return " ".join(words)

def char_noise(text: str, p: float = 0.03) -> str:
    chars = list(text)
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    for i, ch in enumerate(chars):
        if ch.isalpha() and random.random() < p:
            chars[i] = random.choice(alphabet)
    return "".join(chars)

def apply_perturbation(text: str, perturb: str, strength: float) -> str:
    if perturb == "clean":
        return text
    if perturb == "dropout":
        return word_dropout(text, p=strength)
    if perturb == "swap":
        return local_swap(text, p=strength)
    if perturb == "char":
        return char_noise(text, p=strength)
    raise ValueError(f"Unknown perturbation: {perturb}")


# =========================================================
# DATASET
# =========================================================
class ParallelDatasetRobust(Dataset):
    def __init__(
        self,
        tokenizer,
        config_name=LANG_CONFIG,
        split=VAL_SPLIT,
        max_length=MAX_LENGTH,
        perturb="clean",
        strength=0.0,
    ):
        dataset = load_dataset("opus_books", config_name, split=split)

        src_lang, tgt_lang = config_name.split("-")
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.pairs = []
        for sample in dataset:
            src = sample["translation"][src_lang]
            tgt = sample["translation"][tgt_lang]
            if isinstance(src, str) and isinstance(tgt, str):
                src = src.strip()
                tgt = tgt.strip()
                if src and tgt:
                    src = apply_perturbation(src, perturb, strength)
                    self.pairs.append((src, tgt))

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]

        enc1 = self.tokenizer(
            src,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        enc2 = self.tokenizer(
            tgt,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return (
            {k: v.squeeze(0) for k, v in enc1.items()},
            {k: v.squeeze(0) for k, v in enc2.items()},
        )


# =========================================================
# EVAL
# =========================================================
@torch.no_grad()
def encode_dataset(model, dataloader):
    model.eval()
    all_z1, all_z2 = [], []

    for batch in dataloader:
        x1, x2 = batch
        x1 = {k: v.to(DEVICE) for k, v in x1.items()}
        x2 = {k: v.to(DEVICE) for k, v in x2.items()}

        z1 = model(**x1)
        z2 = model(**x2)

        all_z1.append(z1)
        all_z2.append(z2)

    z1 = torch.cat(all_z1, dim=0)
    z2 = torch.cat(all_z2, dim=0)
    return F.normalize(z1, dim=-1), F.normalize(z2, dim=-1)

@torch.no_grad()
def compute_retrieval(z1, z2):
    sim = torch.matmul(z1, z2.T)
    n = sim.size(0)
    targets = torch.arange(n, device=sim.device)

    topk1 = sim.topk(1, dim=-1).indices
    topk5 = sim.topk(5, dim=-1).indices

    top1 = (topk1 == targets.unsqueeze(1)).any(dim=1).float().mean().item()
    top5 = (topk5 == targets.unsqueeze(1)).any(dim=1).float().mean().item()

    return {"Top-1": top1, "Top-5": top5}

def load_model(path):
    model = GeoTransformer().to(DEVICE)
    state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


# =========================================================
# MAIN
# =========================================================
def main():
    tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)

    conditions = [
        ("clean", 0.0),
        ("dropout", 0.10),
        ("dropout", 0.20),
        ("swap", 0.10),
        ("swap", 0.20),
        ("char", 0.03),
        ("char", 0.06),
    ]

    results = []

    for model_name, ckpt in MODEL_PATHS.items():
        model = load_model(ckpt)

        for perturb, strength in conditions:
            dataset = ParallelDatasetRobust(
                tokenizer,
                config_name=LANG_CONFIG,
                split=VAL_SPLIT,
                perturb=perturb,
                strength=strength,
            )
            loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
            z1, z2 = encode_dataset(model, loader)
            metrics = compute_retrieval(z1, z2)

            row = {
                "model": model_name,
                "lang_pair": LANG_CONFIG,
                "perturbation": perturb,
                "strength": strength,
                **metrics,
            }
            results.append(row)
            print(row)

    out_path = os.path.join(OUT_DIR, "robustness_en_fr.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved robustness results to {out_path}")


if __name__ == "__main__":
    main()