import os
import csv
import json
import math
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

from transformers import XLMRobertaTokenizer, XLMRobertaModel
from datasets import load_dataset


# =========================================================
# GLOBAL CONFIG
SEED = 42
MODEL_NAME = "xlm-roberta-base"

TRAIN_SPLIT = "train[:50000]"
VAL_SPLIT = "train[50000:55000]"

MAX_LENGTH = 64
BATCH_SIZE = 32
EPOCHS = 10
LR = 2e-5
PROJ_DIM = 256

ALIGN_TEMP = 0.05
TOPO_TEMP = 0.07
K_NEIGHBORS = 10

LAMBDA_GEO = 0.0005
TOPO_SWEEP = [5e-5, 1e-4, 2e-4]
LANG_CONFIG = "en-ru"
RUN_TAG = "official_en_ru"

# override per language later inside main() if needed
RESULTS_DIR = "tier1_results"
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
LOG_DIR = os.path.join(RESULTS_DIR, "logs")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Helps reduce CUDA fragmentation on some systems
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# =========================================================
# REPRODUCIBILITY
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lang_config", type=str, default="en-fr")
    parser.add_argument("--run_tag", type=str, default="official_run")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
#========================================================
# DATASET
# =========================================================
class ParallelDataset(Dataset):
    def __init__(self, tokenizer, split="train[:50000]", max_length=64):
        dataset = load_dataset("opus_books", LANG_CONFIG, split=split)

        src_lang, tgt_lang = LANG_CONFIG.split("-")

        self.pairs = []
        for sample in dataset:
            src = sample["translation"][src_lang]
            tgt = sample["translation"][tgt_lang]

            if isinstance(src, str) and isinstance(tgt, str):
                src = src.strip()
                tgt = tgt.strip()
                if src and tgt:
                    self.pairs.append((src, tgt))

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]

        enc1 = self.tokenizer(
            src,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        enc2 = self.tokenizer(
            tgt,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return (
            {k: v.squeeze(0) for k, v in enc1.items()},
            {k: v.squeeze(0) for k, v in enc2.items()},
        )

# =========================================================
# ZERO-SHOT DATASET (TATOEBA - ROBUST VERSION)
# =========================================================
from datasets import load_dataset

class TatoebaDataset(Dataset):
    def __init__(self, tokenizer, lang_pair="en-de", max_length=64, max_samples=2000):

        src_lang, tgt_lang = lang_pair.split("-")

        #Load from local file
        file_path = f"data/tatoeba/tatoeba.{src_lang}-{tgt_lang}.txt"

        if not os.path.exists(file_path):
            raise RuntimeError(f"Tatoeba file not found: {file_path}")

        print(f"[Tatoeba] Loading from: {file_path}")

        dataset = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) != 2:
                    continue
                dataset.append((parts[0], parts[1]))

        # Clean + filter
        self.pairs = []
        for src, tgt in dataset:
            if len(self.pairs) >= max_samples:
                break

            if not isinstance(src, str) or not isinstance(tgt, str):
                continue

            src = src.strip()
            tgt = tgt.strip()

            # Filter bad samples
            if len(src) < 3 or len(tgt) < 3:
                continue

            if src != tgt:
                self.pairs.append((src, tgt))

        if len(self.pairs) == 0:
            raise ValueError(f"No valid pairs found for {lang_pair}")

        print(f"[Tatoeba] Loaded {len(self.pairs)} sentence pairs")

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
            return_tensors="pt"
        )

        enc2 = self.tokenizer(
            tgt,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return (
            {k: v.squeeze(0) for k, v in enc1.items()},
            {k: v.squeeze(0) for k, v in enc2.items()},
        )
# =========================================================
# ZERO-SHOT EVALUATION
# =========================================================
@torch.no_grad()
def run_zero_shot_evaluation(model_path: str, tokenizer, lang_pair: str) -> Dict[str, float]:
    print("\n" + "=" * 60)
    print(f"ZERO-SHOT EVAL: {lang_pair}")
    print("=" * 60)

    model = GeoTransformer().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    dataset = TatoebaDataset(
        tokenizer=tokenizer,
        lang_pair=lang_pair,
        max_length=MAX_LENGTH,
        max_samples=2000,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    z1, z2 = encode_dataset(model, loader)
    ret = compute_retrieval_stats(z1, z2)

    print(
        f"{lang_pair} | "
        f"Top1={ret['Top-1']:.4f} | "
        f"Top5={ret['Top-5']:.4f} | "
        f"MeanRank={ret['MeanRank']:.2f} | "
        f"MedianRank={ret['MedianRank']:.1f}"
    )

    return {
        "LanguagePair": lang_pair,
        **ret
    }
# =========================================================
# MODEL
# =========================================================
class GeoTransformer(nn.Module):
    def __init__(self, model_name: str = MODEL_NAME, proj_dim: int = PROJ_DIM):
        super().__init__()
        self.encoder = XLMRobertaModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, proj_dim),
            nn.LayerNorm(proj_dim)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0]
        z = self.projection(cls_emb)
        z = F.normalize(z, dim=-1)
        return z


# =========================================================
# LOSSES
# =========================================================
def alignment_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = ALIGN_TEMP) -> torch.Tensor:
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    sim = torch.matmul(z1, z2.T)

    mask = torch.eye(sim.size(0), device=sim.device).bool()
    sim_neg = sim.masked_fill(mask, -1e9)

    hard_neg_idx = sim_neg.argmax(dim=1)
    hard_neg = sim[torch.arange(sim.size(0)), hard_neg_idx]

    pos = sim.diag()

    margin = 0.3
    loss_hard = F.relu(margin - pos + hard_neg).mean()

    logits = sim / temperature
    labels = torch.arange(z1.size(0), device=z1.device)
    loss_ce = 0.5 * (
        F.cross_entropy(logits, labels, label_smoothing=0.05) +
        F.cross_entropy(logits.T, labels, label_smoothing=0.05)
    )
    return loss_ce + 0.5 * loss_hard

# =========================================================
# REPLACE geometry_loss WITH THIS PAIR-AWARE VERSION
# =========================================================
def geometry_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """
    Pair-aware geometric regularization.
    Penalizes redundancy while NOT penalizing true cross-lingual positives.
    """
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    z_all = torch.cat([z1, z2], dim=0)          # [2B, d]
    sim = torch.matmul(z_all, z_all.T)          # [2B, 2B]
    n = sim.size(0)
    b = z1.size(0)

    # mask out:
    # 1) self-similarity
    # 2) true translation-pair similarity across languages
    mask = torch.zeros_like(sim, dtype=torch.bool)
    eye = torch.eye(n, device=sim.device, dtype=torch.bool)
    mask |= eye

    idx = torch.arange(b, device=sim.device)
    mask[idx, idx + b] = True
    mask[idx + b, idx] = True

    penalty = sim.masked_fill(mask, 0.0)
    loss = (penalty.pow(2).sum()) / ((~mask).sum().clamp_min(1))
    return loss


def topology_loss(z1: torch.Tensor, z2: torch.Tensor, tau_topo: float = TOPO_TEMP) -> torch.Tensor:
    """
    Stable topology-preserving loss:
    matches local neighborhood distributions across languages.
    """
    eps = 1e-6

    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    sim1 = torch.matmul(z1, z1.T) / tau_topo
    sim2 = torch.matmul(z2, z2.T) / tau_topo

    mask = torch.eye(sim1.size(0), device=sim1.device, dtype=torch.bool)
    sim1 = sim1.masked_fill(mask, -1e4)
    sim2 = sim2.masked_fill(mask, -1e4)

    log_p1 = F.log_softmax(sim1, dim=-1)
    log_p2 = F.log_softmax(sim2, dim=-1)

    p1 = torch.clamp(log_p1.exp(), min=eps, max=1.0)
    p2 = torch.clamp(log_p2.exp(), min=eps, max=1.0)

    kl_12 = F.kl_div(log_p1, p2, reduction="batchmean")
    kl_21 = F.kl_div(log_p2, p1, reduction="batchmean")

    loss = 0.5 * (kl_12 + kl_21)
    loss = torch.clamp(loss, min=0.0, max=10.0)

    if torch.isnan(loss) or torch.isinf(loss):
        return torch.tensor(0.0, device=z1.device)

    return loss


# =========================================================
# STRUCTURE METRICS
# =========================================================
@torch.no_grad()
def neighborhood_overlap_at_k(z1: torch.Tensor, z2: torch.Tensor, k: int = K_NEIGHBORS) -> float:
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    sim1 = torch.matmul(z1, z1.T)
    sim2 = torch.matmul(z2, z2.T)

    mask = torch.eye(sim1.size(0), device=sim1.device, dtype=torch.bool)
    sim1 = sim1.masked_fill(mask, -1e9)
    sim2 = sim2.masked_fill(mask, -1e9)

    nn1 = sim1.topk(k, dim=-1).indices
    nn2 = sim2.topk(k, dim=-1).indices

    overlaps = []
    for i in range(nn1.size(0)):
        set1 = set(nn1[i].tolist())
        set2 = set(nn2[i].tolist())
        overlaps.append(len(set1.intersection(set2)) / float(k))
    return float(sum(overlaps) / len(overlaps))


@torch.no_grad()
def uniformity_metric(z: torch.Tensor, t: float = 2.0) -> float:
    """
    Approximate uniformity from Wang & Isola-style formulation.
    Lower is generally better (more uniform spread).
    """
    z = F.normalize(z, dim=-1)
    dists_sq = torch.pdist(z, p=2).pow(2)
    val = torch.log(torch.exp(-t * dists_sq).mean() + 1e-12)
    return float(val.item())


# =========================================================
# RETRIEVAL EVAL
# =========================================================
@torch.no_grad()
def encode_dataset(model: nn.Module, dataloader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
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
def compute_retrieval_stats(z1: torch.Tensor, z2: torch.Tensor) -> Dict[str, float]:
    sim = torch.matmul(z1, z2.T)
    n = sim.size(0)
    targets = torch.arange(n, device=sim.device)

    # ranks
    sorted_idx = torch.argsort(sim, dim=1, descending=True)
    ranks = (sorted_idx == targets.unsqueeze(1)).nonzero(as_tuple=False)[:, 1] + 1
    ranks_np = ranks.cpu().numpy()

    # top-k
    top1 = float(np.mean(ranks_np <= 1))
    top5 = float(np.mean(ranks_np <= 5))
    mean_rank = float(np.mean(ranks_np))
    median_rank = float(np.median(ranks_np))

    # positive / hardest negative / margin
    pos = sim[targets, targets].cpu().numpy()
    sim_np = sim.cpu().numpy()

    hardneg = []
    for i in range(n):
        row = np.delete(sim_np[i], i)
        hardneg.append(np.max(row))
    hardneg = np.array(hardneg)
    margin = pos - hardneg

    return {
        "Top-1": top1,
        "Top-5": top5,
        "MeanRank": mean_rank,
        "MedianRank": median_rank,
        "PosSimMean": float(np.mean(pos)),
        "HardNegMean": float(np.mean(hardneg)),
        "MarginMean": float(np.mean(margin)),
    }


@torch.no_grad()
def evaluate_alignment_loss(model: nn.Module, dataloader: DataLoader) -> float:
    model.eval()
    total = 0.0
    count = 0

    for batch in dataloader:
        x1, x2 = batch
        x1 = {k: v.to(DEVICE) for k, v in x1.items()}
        x2 = {k: v.to(DEVICE) for k, v in x2.items()}

        z1 = model(**x1)
        z2 = model(**x2)

        total += alignment_loss(z1, z2).item()
        count += 1

    return total / max(count, 1)


# =========================================================
# TRAINING
# =========================================================
@dataclass
class ExperimentConfig:
    name: str
    use_align: bool
    use_geo: bool
    use_topo: bool
    lambda_geo: float = LAMBDA_GEO
    lambda_topo: float = 0.0


# =========================================================
# REPLACE get_effective_weights WITH THIS
# =========================================================
def get_effective_weights(cfg: ExperimentConfig, epoch: int) -> Tuple[float, float]:
    """
    More conservative curriculum:
    - baseline: alignment only
    - geo: start geometry late and gently
    - full: first align, then light geometry, then very late topology
    """
    if cfg.use_align and (not cfg.use_geo) and (not cfg.use_topo):
        return 0.0, 0.0

    if cfg.use_align and cfg.use_geo and (not cfg.use_topo):
        if epoch < 3:
            return 0.0, 0.0
        elif epoch < 6:
            return 0.25 * cfg.lambda_geo, 0.0
        elif epoch < 8:
            return 0.50 * cfg.lambda_geo, 0.0
        else:
            return cfg.lambda_geo, 0.0

    if cfg.use_align and cfg.use_geo and cfg.use_topo:
        if epoch < 4:
            return 0.0, 0.0
        elif epoch < 7:
            return 0.25 * cfg.lambda_geo, 0.0
        elif epoch < 9:
            return 0.50 * cfg.lambda_geo, 0.25 * cfg.lambda_topo
        else:
            return cfg.lambda_geo, 0.50 * cfg.lambda_topo

    return 0.0, 0.0


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    cfg: ExperimentConfig,
    epoch: int,
) -> Dict[str, float]:
    model.train()

    total_loss = 0.0
    total_align = 0.0
    total_geo = 0.0
    total_topo = 0.0
    total_batches = 0

    lambda_geo_eff, lambda_topo_eff = get_effective_weights(cfg, epoch)

    for batch in dataloader:
        x1, x2 = batch
        x1 = {k: v.to(DEVICE) for k, v in x1.items()}
        x2 = {k: v.to(DEVICE) for k, v in x2.items()}

        z1 = model(**x1)
        z2 = model(**x2)

        L_align = torch.tensor(0.0, device=DEVICE)
        L_geo = torch.tensor(0.0, device=DEVICE)
        L_topo = torch.tensor(0.0, device=DEVICE)

        if cfg.use_align:
            L_align = alignment_loss(z1, z2)

        if cfg.use_geo and lambda_geo_eff > 0:
            L_geo = geometry_loss(z1, z2)

        if cfg.use_topo and lambda_topo_eff > 0:
            L_topo = topology_loss(z1, z2)
            if torch.isnan(L_topo) or torch.isinf(L_topo):
                print(f"[WARN] Topology exploded in {cfg.name}, epoch {epoch+1}. Skipping batch.")
                continue

        L_total = L_align + lambda_geo_eff * L_geo + lambda_topo_eff * L_topo

        if torch.isnan(L_total) or torch.isinf(L_total):
            print(f"[WARN] NaN/Inf loss in {cfg.name}, epoch {epoch+1}. Skipping batch.")
            continue

        if not L_total.requires_grad:
            print(f"[WARN] Loss has no grad_fn in {cfg.name}, epoch {epoch+1}. Skipping batch.")
            continue

        optimizer.zero_grad()
        L_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += float(L_total.item())
        total_align += float(L_align.item())
        total_geo += float(L_geo.item())
        total_topo += float(L_topo.item())
        total_batches += 1

    denom = max(total_batches, 1)
    return {
        "TrainLoss": total_loss / denom,
        "TrainAlign": total_align / denom,
        "TrainGeo": total_geo / denom,
        "TrainTopo": total_topo / denom,
        "LambdaGeoEff": lambda_geo_eff,
        "LambdaTopoEff": lambda_topo_eff,
    }

def run_experiment(
    cfg: ExperimentConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
) -> Dict[str, float]:
    print("\n" + "=" * 80)
    print(f"Running experiment: {cfg.name}")
    print(asdict(cfg))
    print("=" * 80)

    model = GeoTransformer().to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR)

    best_val_loss = float("inf")
    best_ckpt = os.path.join(CHECKPOINT_DIR, f"{cfg.name}_best.pt")
    last_ckpt = os.path.join(CHECKPOINT_DIR, f"{cfg.name}_last.pt")
    epoch_log_path = os.path.join(LOG_DIR, f"{cfg.name}_epoch_log.csv")

    with open(epoch_log_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Epoch",
                "TrainLoss",
                "TrainAlign",
                "TrainGeo",
                "TrainTopo",
                "LambdaGeoEff",
                "LambdaTopoEff",
                "ValAlignLoss",
                "Top-1",
                "Top-5",
                "MeanRank",
                "MedianRank",
                "PosSimMean",
                "HardNegMean",
                "MarginMean",
                "NeighborOverlap@5",
                "Uniformity",
            ],
        )
        writer.writeheader()

        for epoch in range(EPOCHS):
            train_stats = train_one_epoch(model, train_loader, optimizer, cfg, epoch)

            val_loss = evaluate_alignment_loss(model, val_loader)
            z1, z2 = encode_dataset(model, val_loader)
            ret = compute_retrieval_stats(z1, z2)

            neighbor_overlap = neighborhood_overlap_at_k(z1, z2, k=K_NEIGHBORS)
            uniformity = uniformity_metric(torch.cat([z1, z2], dim=0))

            row = {
                "Epoch": epoch + 1,
                **train_stats,
                "ValAlignLoss": val_loss,
                **ret,
                "NeighborOverlap@5": neighbor_overlap,
                "Uniformity": uniformity,
            }
            writer.writerow(row)

            print(
                f"[{cfg.name}] Epoch {epoch+1}/{EPOCHS} | "
                f"Train={train_stats['TrainLoss']:.4f} | "
                f"Val={val_loss:.4f} | "
                f"Top1={ret['Top-1']:.4f} | Top5={ret['Top-5']:.4f} | "
                f"MedianRank={ret['MedianRank']:.2f} | "
                f"Margin={ret['MarginMean']:.4f} | "
                f"Overlap@5={neighbor_overlap:.4f} | "
                f"Uniformity={uniformity:.4f}"
            )

            current_top1 = ret["Top-1"]

            if epoch == 0 or current_top1 > best_val_loss:
                best_val_loss = current_top1
                torch.save(model.state_dict(), best_ckpt)

    # Final evaluation using best checkpoint
    model.load_state_dict(torch.load(best_ckpt, map_location=DEVICE))
    z1, z2 = encode_dataset(model, val_loader)
    final_ret = compute_retrieval_stats(z1, z2)
    final_neighbor_overlap = neighborhood_overlap_at_k(z1, z2, k=K_NEIGHBORS)
    final_uniformity = uniformity_metric(torch.cat([z1, z2], dim=0))
    final_val_loss = evaluate_alignment_loss(model, val_loader)

    summary = {
        "Experiment": cfg.name,
        "UseAlign": int(cfg.use_align),
        "UseGeo": int(cfg.use_geo),
        "UseTopo": int(cfg.use_topo),
        "LambdaGeo": cfg.lambda_geo,
        "LambdaTopo": cfg.lambda_topo,
        "BestValAlignLoss": final_val_loss,
        **final_ret,
        "NeighborOverlap@5": final_neighbor_overlap,
        "Uniformity": final_uniformity,
        "BestCheckpoint": best_ckpt,
        "LastCheckpoint": last_ckpt,
    }

    with open(os.path.join(LOG_DIR, f"{cfg.name}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# =======================
# REPLACE YOUR CURRENT main() WITH THIS
# =======================
def main():
    args = parse_args()
    set_seed(args.seed)

    # ---------------------------------
    # Runtime globals for this run
    # ---------------------------------
    global LANG_CONFIG, RUN_TAG, RESULTS_DIR, CHECKPOINT_DIR, LOG_DIR
    LANG_CONFIG = args.lang_config
    RUN_TAG = args.run_tag

    RESULTS_DIR = os.path.join("tier1_results", RUN_TAG)
    CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
    LOG_DIR = os.path.join(RESULTS_DIR, "logs")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    print(f"Using device: {DEVICE}")
    print(f"Using seed: {args.seed}")
    print(f"Using language pair: {LANG_CONFIG}")
    print(f"Using run tag: {RUN_TAG}")
    print(f"Results directory: {RESULTS_DIR}")

    # ---------------------------------
    # Save configuration for reproducibility
    # ---------------------------------
    config_dump = {
        "SEED": args.seed,
        "LANG_CONFIG": LANG_CONFIG,
        "RUN_TAG": RUN_TAG,
        "MODEL_NAME": MODEL_NAME,
        "TRAIN_SPLIT": TRAIN_SPLIT,
        "VAL_SPLIT": VAL_SPLIT,
        "MAX_LENGTH": MAX_LENGTH,
        "BATCH_SIZE": BATCH_SIZE,
        "EPOCHS": EPOCHS,
        "LR": LR,
        "PROJ_DIM": PROJ_DIM,
        "ALIGN_TEMP": ALIGN_TEMP,
        "TOPO_TEMP": TOPO_TEMP,
        "K_NEIGHBORS": K_NEIGHBORS,
        "LAMBDA_GEO": LAMBDA_GEO,
    }
    with open(os.path.join(RESULTS_DIR, "official_config.json"), "w") as f:
        json.dump(config_dump, f, indent=2)

       # ---------------------------------
    # Tokenizer and datasets
    # ---------------------------------
    tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)

    # Load full split once to determine size
    
    if LANG_CONFIG == "en-ko":
        print("[INFO] Using Tatoeba for Korean (no OPUS support)")

        # Load from your prepared file instead
        train_dataset = TatoebaDataset(
            tokenizer,
            lang_pair="en-ko",
            max_samples=5000  # you can increase if needed
        )
        total_samples = len(full_dataset)

    else:
        full_dataset = load_dataset("opus_books", LANG_CONFIG, split="train")
        total_samples = len(full_dataset)

        train_dataset = full_dataset

    # Dynamic split: 90% train, 10% validation
    train_end = int(total_samples * 0.9)
    val_start = train_end

    train_split = f"train[:{train_end}]"
    val_split = f"train[{val_start}:]"

    print(f"Total samples: {total_samples}")
    print(f"Train split  : {train_split}")
    print(f"Val split    : {val_split}")

    train_dataset = ParallelDataset(
        tokenizer=tokenizer,
        split=train_split,
        max_length=MAX_LENGTH,
    )
    val_dataset = ParallelDataset(
        tokenizer=tokenizer,
        split=val_split,
        max_length=MAX_LENGTH,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples  : {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # ---------------------------------
# Official experiment list
# ---------------------------------
# Use best full model by language pair

    if LANG_CONFIG in ["en-fr", "fr-en"]:
        full_lambda_topo = 0.0005
        full_name = "full_topo_0p0005"

    elif LANG_CONFIG in ["en-ru", "ru-en"]:
        full_lambda_topo = 0.0004
        full_name = "full_topo_0p0004"

    elif LANG_CONFIG in ["en-de", "de-en"]:
        full_lambda_topo = 0.0002
        full_name = "full_topo_0p0002"

    elif LANG_CONFIG in ["en-es", "es-en"]:
        full_lambda_topo = 0.0002
        full_name = "full_topo_0p0002"

    elif LANG_CONFIG in ["en-it", "it-en"]:
        full_lambda_topo = 0.0002
        full_name = "full_topo_0p0002"

    elif LANG_CONFIG in ["en-pt", "pt-en"]:
        full_lambda_topo = 0.0002
        full_name = "full_topo_0p0002"

    elif LANG_CONFIG in ["en-ko", "ko-en"]:
        full_lambda_topo = 0.0003
        full_name = "full_topo_0p0003"

    else:
        full_lambda_topo = 0.0002
        full_name = "full_topo_0p0002"

    experiments: List[ExperimentConfig] = [
        ExperimentConfig(
            name="baseline",
            use_align=True,
            use_geo=False,
            use_topo=False,
        ),
        ExperimentConfig(
            name="geo",
            use_align=True,
            use_geo=True,
            use_topo=False,
            lambda_geo=LAMBDA_GEO,
        ),
        ExperimentConfig(
            name=full_name,
            use_align=True,
            use_geo=True,
            use_topo=True,
            lambda_geo=LAMBDA_GEO,
            lambda_topo=full_lambda_topo,
        ),
    ]

    # ---------------------------------
    # Run experiments
    # ---------------------------------
    all_summaries: List[Dict[str, float]] = []

    for cfg in experiments:
        summary = run_experiment(cfg, train_loader, val_loader)
        all_summaries.append(summary)

    # ---------------------------------
    # Save final summary CSV in run-specific folder
    # ---------------------------------
    final_summary_path = os.path.join(RESULTS_DIR, "final_summary.csv")
    final_df = pd.DataFrame(all_summaries)
    final_df.to_csv(final_summary_path, index=False)

    # Also save JSON copy
    with open(os.path.join(RESULTS_DIR, "final_summary.json"), "w") as f:
        json.dump(all_summaries, f, indent=2)

    # ---------------------------------
    # Print compact leaderboard
    # ---------------------------------
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    sorted_runs = sorted(
        all_summaries,
        key=lambda x: (-x["Top-1"], -x["Top-5"])
    )

    for row in sorted_runs:
        print(
            f"{row['Experiment']:18s} | "
            f"Top1={row['Top-1']:.4f} | "
            f"Top5={row['Top-5']:.4f} | "
            f"MedianRank={row['MedianRank']:.1f} | "
            f"Margin={row['MarginMean']:.4f} | "
            f"Overlap@5={row['NeighborOverlap@5']:.4f} | "
            f"Uniformity={row['Uniformity']:.4f}"
        )

    print(f"\nSaved to: {RESULTS_DIR}")
    
        # =========================================
    # ZERO-SHOT EVALUATION (NEW SECTION)
    # =========================================
    print("\n" + "=" * 80)
    print("ZERO-SHOT EVALUATION (TATOEBA)")
    print("=" * 80)

    zero_shot_results = []

    # Evaluate ONLY best model (Full)
    best_model_path = sorted(
        all_summaries,
        key=lambda x: -x["Top-1"]
    )[0]["BestCheckpoint"]

    tatoeba_dir = "data/tatoeba"

    available_langs = [
        f.replace("tatoeba.", "").replace(".txt", "")
        for f in os.listdir(tatoeba_dir)
        if f.startswith("tatoeba.")
    ]

    for lang in available_langs:
        try:
            res = run_zero_shot_evaluation(
                model_path=best_model_path,
                tokenizer=tokenizer,
                lang_pair=lang
            )
            zero_shot_results.append(res)
        except Exception as e:
            print(f"Skipping {lang}: {e}")
    # Save results
    zero_path = os.path.join(RESULTS_DIR, "zero_shot_results.csv")
    pd.DataFrame(zero_shot_results).to_csv(zero_path, index=False)

    print(f"\nZero-shot results saved to: {zero_path}")


if __name__ == "__main__":
    main()


