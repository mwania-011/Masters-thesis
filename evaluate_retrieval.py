import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import XLMRobertaTokenizer
from datasets import load_dataset

# Import your model + dataset class
from train_geo_tatoeba import GeoTransformer, ParallelDataset

# -----------------------
# Device
# -----------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# -----------------------
# Encode dataset
# -----------------------
def encode_dataset(model, dataloader):
    model.eval()
    all_z1, all_z2 = [], []

    with torch.no_grad():
        for batch in dataloader:
            x1, x2 = batch
            x1 = {k: v.to(device) for k, v in x1.items()}
            x2 = {k: v.to(device) for k, v in x2.items()}

            z1 = model(**x1)
            z2 = model(**x2)

            all_z1.append(z1)
            all_z2.append(z2)

    return torch.cat(all_z1), torch.cat(all_z2)

# -----------------------
# Retrieval Metrics
# -----------------------
def compute_retrieval(z1, z2, topk=(1, 5)):
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    sim = torch.matmul(z1, z2.T)
    targets = torch.arange(z1.size(0)).to(z1.device)

    results = {}

    for k in topk:
        topk_preds = sim.topk(k, dim=-1).indices
        correct = (topk_preds == targets.unsqueeze(1)).any(dim=1).float()
        acc = correct.mean().item()
        results[f"Top-{k}"] = acc

    return results

# -----------------------
# Evaluate One Model
# -----------------------
def evaluate_model(model_path, dataloader):
    model = GeoTransformer().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    print(f"\nEvaluating: {model_path}")

    z1, z2 = encode_dataset(model, dataloader)
    results = compute_retrieval(z1, z2)

    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    return results

# -----------------------
# Main
# -----------------------
def main():
    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

    # Same validation split (VERY IMPORTANT for fairness)
    dataset = ParallelDataset(tokenizer, split="train[50000:55000]")
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

    # Models to evaluate
    models = {
        "Baseline": "baseline_model.pt",
        "Geo Only": "geo_only_model.pt",
        "Full Model": "full_model.pt"
    }

    all_results = {}

    for name, path in models.items():
        results = evaluate_model(path, dataloader)
        all_results[name] = results

    # -----------------------
    # Final Table
    # -----------------------
    print("\n ===== FINAL RESULTS TABLE =====")
    print("{:<15} {:<10} {:<10}".format("Model", "Top-1", "Top-5"))

    for name, res in all_results.items():
        print("{:<15} {:<10.4f} {:<10.4f}".format(
            name,
            res["Top-1"],
            res["Top-5"]
        ))

if __name__ == "__main__":
    main()