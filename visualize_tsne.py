import torch
import torch.nn.functional as F
from transformers import XLMRobertaTokenizer, XLMRobertaModel
from datasets import load_dataset
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# -----------------------
# Device
# -----------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# -----------------------
# Load Data
# -----------------------
def load_samples(n=500):
    dataset = load_dataset("xnli", "en", split=f"validation[:{n}]")
    texts = [x["premise"] for x in dataset if isinstance(x["premise"], str)]
    return texts[:n]

# -----------------------
# Encode with model
# -----------------------
def encode_texts(model, tokenizer, texts):
    embeddings = []

    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=64
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

            # Handle both base model and geo model
            if hasattr(outputs, "last_hidden_state"):
                cls_emb = outputs.last_hidden_state[:, 0]
                cls_emb = F.normalize(cls_emb, dim=-1)
            else:
                cls_emb = outputs  # already normalized embedding

        embeddings.append(cls_emb.squeeze(0).cpu())

    return torch.stack(embeddings)

# -----------------------
# Load Models
# -----------------------
def load_models():
    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

    base_model = XLMRobertaModel.from_pretrained("xlm-roberta-base").to(device)

    # Load trained model
    from train_geo_xnli import GeoTransformer
    geo_model = GeoTransformer().to(device)
    geo_model.load_state_dict(torch.load("geo_tatoeba_model.pt"))
    geo_model.eval()

    return tokenizer, base_model, geo_model

# -----------------------
# t-SNE Plot
# -----------------------
def plot_tsne(emb_before, emb_after):
    tsne = TSNE(n_components=2, perplexity=30)

    emb_before_2d = tsne.fit_transform(emb_before.numpy())
    emb_after_2d = tsne.fit_transform(emb_after.numpy())

    plt.figure(figsize=(12, 5))

    # BEFORE
    plt.subplot(1, 2, 1)
    plt.scatter(emb_before_2d[:, 0], emb_before_2d[:, 1], s=10)
    plt.title("Before Geometric Regularization")

    # AFTER
    plt.subplot(1, 2, 2)
    plt.scatter(emb_after_2d[:, 0], emb_after_2d[:, 1], s=10)
    plt.title("After Geometric Regularization")

    plt.tight_layout()
    plt.savefig("tsne_comparison.png", dpi=300)
    plt.show()

# -----------------------
# Main
# -----------------------
def main():
    texts = load_samples(500)

    tokenizer, base_model, geo_model = load_models()

    print("Encoding with base model...")
    emb_before = encode_texts(base_model, tokenizer, texts)

    print("Encoding with geo model...")
    emb_after = encode_texts(geo_model, tokenizer, texts)

    print("Running t-SNE...")
    plot_tsne(emb_before, emb_after)

    print("Saved as tsne_comparison.png")

if __name__ == "__main__":
    main()