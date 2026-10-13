import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def compute_retrieval_stats_chunked(z1, z2, chunk_size=256):
    """
    Exact source-to-target retrieval metrics using row chunks.

    Assumes corresponding translations share the same row index.
    Preserves the original journal evaluator's ranking definition.
    """
    if z1.ndim != 2 or z2.ndim != 2:
        raise ValueError("Expected two-dimensional embedding tensors")

    if z1.shape != z2.shape:
        raise ValueError("Embedding shapes must match")

    n = z1.shape[0]

    if n < 2:
        raise ValueError("At least two sentence pairs are required")

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    ranks = []
    positives = []
    hard_negatives = []

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)

        sim = z1[start:end] @ z2.T

        targets = torch.arange(start, end, device=sim.device)
        local_rows = torch.arange(end - start, device=sim.device)

        # Match the original evaluator's argsort-based ranks.
        sorted_indices = torch.argsort(
            sim, dim=1, descending=True
        )

        rank = (
            (sorted_indices == targets[:, None])
            .nonzero(as_tuple=False)[:, 1] + 1
        )

        pos = sim[local_rows, targets]

        negative_sim = sim.clone()
        negative_sim[local_rows, targets] = -torch.inf
        hard_neg = negative_sim.max(dim=1).values

        ranks.append(rank.cpu())
        positives.append(pos.cpu())
        hard_negatives.append(hard_neg.cpu())

    ranks = torch.cat(ranks).numpy()
    positives = torch.cat(positives).numpy()
    hard_negatives = torch.cat(hard_negatives).numpy()

    return {
        "Top-1": float(np.mean(ranks <= 1)),
        "Top-5": float(np.mean(ranks <= 5)),
        "MeanRank": float(np.mean(ranks)),
        "MedianRank": float(np.median(ranks)),
        "PosSimMean": float(np.mean(positives)),
        "HardNegMean": float(np.mean(hard_negatives)),
        "MarginMean": float(
            np.mean(positives - hard_negatives)
        ),
    }
