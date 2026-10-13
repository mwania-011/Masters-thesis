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


@torch.no_grad()
def neighborhood_overlap_chunked(z1, z2, k=10, chunk_size=256):
    """
    Exact neighborhood overlap using chunked similarity calculations.

    Matches the journal's neighborhood_overlap_at_k definition.
    """
    if z1.shape != z2.shape or z1.ndim != 2:
        raise ValueError("Embedding shapes must match and be 2D")

    n = z1.shape[0]

    if not (1 <= k < n):
        raise ValueError("k must satisfy 1 <= k < n")

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    overlap_total = 0

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)

        sim1 = z1[start:end] @ z1.T
        sim2 = z2[start:end] @ z2.T

        local = torch.arange(end - start, device=z1.device)
        global_idx = torch.arange(start, end, device=z1.device)

        sim1[local, global_idx] = -1e9
        sim2[local, global_idx] = -1e9

        nn1 = sim1.topk(k, dim=1).indices
        nn2 = sim2.topk(k, dim=1).indices

        # Count matching neighbor indices for each query.
        overlap_counts = (
            nn1.unsqueeze(2) == nn2.unsqueeze(1)
        ).sum(dim=(1, 2))

        overlap_total += int(overlap_counts.sum().item())

    return float(overlap_total / (n * k))


@torch.no_grad()
def uniformity_chunked(z, t=2.0, chunk_size=256):
    """
    Exact all-pairs uniformity with bounded intermediate memory.

    Uses unordered pairs (i < j), matching torch.pdist.
    """
    if z.ndim != 2:
        raise ValueError("Expected a 2D embedding tensor")

    n = z.shape[0]

    if n < 2:
        raise ValueError("At least two embeddings are required")

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    z = F.normalize(z, dim=-1)

    total = 0.0
    count = 0

    for i in range(0, n, chunk_size):
        zi = z[i:i + chunk_size]

        for j in range(i, n, chunk_size):
            zj = z[j:j + chunk_size]

            # Squared Euclidean distance.
            distances = torch.cdist(
                zi, zj, p=2
            ).square()

            values = torch.exp(-t * distances)

            if i == j:
                values = torch.triu(values, diagonal=1)
                block_count = (
                    zi.shape[0] * (zi.shape[0] - 1) // 2
                )
            else:
                block_count = zi.shape[0] * zj.shape[0]

            total += values.sum(dtype=torch.float64).item()
            count += block_count

    mean_value = total / count

    return float(np.log(mean_value + 1e-12))
