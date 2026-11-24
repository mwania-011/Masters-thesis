#!/usr/bin/env python3
"""E1-A validation-only extraction from a frozen E0 last.pt checkpoint.
Run from the Multilingual-Thesis repository root. Never writes to E0.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import XLMRobertaTokenizer

import train_structure_preserving_multilingual as journal
from thesis.data.dataset import ThesisParallelDataset
from thesis.experiments.e0_config import BATCH_SIZE, MAX_LENGTH, E0Run
from thesis.experiments.run_e0 import make_loader, encode_to_cpu
from thesis.evaluation.retrieval import compute_retrieval_stats_chunked


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pair', default='en-ru')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--chunk-size', type=int, default=256)
    args = p.parse_args()
    if args.pair not in ('en-fr', 'en-ru', 'en-it', 'en-es') or args.seed not in range(1, 6):
        p.error('Unsupported pair or seed')
    if args.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA unavailable')
    run = E0Run(args.pair, args.seed, 'baseline')
    ckpt_path = run.output_dir / 'last.pt'
    if not ckpt_path.is_file():
        p.error(f'Missing {ckpt_path}')
    out = Path('thesis/results/e1/a_embeddings') / run.run_id
    if out.exists():
        p.error(f'Refusing to overwrite existing output directory: {out}')
    ckpt_hash = digest(ckpt_path)
    # Only deserialize checkpoints created by your trusted E0 training pipeline.
    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if state.get('run') != run.metadata():
        raise RuntimeError('Checkpoint run metadata mismatch')
    epoch = int(state['epoch'])
    if epoch != 10:
        raise RuntimeError(f'Expected epoch 10, found {epoch}')
    manifest = Path('thesis/data/processed') / args.pair / 'manifest.json'
    if state['metadata']['manifest_sha256'] != digest(manifest):
        raise RuntimeError('Checkpoint dataset manifest hash mismatch')
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    dataset = ThesisParallelDataset(tokenizer, args.pair, 'validation', max_length=MAX_LENGTH)
    loader = make_loader(dataset, BATCH_SIZE, False, args.seed)
    device = torch.device('cuda:0' if args.device == 'cuda' else 'cpu')
    model = journal.GeoTransformer()
    model.load_state_dict(state['model'], strict=True)
    model.to(device).eval()
    del state
    source, target = encode_to_cpu(model, loader, device)
    if source.shape != target.shape or source.shape != (len(dataset), 256):
        raise RuntimeError(f'Unexpected embedding shapes: {source.shape}, {target.shape}')
    if not torch.isfinite(source).all() or not torch.isfinite(target).all():
        raise RuntimeError('Non-finite embeddings')
    for name, z in [('source', source), ('target', target)]:
        err = (z.norm(dim=1) - 1).abs().max().item()
        if err > 1e-4:
            raise RuntimeError(f'{name} not L2-normalized; max error={err}')
    metrics = compute_retrieval_stats_chunked(source, target, chunk_size=args.chunk_size)
    with (run.output_dir / 'epochs.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    matched = [row for row in rows if int(row['Epoch']) == epoch]
    if len(matched) != 1:
        raise RuntimeError('Cannot identify unique epoch-10 validation row')
    reference_top1 = float(matched[0]['Top-1'])
    observed_top1 = float(metrics['Top-1'])
    difference = abs(observed_top1 - reference_top1)
    # Same model, split, evaluation and ordering should agree up to numerical tolerance.
    if difference > 1e-5:
        raise RuntimeError(f'E0 Top-1 mismatch: observed={observed_top1}, reference={reference_top1}, delta={difference}')
    out.mkdir(parents=True, exist_ok=False)
    npz = out / 'validation_embeddings.npz'
    np.savez_compressed(npz, source=source.numpy(), target=target.numpy())
    report = {
        'run_id': run.run_id, 'checkpoint': str(ckpt_path), 'checkpoint_sha256': ckpt_hash,
        'checkpoint_epoch': epoch, 'split': 'validation', 'dataset_manifest_sha256': digest(manifest),
        'n_pairs': len(dataset), 'embedding_dimension': 256,
        'e0_validation_metrics': metrics, 'e0_epoch10_top1': reference_top1,
        'top1_absolute_difference': difference, 'npz_sha256': digest(npz),
        'device': str(device), 'status': 'PASS',
    }
    (out / 'extraction_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    print('EXTRACTED:', npz)

if __name__ == '__main__':
    main()
