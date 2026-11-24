#!/usr/bin/env python3
"""Isolated convergence-controlled reference-training launcher.

Usage:
 python thesis_extended_training.py --prepare
 python thesis_extended_training.py --smoke --device cuda
 python thesis_extended_training.py --run --device cuda

Does not modify archived reference experiments. Generated runner is based on the
verified project runner; audit its SHA before execution.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'thesis/experiments/run_e0.py'
OUTPUT_RUNNER = ROOT / 'thesis/experiments/run_thesis_extended.py'
RESULTS = ROOT / 'thesis/results/thesis_extended_reference'
AUDIT = ROOT / 'thesis/audits/thesis_extended_reference'
PAIRS = ('en-fr', 'en-ru', 'en-it', 'en-es')
SEEDS = range(1, 6)
VARIANTS = ('baseline', 'geo', 'full')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate():
    if not SOURCE.is_file():
        raise RuntimeError(f'Missing verified runner: {SOURCE}')
    s = SOURCE.read_text()
    replacements = [
        ('    EPOCHS,\n', '    EPOCHS as ORIGINAL_EPOCHS,\n'),
        ('    E0Run,\n', '    E0Run as OriginalRun,\n'),
        ('\n\ndef seed_everything(seed):', '''

EPOCHS = 40

class E0Run(OriginalRun):
    @property
    def output_dir(self):
        return Path("thesis/results/thesis_extended_reference") / self.run_id

    def metadata(self):
        d = super().metadata()
        d["epochs"] = EPOCHS
        d["experiment_family"] = "convergence_controlled_reference"
        return d


def seed_everything(seed):'''),
        ('    parser.add_argument("--chunk-size", type=int, default=256)',
         '    parser.add_argument("--chunk-size", type=int, default=256)\n'
         '    parser.add_argument("--patience", type=int, default=5)\n'
         '    parser.add_argument("--min-epochs", type=int, default=5)\n'
         '    parser.add_argument("--min-delta", type=float, default=0.0001)'),
        ('Path("thesis/results/e0_smoke")', 'Path("thesis/results/thesis_extended_smoke")'),
        ('    if args.chunk_size < 1:',
         '    if args.patience < 1 or args.min_epochs < 1 or args.min_delta < 0:\n'
         '        parser.error("Invalid early-stopping settings")\n'
         '    if args.chunk_size < 1:'),
        ('    if args.resume:\n        metadata = saved_metadata\n\n    for epoch in range(start_epoch, epochs):', '''    if args.resume:
        metadata = saved_metadata

    # Reconstruct patience from historical validation values on resume.
    no_improvement = 0
    stopping_best = -1.0
    if args.resume:
        for r in rows:
            score = float(r["Top-1"])
            if score > stopping_best + args.min_delta:
                stopping_best = score
                no_improvement = 0
            else:
                no_improvement += 1
        if start_epoch >= args.min_epochs and no_improvement >= args.patience:
            raise RuntimeError("Run already satisfies stopping criterion; refusing continuation")

    for epoch in range(start_epoch, epochs):'''),
        ('        if current_top1 > best_top1:',
         '        if current_top1 > stopping_best + args.min_delta:\n'
         '            stopping_best = current_top1\n'
         '            no_improvement = 0\n'
         '        else:\n'
         '            no_improvement += 1\n\n'
         '        if current_top1 > best_top1:'),
        ('            flush=True,\n        )\n\n    best_state = torch.load(', '''            flush=True,
        )
        if epoch + 1 >= args.min_epochs and no_improvement >= args.patience:
            print(f"EARLY STOP: epoch={epoch+1}, patience={args.patience}", flush=True)
            break

    best_state = torch.load('''),
        ('        "best_epoch": best_epoch,',
         '        "best_epoch": best_epoch,\n'
         '        "epochs_completed": epoch + 1,\n'
         '        "stopping_rule": {"min_epochs": args.min_epochs, "patience": args.patience, "min_delta": args.min_delta},'),
    ]
    for old, new in replacements:
        count = s.count(old)
        if count != 1:
            raise RuntimeError(f'Unexpected source structure; replacement count {count}: {old[:80]!r}')
        s = s.replace(old, new, 1)
    OUTPUT_RUNNER.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_RUNNER.exists():
        if OUTPUT_RUNNER.read_text() != s:
            raise RuntimeError(f'Existing generated runner differs; refusing overwrite: {OUTPUT_RUNNER}')
    else:
        OUTPUT_RUNNER.write_text(s)
    AUDIT.mkdir(parents=True, exist_ok=True)
    manifest = {'source': str(SOURCE.relative_to(ROOT)), 'source_sha256': digest(SOURCE),
                'generated_runner': str(OUTPUT_RUNNER.relative_to(ROOT)),
                'generated_sha256': digest(OUTPUT_RUNNER), 'max_epochs': 40,
                'min_epochs': 5, 'patience': 5, 'min_delta': 0.0001,
                'language_pairs': PAIRS, 'seeds': list(SEEDS), 'variants': VARIANTS,
                'output_root': str(RESULTS.relative_to(ROOT)),
                'status': 'prepared; not evidence of completed training'}
    manifest_file = AUDIT / 'implementation_manifest.json'
    if manifest_file.exists() and json.loads(manifest_file.read_text()) != json.loads(json.dumps(manifest)):
        raise RuntimeError('Implementation manifest changed; refusing to overwrite')
    manifest_file.write_text(json.dumps(manifest, indent=2))
    print('Prepared isolated runner:', OUTPUT_RUNNER.relative_to(ROOT))
    print('Runner SHA256:', manifest['generated_sha256'])


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--prepare', action='store_true')
    g.add_argument('--smoke', action='store_true')
    g.add_argument('--run', action='store_true')
    p.add_argument('--device', choices=['cuda','cpu'], default='cuda')
    p.add_argument('--resume-incomplete', action='store_true')
    args = p.parse_args()
    os.chdir(ROOT)
    generate()
    if args.prepare:
        return
    if args.smoke:
        # The verified smoke mode uses its own isolated output path.
        cmd = [sys.executable, str(OUTPUT_RUNNER), '--pair', 'en-fr', '--seed', '1',
               '--variant', 'baseline', '--device', args.device, '--smoke-test']
        print('SMOKE:', ' '.join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        return
    completed = 0
    for pair in PAIRS:
        for seed in SEEDS:
            for variant in VARIANTS:
                run_id = f'{pair}_seed{seed}_{variant}'
                folder = RESULTS / run_id
                if (folder / 'summary.json').is_file():
                    completed += 1
                    print('SKIP completed:', run_id, flush=True)
                    continue
                if folder.exists() and any(folder.iterdir()) and not args.resume_incomplete:
                    raise RuntimeError(f'Incomplete output at {folder}; rerun with --resume-incomplete after inspection')
                cmd = [sys.executable, str(OUTPUT_RUNNER), '--pair', pair,
                       '--seed', str(seed), '--variant', variant, '--device', args.device,
                       '--structural-every-epoch']
                if folder.exists() and any(folder.iterdir()):
                    cmd.append('--resume')
                print(f'RUN {completed+1}/60: {run_id}', flush=True)
                subprocess.run(cmd, check=True)
                completed += 1
    print('EXTENDED REFERENCE TRAINING COMPLETE:', completed, '/60')

if __name__ == '__main__':
    main()
