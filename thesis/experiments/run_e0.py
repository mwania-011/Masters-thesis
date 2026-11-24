import argparse
import csv
import hashlib
import json
import random
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from transformers import XLMRobertaTokenizer

import train_structure_preserving_multilingual as journal
from thesis.data.dataset import ThesisParallelDataset
from thesis.evaluation.retrieval import (
    compute_retrieval_stats_chunked,
    neighborhood_overlap_chunked,
    uniformity_chunked,
)
from thesis.experiments.e0_config import (
    BATCH_SIZE,
    EPOCHS,
    LANGUAGE_PAIRS,
    LEARNING_RATE,
    MAX_LENGTH,
    SEEDS,
    VARIANTS,
    E0Run,
)


def seed_everything(seed):
    journal.set_seed(seed)


def git_revision():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_loader(dataset, batch_size, shuffle, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )


def encode_to_cpu(model, loader, device):
    model.eval()
    left, right = [], []

    with torch.inference_mode():
        for x1, x2 in loader:
            x1 = {k: v.to(device) for k, v in x1.items()}
            x2 = {k: v.to(device) for k, v in x2.items()}

            left.append(model(**x1).cpu())
            right.append(model(**x2).cpu())

    return torch.cat(left), torch.cat(right)


def evaluate(model, loader, device, chunk_size, structural):
    z1, z2 = encode_to_cpu(model, loader, device)

    results = compute_retrieval_stats_chunked(
        z1, z2, chunk_size=chunk_size
    )

    if structural:
        results["NeighborOverlap@10"] = neighborhood_overlap_chunked(
            z1, z2, k=10, chunk_size=chunk_size
        )
        results["Uniformity"] = uniformity_chunked(
            torch.cat((z1, z2)),
            t=2.0,
            chunk_size=chunk_size,
        )

    return results


def save_checkpoint(path, model, optimizer, epoch, best_top1,
                    run, train_generator, metadata):
    state = {
        "epoch": epoch,
        "best_validation_top1": best_top1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "run": run.metadata(),
        "metadata": metadata,
        "rng_python": random.getstate(),
        "rng_numpy": np.random.get_state(),
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available() else None,
        "loader_generator": train_generator.get_state(),
    }
    torch.save(state, path)



def save_best_checkpoint(path, model, epoch, best_top1, run, metadata):
    """Save validation-selected weights without optimizer state."""
    torch.save(
        {
            "epoch": epoch,
            "best_validation_top1": best_top1,
            "model": model.state_dict(),
            "run": run.metadata(),
            "metadata": metadata,
            "checkpoint_type": "best_model",
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", choices=LANGUAGE_PAIRS, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted E0 run from last.pt",
    )
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--structural-every-epoch", action="store_true")
    args = parser.parse_args()

    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive")
    if args.resume and (args.smoke_test or args.dry_run):
        parser.error("--resume cannot be combined with --smoke-test or --dry-run")

    run = E0Run(args.pair, args.seed, args.variant)
    cfg = run.experiment_config()

    manifest_path = (
        Path("thesis/data/processed")
        / args.pair
        / "manifest.json"
    )

    metadata = {
        **run.metadata(),
        "git_revision": git_revision(),
        "manifest_sha256": sha256(manifest_path),
        "journal_sha256": sha256(
            "train_structure_preserving_multilingual.py"
        ),
        "evaluator_sha256": sha256(
            "thesis/evaluation/retrieval.py"
        ),
        "runner_sha256": sha256(__file__),
        "device": args.device,
        "smoke_test": args.smoke_test,
        "structural_every_epoch": args.structural_every_epoch,
        "torch_version": torch.__version__,
    }

    print(json.dumps(metadata, indent=2))

    if args.dry_run:
        print("PASS: Dry run completed; no model loaded.")
        return

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    if args.smoke_test:
        output_dir = Path("thesis/results/e0_smoke") / run.run_id
    else:
        output_dir = run.output_dir

    if args.resume:
        if not (output_dir / "last.pt").is_file():
            raise RuntimeError("Cannot resume: last.pt is missing")
        if not (output_dir / "best.pt").is_file():
            raise RuntimeError("Cannot resume: best.pt is missing")
        if (output_dir / "summary.json").exists():
            raise RuntimeError("Cannot resume an already completed run")
    else:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise RuntimeError(
                f"Output directory already contains files: {output_dir}"
            )
        output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda:0" if args.device == "cuda" else "cpu"
    )
    journal.DEVICE = device

    seed_everything(args.seed)

    tokenizer = XLMRobertaTokenizer.from_pretrained(
        "xlm-roberta-base"
    )

    train_dataset = ThesisParallelDataset(
        tokenizer=tokenizer,
        lang_pair=args.pair,
        split="train",
        max_length=MAX_LENGTH,
    )
    val_dataset = ThesisParallelDataset(
        tokenizer=tokenizer,
        lang_pair=args.pair,
        split="validation",
        max_length=MAX_LENGTH,
    )
    test_dataset = ThesisParallelDataset(
        tokenizer=tokenizer,
        lang_pair=args.pair,
        split="test",
        max_length=MAX_LENGTH,
    )

    if args.smoke_test:
        train_dataset = Subset(train_dataset, range(32))
        val_dataset = Subset(val_dataset, range(32))
        test_dataset = Subset(test_dataset, range(32))

    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed + 10000)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=train_generator,
        num_workers=0,
    )

    val_loader = make_loader(
        val_dataset, BATCH_SIZE, False, args.seed
    )
    test_loader = make_loader(
        test_dataset, BATCH_SIZE, False, args.seed
    )

    # Reset before initialization to match model initialization
    # across variants with the same language pair and seed.
    seed_everything(args.seed)

    model = journal.GeoTransformer().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    epochs = 1 if args.smoke_test else EPOCHS
    best_top1 = -1.0
    best_epoch = None
    start_epoch = 0
    log_path = output_dir / "epochs.csv"

    if args.resume:
        saved_metadata = json.loads(
            (output_dir / "metadata.json").read_text()
        )
        state = torch.load(
            output_dir / "last.pt",
            map_location="cpu",
            weights_only=False,
        )
        best_state = torch.load(
            output_dir / "best.pt",
            map_location="cpu",
            weights_only=False,
        )

        for key in (
            "run_id", "manifest_sha256", "journal_sha256",
            "evaluator_sha256", "device", "smoke_test",
            "structural_every_epoch", "torch_version",
        ):
            if saved_metadata[key] != metadata[key]:
                raise RuntimeError(f"Resume metadata mismatch: {key}")
            if state["metadata"][key] != saved_metadata[key]:
                raise RuntimeError(f"Checkpoint metadata mismatch: {key}")

        if state["run"] != run.metadata():
            raise RuntimeError("Checkpoint run configuration mismatch")
        if best_state["run"] != run.metadata():
            raise RuntimeError("Best checkpoint run mismatch")

        start_epoch = int(state["epoch"])
        best_epoch = int(best_state["epoch"])
        best_top1 = float(state["best_validation_top1"])

        if not (0 < start_epoch < epochs):
            raise RuntimeError("Invalid resume epoch")
        if not (0 < best_epoch <= start_epoch):
            raise RuntimeError("Invalid best epoch")
        if abs(
            float(best_state["best_validation_top1"]) - best_top1
        ) > 1e-10:
            raise RuntimeError("Best checkpoint score mismatch")

        with log_path.open(newline="") as f:
            rows = list(csv.DictReader(f))

        recorded_epochs = [int(row["Epoch"]) for row in rows]
        if recorded_epochs != list(range(1, start_epoch + 1)):
            raise RuntimeError("epochs.csv does not match checkpoint")

        if abs(
            float(rows[best_epoch - 1]["Top-1"]) - best_top1
        ) > 1e-6:
            raise RuntimeError("Best validation score differs from epochs.csv")

        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])

        # Restore RNG only after model/optimizer initialization.
        random.setstate(state["rng_python"])
        np.random.set_state(state["rng_numpy"])
        torch.set_rng_state(state["rng_torch"])
        if args.device == "cuda":
            if state["rng_cuda"] is None:
                raise RuntimeError("Missing CUDA RNG state")
            torch.cuda.set_rng_state_all(state["rng_cuda"])
        train_generator.set_state(state["loader_generator"])

        print(
            f"RESUMING {run.run_id}: "
            f"completed={start_epoch}, next={start_epoch + 1}, "
            f"best_epoch={best_epoch}",
            flush=True,
        )
    else:
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

    if args.resume:
        metadata = saved_metadata

    for epoch in range(start_epoch, epochs):
        train_stats = journal.train_one_epoch(
            model, train_loader, optimizer, cfg, epoch
        )

        val_stats = evaluate(
            model,
            val_loader,
            device,
            args.chunk_size,
            structural=args.structural_every_epoch,
        )

        row = {
            "Epoch": epoch + 1,
            **train_stats,
            **val_stats,
        }

        with log_path.open("a", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=list(row.keys())
            )
            if epoch == 0 and not args.resume:
                writer.writeheader()
            writer.writerow(row)

        current_top1 = val_stats["Top-1"]

        if current_top1 > best_top1:
            best_top1 = current_top1
            best_epoch = epoch + 1
            save_best_checkpoint(
                output_dir / "best.pt",
                model, epoch + 1,
                best_top1, run, metadata,
            )

        save_checkpoint(
            output_dir / "last.pt",
            model, optimizer, epoch + 1,
            best_top1, run, train_generator, metadata,
        )

        print(
            f"[{run.run_id}] "
            f"Epoch {epoch + 1}/{epochs} "
            f"TrainLoss={train_stats['TrainLoss']:.5f} "
            f"ValTop1={current_top1:.5f} "
            f"BestTop1={best_top1:.5f}",
            flush=True,
        )

    best_state = torch.load(
        output_dir / "best.pt",
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(best_state["model"])

    # Re-evaluate best model and evaluate held-out test only once.
    best_val = evaluate(
        model, val_loader, device,
        args.chunk_size, structural=True,
    )
    test = evaluate(
        model, test_loader, device,
        args.chunk_size, structural=True,
    )

    summary = {
        "run_id": run.run_id,
        "smoke_test": args.smoke_test,
        "best_epoch": best_epoch,
        "best_validation": best_val,
        "held_out_test": test,
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
