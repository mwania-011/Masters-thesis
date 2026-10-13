import argparse
import hashlib
import json
import random
from pathlib import Path

from datasets import load_dataset

LANGUAGE_PAIRS = ("en-fr", "en-ru", "en-it", "en-es")
SPLIT_SEED = 20261013
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10


def prepare_pair(lang_pair, output_dir):
    src_lang, tgt_lang = lang_pair.split("-")

    dataset = load_dataset("opus_books", lang_pair, split="train")

    # Preserve the journal's sentence-cleaning procedure.
    valid_records = []

    for original_index, sample in enumerate(dataset):
        translation = sample["translation"]
        src = translation[src_lang]
        tgt = translation[tgt_lang]

        if isinstance(src, str) and isinstance(tgt, str):
            src = src.strip()
            tgt = tgt.strip()

            if src and tgt:
                valid_records.append({
                    "original_index": original_index,
                    "source": src,
                    "target": tgt,
                })

    # Group identical bilingual sentence pairs before splitting.
    # This prevents exact-pair contamination across partitions.
    groups = {}

    for index, record in enumerate(valid_records):
        key = (record["source"], record["target"])
        groups.setdefault(key, []).append(index)

    group_keys = sorted(groups)
    random.Random(SPLIT_SEED).shuffle(group_keys)

    n = len(valid_records)
    target_train = int(n * TRAIN_RATIO)
    target_validation = int(n * VAL_RATIO)

    partitions = {
        "train": [],
        "validation": [],
        "test": [],
    }

    # Allocate complete duplicate groups, approximately preserving
    # the requested 80/10/10 row proportions.
    for key in group_keys:
        group_indices = groups[key]

        if len(partitions["train"]) < target_train:
            destination = "train"
        elif len(partitions["validation"]) < target_validation:
            destination = "validation"
        else:
            destination = "test"

        partitions[destination].extend(group_indices)

    pair_dir = output_dir / lang_pair
    pair_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "language_pair": lang_pair,
        "dataset": "opus_books",
        "split_seed": SPLIT_SEED,
        "source_examples": len(dataset),
        "valid_examples": n,
        "cleaning": "strip whitespace; remove empty/non-string pairs",
        "split_counts": {
            name: len(values)
            for name, values in partitions.items()
        },
        "partitions": {},
    }

    for split_name, split_indices in partitions.items():
        path = pair_dir / f"{split_name}.jsonl"

        digest = hashlib.sha256()

        with path.open("w", encoding="utf-8") as f:
            for index in split_indices:
                record = valid_records[index]
                line = json.dumps(record, ensure_ascii=False)
                f.write(line + "\n")
                digest.update((line + "\n").encode("utf-8"))

        manifest["partitions"][split_name] = {
            "file": path.name,
            "sha256": digest.hexdigest(),
        }

    with (pair_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{lang_pair}:")
    print(f"  Original:   {len(dataset):,}")
    print(f"  Valid:      {n:,}")

    for name, count in manifest["split_counts"].items():
        print(f"  {name:11s}: {count:,}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        default="thesis/data/processed",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    for lang_pair in LANGUAGE_PAIRS:
        prepare_pair(lang_pair, output_dir)


if __name__ == "__main__":
    main()
