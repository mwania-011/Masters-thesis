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

    # Group records connected by identical source OR target sentences.
    # All records in a connected component enter the same partition.
    n = len(valid_records)
    parent = list(range(n))
    sizes = [1] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if sizes[ra] < sizes[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        sizes[ra] += sizes[rb]

    first_source = {}
    first_target = {}

    for i, record in enumerate(valid_records):
        src = record["source"]
        tgt = record["target"]

        if src in first_source:
            union(i, first_source[src])
        else:
            first_source[src] = i

        if tgt in first_target:
            union(i, first_target[tgt])
        else:
            first_target[tgt] = i

    components = {}

    for i in range(n):
        components.setdefault(find(i), []).append(i)

    component_list = list(components.values())
    component_list.sort(key=lambda group: group[0])

    random.Random(SPLIT_SEED).shuffle(component_list)

    partitions = {
        "train": [],
        "validation": [],
        "test": [],
    }

    target_counts = {
        "train": int(n * TRAIN_RATIO),
        "validation": int(n * VAL_RATIO),
    }

    # Allocate complete components; actual counts may vary slightly.
    for component in component_list:
        if len(partitions["train"]) < target_counts["train"]:
            destination = "train"
        elif len(partitions["validation"]) < target_counts["validation"]:
            destination = "validation"
        else:
            destination = "test"

        partitions[destination].extend(component)

    print(
        f"  Connected components: {len(component_list):,}; "
        f"largest: {max(map(len, component_list)):,}"
    )

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
