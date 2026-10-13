import hashlib
import json
from pathlib import Path

from torch.utils.data import Dataset


SUPPORTED_PAIRS = ("en-fr", "en-ru", "en-it", "en-es")
SUPPORTED_SPLITS = ("train", "validation", "test")


class ThesisParallelDataset(Dataset):
    """Load frozen, duplicate-aware OPUS Books thesis partitions."""

    def __init__(
        self,
        tokenizer,
        lang_pair,
        split,
        data_root="thesis/data/processed",
        max_length=64,
        verify_hash=True,
    ):
        if lang_pair not in SUPPORTED_PAIRS:
            raise ValueError(f"Unsupported language pair: {lang_pair}")

        if split not in SUPPORTED_SPLITS:
            raise ValueError(f"Unsupported split: {split}")

        folder = Path(data_root) / lang_pair
        manifest_path = folder / "manifest.json"
        data_path = folder / f"{split}.jsonl"

        with manifest_path.open(encoding="utf-8") as f:
            manifest = json.load(f)

        if manifest["language_pair"] != lang_pair:
            raise ValueError("Manifest language-pair mismatch")

        if verify_hash:
            expected = manifest["partitions"][split]["sha256"]
            digest = hashlib.sha256(data_path.read_bytes()).hexdigest()

            if digest != expected:
                raise ValueError(
                    f"Dataset SHA-256 mismatch: {data_path}"
                )

        self.pairs = []

        with data_path.open(encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                self.pairs.append(
                    (record["source"], record["target"])
                )

        expected_count = manifest["split_counts"][split]

        if len(self.pairs) != expected_count:
            raise ValueError(
                f"Dataset count mismatch: {len(self.pairs)} "
                f"!= {expected_count}"
            )

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        source, target = self.pairs[index]

        enc_source = self.tokenizer(
            source,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        enc_target = self.tokenizer(
            target,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return (
            {k: v.squeeze(0) for k, v in enc_source.items()},
            {k: v.squeeze(0) for k, v in enc_target.items()},
        )
