from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

from train_structure_preserving_multilingual import ExperimentConfig


LANGUAGE_PAIRS = ("en-fr", "en-ru", "en-it", "en-es")
SEEDS = (1, 2, 3, 4, 5)
VARIANTS = ("baseline", "geo", "full")

EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
MAX_LENGTH = 64
PROJECTION_DIM = 256

DATA_ROOT = Path("thesis/data/processed")
OUTPUT_ROOT = Path("thesis/results/e0")

# Preserve original journal topology coefficients.
TOPOLOGY_WEIGHTS = {
    "en-fr": 0.0005,
    "en-ru": 0.0004,
    "en-it": 0.0002,
    "en-es": 0.0002,
}


@dataclass(frozen=True)
class E0Run:
    language_pair: str
    seed: int
    variant: str

    @property
    def run_id(self):
        return (
            f"{self.language_pair}_"
            f"seed{self.seed}_"
            f"{self.variant}"
        )

    @property
    def output_dir(self):
        return OUTPUT_ROOT / self.run_id

    def experiment_config(self):
        return ExperimentConfig(
            name=self.variant,
            use_align=True,
            use_geo=self.variant in ("geo", "full"),
            use_topo=self.variant == "full",
            lambda_geo=0.0005,
            lambda_topo=TOPOLOGY_WEIGHTS[self.language_pair]
            if self.variant == "full" else 0.0,
        )

    def metadata(self):
        return {
            **asdict(self),
            "run_id": self.run_id,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "max_length": MAX_LENGTH,
            "projection_dim": PROJECTION_DIM,
            "experiment": asdict(self.experiment_config()),
        }


def all_runs():
    return [
        E0Run(pair, seed, variant)
        for pair, seed, variant in product(
            LANGUAGE_PAIRS, SEEDS, VARIANTS
        )
    ]


if __name__ == "__main__":
    runs = all_runs()

    assert len(runs) == 60
    assert len({run.run_id for run in runs}) == 60

    for run in runs:
        assert (DATA_ROOT / run.language_pair / "manifest.json").exists()

    for run in runs[:6]:
        print(run.metadata())

    print(f"\nPASS: {len(runs)} E0 configurations verified.")
