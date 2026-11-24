#!/usr/bin/env python3
"""Regenerate the four supplied thesis figures while removing the marked labels.

Input:  thesis_prospective_figures.zip (contains the four original PNG files)
Output: thesis_prospective_figures_cleaned.zip and a folder of cleaned PNGs

This is a pixel-preserving edit of the supplied PNG figures. The ZIP contained
only raster images (no plotting script or numerical source data), so this
script preserves the plotted curves/points exactly rather than pretending to
reconstruct unavailable experimental data.

Required package: Pillow
    python -m pip install Pillow
Usage:
    python regenerate_figures.py thesis_prospective_figures.zip
"""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from PIL import Image, ImageDraw
import sys

FILES = [
    "figure_A_gating_dynamics.png",
    "figure_B_diagnostic_predictive_relationship.png",
    "figure_C_embedding_spectrum.png",
    "figure_D_seed_consistency.png",
]


def erase_marked_labels(name: str, image: Image.Image) -> Image.Image:
    """Remove only the three marked text labels; leave plotted data untouched."""
    im = image.convert("RGB")
    draw = ImageDraw.Draw(im)

    if name == "figure_B_diagnostic_predictive_relationship.png":
        # Remove the small top-left legend: its line sample and the words
        # “Illustrative trend”. The horizontal zero reference line is lower.
        draw.rectangle((163, 78, 493, 121), fill="white")
        # Restore the faint vertical gridline that passes behind the legend.
        draw.line((417, 72, 417, 804), fill=(235, 235, 235), width=2)

    elif name == "figure_C_embedding_spectrum.png":
        # Remove “Figure C. Illustrative embedding covariance spectra”.
        # This is in the white title margin above the plotting axes.
        draw.rectangle((360, 13, 1410, 64), fill="white")

    elif name == "figure_D_seed_consistency.png":
        # Remove “Figure D. Seed-level consistency of intervention effects”.
        draw.rectangle((330, 13, 1430, 64), fill="white")

    # Figure A has no marked label to remove.
    return im


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python regenerate_figures.py thesis_prospective_figures.zip")

    source_zip = Path(sys.argv[1]).expanduser().resolve()
    if not source_zip.is_file():
        raise SystemExit(f"Input ZIP not found: {source_zip}")

    output_dir = source_zip.parent / "thesis_figures_cleaned"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_zip = source_zip.parent / "thesis_prospective_figures_cleaned.zip"

    with ZipFile(source_zip, "r") as zin:
        available = set(zin.namelist())
        missing = [f for f in FILES if f not in available]
        if missing:
            raise SystemExit("ZIP is missing expected files: " + ", ".join(missing))

        for filename in FILES:
            with zin.open(filename) as fp:
                original = Image.open(fp)
                original.load()
            cleaned = erase_marked_labels(filename, original)
            cleaned.save(output_dir / filename, format="PNG", optimize=True)

    with ZipFile(output_zip, "w", compression=ZIP_DEFLATED) as zout:
        for filename in FILES:
            path = output_dir / filename
            zout.write(path, arcname=filename)

    print(f"Saved cleaned figures to: {output_dir}")
    print(f"Saved ZIP archive to:     {output_zip}")


if __name__ == "__main__":
    main()
