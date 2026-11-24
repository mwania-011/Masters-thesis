#!/usr/bin/env python3
"""Non-overwriting sequential launcher for E1-B en-ru seeds 2-5."""

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "thesis/experiments/run_e1b.py"
RESULTS = ROOT / "thesis/results/e1/b_interventions/full"
LOGS = ROOT / "thesis/audits/e1/e1b_batch_logs"

PAIR = "en-ru"
SEEDS = (2, 3, 4, 5)
EXPECTED_STEPS = 438
EXPECTED_BRANCHES = {"control", "geo", "topo", "full"}

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def timestamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

def target(seed):
    return RESULTS / f"{PAIR}_seed{seed}"

def inspect():
    if not RUNNER.is_file():
        raise RuntimeError(f"Missing runner: {RUNNER}")

    if not RESULTS.is_dir():
        raise RuntimeError(f"Missing results directory: {RESULTS}")

    pilot = target(1)
    pilot_json = pilot / "results.json"

    if not pilot_json.is_file():
        raise RuntimeError("Completed seed-1 pilot not found")

    pilot_data = json.loads(pilot_json.read_text())
    if (
        pilot_data.get("status") != "PASS"
        or pilot_data.get("pair") != PAIR
        or pilot_data.get("seed") != 1
        or pilot_data.get("steps") != EXPECTED_STEPS
    ):
        raise RuntimeError("Seed-1 pilot metadata is unexpected")

    pilot_hash = sha256(pilot_json)
    runner_hash = sha256(RUNNER)

    if pilot_data.get("code_sha256") != runner_hash:
        raise RuntimeError(
            "Runner differs from the completed pilot; "
            "do not mix implementations"
        )

    print("=== E1-B BATCH PREFLIGHT ===", flush=True)
    print(f"Repository: {ROOT}", flush=True)
    print(f"Pair: {PAIR}", flush=True)
    print(f"Seeds: {SEEDS}", flush=True)
    print(f"Steps per branch: {EXPECTED_STEPS}", flush=True)
    print(f"Runner SHA256: {runner_hash}", flush=True)
    print(f"Seed-1 pilot SHA256: {pilot_hash}", flush=True)

    for seed in SEEDS:
        out = target(seed)

        if out.exists() or out.is_symlink():
            raise RuntimeError(
                f"REFUSING OVERWRITE: target already exists: {out}"
            )

        checkpoint = (
            ROOT
            / "thesis/results/e0"
            / f"{PAIR}_seed{seed}_baseline"
            / "last.pt"
        )

        if not checkpoint.is_file():
            raise RuntimeError(
                f"Missing baseline checkpoint: {checkpoint}"
            )

        print(
            f"seed {seed}: READY | "
            f"checkpoint={checkpoint.name} | "
            f"output absent",
            flush=True,
        )

    print("BATCH PREFLIGHT: PASS", flush=True)
    return pilot_hash, runner_hash

def run_batch():
    LOGS.mkdir(parents=True, exist_ok=True)

    lock_path = LOGS / "enru_seeds2to5.lock"

    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(
                lock.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB
            )
        except BlockingIOError:
            raise RuntimeError(
                "Another E1-B en-ru batch launcher is running"
            )

        pilot_hash, runner_hash = inspect()

        batch_id = timestamp()
        batch_dir = LOGS / f"batch_{batch_id}"

        # Exclusive creation prevents replacing an existing log batch.
        batch_dir.mkdir(parents=False, exist_ok=False)

        manifest = {
            "phase": "E1-B",
            "pair": PAIR,
            "seeds": list(SEEDS),
            "expected_steps_per_branch": EXPECTED_STEPS,
            "runner_sha256": runner_hash,
            "seed1_pilot_results_sha256": pilot_hash,
            "batch_id": batch_id,
            "status": "RUNNING",
            "runs": [],
        }

        manifest_path = batch_dir / "manifest.json"

        def save_manifest():
            tmp = batch_dir / "manifest.json.tmp"
            tmp.write_text(json.dumps(manifest, indent=2))
            tmp.replace(manifest_path)

        save_manifest()

        try:
            for seed in SEEDS:
                out = target(seed)

                if out.exists() or out.is_symlink():
                    raise RuntimeError(
                        f"Output appeared during execution: {out}"
                    )

                command = [
                    sys.executable,
                    str(RUNNER),
                    "--mode", "run",
                    "--pair", PAIR,
                    "--seed", str(seed),
                    "--device", "cuda",
                    "--approve-full",
                ]

                log_path = batch_dir / f"seed{seed}.log"

                entry = {
                    "seed": seed,
                    "status": "RUNNING",
                    "log": str(log_path),
                    "output": str(out),
                    "command": command,
                }

                manifest["runs"].append(entry)
                save_manifest()

                print(
                    f"\n=== RUNNING en-ru seed {seed} ===",
                    flush=True
                )

                env = os.environ.copy()
                env["PYTHONPATH"] = str(ROOT)

                with log_path.open("x") as log:
                    process = subprocess.Popen(
                        command,
                        cwd=ROOT,
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                    )

                    for line in process.stdout:
                        print(line, end="", flush=True)
                        log.write(line)
                        log.flush()

                    returncode = process.wait()

                entry["returncode"] = returncode

                if returncode != 0:
                    raise RuntimeError(
                        f"Seed {seed} failed with exit code "
                        f"{returncode}; inspect {log_path}"
                    )

                result_path = out / "results.json"

                if not result_path.is_file():
                    raise RuntimeError(
                        f"Missing result artifact for seed {seed}"
                    )

                data = json.loads(result_path.read_text())

                if (
                    data.get("status") != "PASS"
                    or data.get("mode") != "run"
                    or data.get("pair") != PAIR
                    or data.get("seed") != seed
                    or data.get("steps") != EXPECTED_STEPS
                    or data.get("validation_limit") is not None
                    or data.get("code_sha256") != runner_hash
                ):
                    raise RuntimeError(
                        f"Result metadata failed for seed {seed}"
                    )

                branches = data.get("results", {})

                if set(branches) != EXPECTED_BRANCHES:
                    raise RuntimeError(
                        f"Incorrect branch set for seed {seed}"
                    )

                for branch in EXPECTED_BRANCHES:
                    r = branches[branch]

                    if (
                        r.get("optimizer_steps")
                        != EXPECTED_STEPS
                        or len(r.get("losses", []))
                        != EXPECTED_STEPS
                    ):
                        raise RuntimeError(
                            f"Incomplete branch: seed {seed}, "
                            f"{branch}"
                        )

                entry["status"] = "PASS"
                entry["results_sha256"] = sha256(result_path)
                save_manifest()

                print(
                    f"=== en-ru seed {seed}: PASS ===",
                    flush=True
                )

            if sha256(target(1) / "results.json") != pilot_hash:
                raise RuntimeError(
                    "Seed-1 pilot changed during batch"
                )

            if sha256(RUNNER) != runner_hash:
                raise RuntimeError(
                    "Runner changed during batch"
                )

            manifest["status"] = "PASS"
            save_manifest()

            print("\n=== E1-B BATCH PASS ===", flush=True)
            print(f"Manifest: {manifest_path}", flush=True)

        except BaseException as exc:
            manifest["status"] = "FAILED"
            manifest["error"] = str(exc)

            if manifest["runs"]:
                last = manifest["runs"][-1]
                if last["status"] == "RUNNING":
                    last["status"] = "FAILED"

            save_manifest()
            raise

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute four full-epoch experiments"
    )
    args = parser.parse_args()

    if args.execute:
        run_batch()
    else:
        inspect()
        print("DRY RUN ONLY: no training started", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"BATCH BLOCKED/FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
