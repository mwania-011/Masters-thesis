#!/usr/bin/env python3
"""Controlled E1-B expansion for registered remaining language pairs."""

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "thesis/experiments/run_e1b.py"
REG = ROOT / "thesis/audits/e1/e1b_protocol_registration_v1"
BASE = ROOT / "thesis/results/e1/b_interventions"
LOGS = ROOT / "thesis/audits/e1/e1b_remaining_logs"

BRANCHES = ("control", "geo", "topo", "full")
PAIRS = ("en-it", "en-fr", "en-es")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_plan(pair):
    record = json.loads((REG / "registration.json").read_text())

    for filename, key in (
        ("analysis_protocol.md", "protocol_sha256"),
        ("execution_matrix.csv", "execution_matrix_sha256"),
    ):
        path = REG / filename
        if sha(path) != record[key]:
            raise RuntimeError(f"Registered artifact changed: {path}")

    with (REG / "execution_matrix.csv").open(newline="") as f:
        all_rows = list(csv.DictReader(f))

    if len(all_rows) != 15:
        raise RuntimeError("Unexpected registered matrix size")

    expected = {(p, s) for p in PAIRS for s in range(1, 6)}
    observed = {(r["pair"], int(r["seed"])) for r in all_rows}

    if observed != expected or len(observed) != len(all_rows):
        raise RuntimeError("Registered experimental units mismatch")

    if sum(int(r["total_optimizer_steps"]) for r in all_rows) != 126480:
        raise RuntimeError("Registered optimizer-step total mismatch")

    rows = sorted(
        (r for r in all_rows if r["pair"] == pair),
        key=lambda r: int(r["seed"]),
    )

    expected_steps = {
        "en-it": 809,
        "en-fr": 3178,
        "en-es": 2337,
    }[pair]

    expected_topo = {
        "en-it": 0.0002,
        "en-fr": 0.0005,
        "en-es": 0.0002,
    }[pair]

    for r in rows:
        if (
            r["branches"] != "control|geo|topo|full"
            or float(r["lambda_geo"]) != 0.0005
            or float(r["lambda_topo"]) != expected_topo
            or int(r["steps_per_branch"]) != expected_steps
            or int(r["total_optimizer_steps"]) != 4 * expected_steps
            or r["status"] != "PLANNED_NOT_STARTED"
        ):
            raise RuntimeError(f"Unexpected matrix row: {r}")

    return rows, record


def paths(pair, seed):
    name = f"{pair}_seed{seed}"
    checkpoint = (
        ROOT / "thesis/results/e0"
        / f"{name}_baseline" / "last.pt"
    )
    gate = BASE / "smoke" / name / "smoke_gate.json"
    output = BASE / "full" / name
    return checkpoint, gate, output


def check_gate(pair, seed, checkpoint_hash, runner_hash):
    _, path, _ = paths(pair, seed)

    if not path.is_file():
        raise RuntimeError(f"Missing smoke gate: {path}")

    data = json.loads(path.read_text())

    expected = {
        "status": "PASS",
        "mode": "smoke-test",
        "pair": pair,
        "seed": seed,
        "device": "cuda",
        "checkpoint_sha256": checkpoint_hash,
        "code_sha256": runner_hash,
        "replay_passed": True,
    }

    for key, value in expected.items():
        if data.get(key) != value:
            raise RuntimeError(
                f"Smoke gate mismatch: {pair} seed {seed}, {key}"
            )

    if data.get("steps") != 2:
        raise RuntimeError(f"Unexpected smoke length: {path}")

    return data


def inspect(pair, mode):
    rows, record = load_plan(pair)

    if not RUNNER.is_file():
        raise RuntimeError("Missing E1-B runner")

    runner_hash = sha(RUNNER)

    # Protect the existing, successfully audited runner version.
    expected_runner_hash = (
        "00d0701b70c7714f2d5f4a8b272ed2dd67f8079939064bb7f063a43a435521bd"
    )
    if runner_hash != expected_runner_hash:
        raise RuntimeError("E1-B runner differs from audited version")

    print(f"=== E1-B {pair} {mode.upper()} PREFLIGHT ===")
    print("Runner SHA256:", runner_hash)
    print("Protocol SHA256:", record["protocol_sha256"])
    print("Matrix SHA256:", record["execution_matrix_sha256"])

    for r in rows:
        seed = int(r["seed"])
        checkpoint, gate, output = paths(pair, seed)

        if not checkpoint.is_file():
            raise RuntimeError(f"Missing checkpoint: {checkpoint}")

        checkpoint_hash = sha(checkpoint)

        if output.exists() or output.is_symlink():
            raise RuntimeError(f"REFUSING OVERWRITE: {output}")

        smoke_dir = gate.parent

        if mode == "smoke":
            if smoke_dir.exists() or smoke_dir.is_symlink():
                # Existing smoke outputs are never overwritten.
                check_gate(pair, seed, checkpoint_hash, runner_hash)
                state = "EXISTING_GATE_PASS"
            else:
                state = "SMOKE_REQUIRED"
        else:
            if gate.is_file():
                check_gate(pair, seed, checkpoint_hash, runner_hash)
                state = "GATE_PASS"
            elif mode == "run":
                raise RuntimeError(f"Full run blocked: missing gate {gate}")
            else:
                state = "GATE_MISSING"

        print(
            f"seed={seed} steps={r['steps_per_branch']} "
            f"checkpoint={checkpoint_hash[:16]} "
            f"smoke={state} output=ABSENT"
        )

    print("PREFLIGHT: PASS")
    return rows, runner_hash, record


def coefficients(branch, geo, topo):
    return {
        "control": (0.0, 0.0),
        "geo": (geo, 0.0),
        "topo": (0.0, topo),
        "full": (geo, topo),
    }[branch]


def audit_result(pair, row, runner_hash, checkpoint_hash):
    seed = int(row["seed"])
    steps = int(row["steps_per_branch"])
    _, _, output = paths(pair, seed)

    result_path = output / "results.json"
    if not result_path.is_file():
        raise RuntimeError(f"Missing results: {result_path}")

    data = json.loads(result_path.read_text())

    required = {
        "status": "PASS",
        "mode": "run",
        "pair": pair,
        "seed": seed,
        "steps": steps,
        "validation_limit": None,
        "device": "cuda",
        "code_sha256": runner_hash,
        "checkpoint_sha256": checkpoint_hash,
    }

    for key, value in required.items():
        if data.get(key) != value:
            raise RuntimeError(f"Result mismatch: seed={seed} key={key}")

    if set(data.get("results", {})) != set(BRANCHES):
        raise RuntimeError(f"Branch set mismatch: seed {seed}")

    control = data["results"]["control"]["validation"]["Top-1"]

    for branch in BRANCHES:
        result = data["results"][branch]

        expected = coefficients(
            branch,
            float(row["lambda_geo"]),
            float(row["lambda_topo"]),
        )

        actual = (result["lambda_geo"], result["lambda_topo"])

        if actual != expected:
            raise RuntimeError(f"Coefficient mismatch: {seed} {branch}")

        if (
            result["optimizer_steps"] != steps
            or len(result["losses"]) != steps
        ):
            raise RuntimeError(f"Incomplete training: {seed} {branch}")

        for loss in result["losses"]:
            for key in ("alignment", "geometry", "topology", "total"):
                if not math.isfinite(loss[key]):
                    raise RuntimeError(f"Nonfinite loss: {seed} {branch}")

        for value in result["validation"].values():
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise RuntimeError(f"Invalid metric: {seed} {branch}")

        if branch != "control":
            delta = result["validation"]["Top-1"] - control
            saved = data["paired_top1_deltas"][branch]

            if not math.isclose(delta, saved, abs_tol=1e-12):
                raise RuntimeError(f"Paired delta mismatch: {seed} {branch}")

    return sha(result_path)


def execute(pair, mode):
    LOGS.mkdir(parents=True, exist_ok=True)

    with (LOGS / f"{pair}.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

        rows, runner_hash, record = inspect(pair, mode)

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        batch_dir = LOGS / f"{pair}_{mode}_{stamp}"
        batch_dir.mkdir(exist_ok=False)

        manifest = {
            "pair": pair,
            "mode": mode,
            "runner_sha256": runner_hash,
            "protocol_sha256": record["protocol_sha256"],
            "matrix_sha256": record["execution_matrix_sha256"],
            "status": "RUNNING",
            "runs": [],
        }

        manifest_path = batch_dir / "manifest.json"

        def save():
            temp = batch_dir / "manifest.json.tmp"
            temp.write_text(json.dumps(manifest, indent=2))
            temp.replace(manifest_path)

        save()

        try:
            for row in rows:
                seed = int(row["seed"])
                checkpoint, gate, output = paths(pair, seed)
                checkpoint_hash = sha(checkpoint)

                if mode == "smoke" and gate.is_file():
                    print(f"seed {seed}: existing smoke gate PASS")
                    continue

                if output.exists() or output.is_symlink():
                    raise RuntimeError(f"Output appeared: {output}")

                command = [
                    sys.executable,
                    str(RUNNER),
                    "--mode", "smoke-test" if mode == "smoke" else "run",
                    "--pair", pair,
                    "--seed", str(seed),
                    "--device", "cuda",
                ]

                if mode == "smoke":
                    command.extend(["--steps", "2"])
                else:
                    command.append("--approve-full")

                entry = {
                    "seed": seed,
                    "status": "RUNNING",
                    "command": command,
                }
                manifest["runs"].append(entry)
                save()

                env = os.environ.copy()
                env["PYTHONPATH"] = str(ROOT)

                log_path = batch_dir / f"seed{seed}.log"

                print(f"\n=== {pair} seed {seed}: {mode} ===", flush=True)

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

                if returncode:
                    raise RuntimeError(
                        f"Seed {seed} exited {returncode}; see {log_path}"
                    )

                if mode == "smoke":
                    check_gate(pair, seed, checkpoint_hash, runner_hash)
                    entry["gate_sha256"] = sha(gate)
                else:
                    entry["results_sha256"] = audit_result(
                        pair, row, runner_hash, checkpoint_hash
                    )

                entry["status"] = "PASS"
                save()
                print(f"=== {pair} seed {seed}: PASS ===", flush=True)

            if sha(RUNNER) != runner_hash:
                raise RuntimeError("Runner changed during batch")

            current = json.loads((REG / "registration.json").read_text())
            for filename, key in (
                ("analysis_protocol.md", "protocol_sha256"),
                ("execution_matrix.csv", "execution_matrix_sha256"),
            ):
                if sha(REG / filename) != current[key] or current[key] != record[key]:
                    raise RuntimeError("Registered protocol changed during batch")

            manifest["status"] = "PASS"
            save()
            print(f"\nBATCH PASS: {manifest_path}")

        except BaseException as exc:
            manifest["status"] = "FAILED"
            manifest["error"] = str(exc)
            if manifest["runs"] and manifest["runs"][-1]["status"] == "RUNNING":
                manifest["runs"][-1]["status"] = "FAILED"
            save()
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", choices=PAIRS, required=True)
    parser.add_argument(
        "--mode", choices=("preflight", "smoke", "run"),
        default="preflight",
    )
    parser.add_argument("--approve-full", action="store_true")
    args = parser.parse_args()

    if args.mode == "run" and not args.approve_full:
        parser.error("Full execution requires --approve-full")

    if args.mode != "run" and args.approve_full:
        parser.error("--approve-full is valid only for run")

    if args.mode == "preflight":
        inspect(args.pair, args.mode)
        print("DRY RUN ONLY: NO TRAINING STARTED")
    else:
        execute(args.pair, args.mode)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"E1-B BLOCKED/FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
