"""E3-v2: controlled one-epoch structural-regularization ablations.

Exploratory thresholds are fixed, not calibrated. Never tune on evaluation results.
Run as: python -m thesis.experiments.run_e3_ablations ...
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
import math
from pathlib import Path
import torch
import train_structure_preserving_multilingual as journal
from thesis.experiments import run_e1b as e1b
from thesis.experiments import run_e2_gated as e2
from thesis.experiments.e0_config import BATCH_SIZE

VERSION = 'e3_ablation_v2'
ROOT = Path('thesis/results/e3/ablations_v2')
BRANCHES = ('continuous', 'geometry_only', 'neighborhood_only',
            'selective_neighborhood', 'selective_geometry', 'joint_selective',
            'alignment_only')
CONFIG = dict(e2.CONFIG, threshold_geometry=0.54, threshold_neighborhood=0.48,
              threshold_basis='EMA diagnostics', calibration_status='exploratory_not_independently_calibrated')


def weights(d, step, pair, branch):
    if branch not in BRANCHES:
        raise ValueError(branch)
    c = CONFIG
    maturity = e2.ramp(d['alignment_margin'], c['maturity_low'], c['maturity_high'])
    geo_factor = e2.ramp(d['concentration'], c['geometry_low'], c['geometry_high'])
    topo_factor = e2.ramp(d['neighborhood_overlap'], c['overlap_low'], c['overlap_high'])
    warmup = min(1., step / c['warmup_steps'])
    geo_on = branch in ('continuous', 'geometry_only', 'selective_neighborhood') or (
        branch in ('selective_geometry', 'joint_selective') and
        d['concentration'] >= c['threshold_geometry'])
    topo_on = branch in ('continuous', 'neighborhood_only', 'selective_geometry') or (
        branch in ('selective_neighborhood', 'joint_selective') and
        d['neighborhood_overlap'] >= c['threshold_neighborhood'])
    gg = maturity * geo_factor * warmup * float(geo_on)
    tg = maturity * topo_factor * warmup * float(topo_on)
    lg, lt = e1b.coefficients(pair, 'full')
    return lg * gg, lt * tg, gg, tg


def train(ck, pair, branch, schedule, train_data, val_data, device, validation_limit):
    model, opt = e1b.construct_model_optimizer(ck, device)
    e1b.restore_rng(ck)
    model.train()
    trace = []
    ema = None
    for step, indices in enumerate(schedule, 1):
        x1, x2 = e1b.collate_indices(train_data, indices, device)
        a, b = model(**x1), model(**x2)
        valid = a.shape[0] >= CONFIG['k'] + 2
        if valid:
            raw = e2.diagnostics(a, b, k=CONFIG['k'])
            ema = raw if ema is None else {
                key: CONFIG['ema_decay'] * ema[key] + (1-CONFIG['ema_decay']) * raw[key]
                for key in raw
            }
            lg, lt, gg, tg = weights(ema, step, pair, branch)
        else:
            raw = None
            lg = lt = gg = tg = 0.
        la = journal.alignment_loss(a, b)
        geo = journal.geometry_loss(a, b) if lg > 0 else None
        topo = journal.topology_loss(a, b) if lt > 0 else None
        total = la + (lg*geo if geo is not None else 0) + (lt*topo if topo is not None else 0)
        if not bool(torch.isfinite(total).item()):
            raise FloatingPointError(f'Nonfinite loss at {step}')
        opt.zero_grad()
        total.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        if not bool(torch.isfinite(norm).item()):
            raise FloatingPointError(f'Nonfinite gradient at {step}')
        opt.step()
        row = {'step':step, 'batch_size':int(a.shape[0]), 'diagnostic_valid':valid,
               'structural_regularization_skipped':not valid,
               **{'raw_'+k:raw[k] if raw is not None else None for k in ('alignment_margin','concentration','neighborhood_overlap')},
               **{'ema_'+k:ema[k] if ema is not None else None for k in ('alignment_margin','concentration','neighborhood_overlap')},
               'lambda_geo':lg, 'lambda_topo':lt, 'geo_gate':gg, 'topo_gate':tg,
               'alignment':float(la.detach()), 'geometry':float(geo.detach()) if geo is not None else 0.,
               'topology':float(topo.detach()) if topo is not None else 0.,
               'total':float(total.detach()), 'grad_norm':float(norm.detach())}
        if not all(math.isfinite(v) for v in row.values() if isinstance(v,(int,float)) and not isinstance(v,bool)):
            raise FloatingPointError(f'Nonfinite trace at {step}')
        trace.append(row)
    result = {'optimizer_steps':len(trace), 'model_sha256':e1b.canonical_hash(model.state_dict()),
              'optimizer_sha256':e1b.canonical_hash(opt.state_dict()),
              'validation':e1b.evaluate(model, val_data, device, validation_limit), 'trace':trace}
    del model, opt
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', required=True, choices=('preflight','smoke-test','run'))
    p.add_argument('--branch', required=True, choices=BRANCHES)
    p.add_argument('--pair', required=True, choices=e1b.PAIRS)
    p.add_argument('--seed', required=True, type=int, choices=range(1,6))
    p.add_argument('--device', default='cuda', choices=('cuda','cpu'))
    p.add_argument('--approve-full', action='store_true')
    args = p.parse_args()
    if args.mode == 'run' and not args.approve_full:
        p.error('Full run requires --approve-full')
    if args.mode != 'run' and args.approve_full:
        p.error('--approve-full is only valid with --mode run')
    if args.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA unavailable')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda:0' if args.device == 'cuda' else 'cpu')
    journal.DEVICE = device
    ck, ck_hash = e1b.load_verified_checkpoint(args.pair, args.seed)
    data = e1b.dataset_for(args.pair, 'train')
    steps = 2 if args.mode != 'run' else math.ceil(len(data)/BATCH_SIZE)
    ck, ck_hash, data, schedule, states = e1b.preflight(args.pair, args.seed, device, steps)
    provenance = {'controller_version':VERSION, 'branch':args.branch, 'config':CONFIG,
                  'pair':args.pair, 'seed':args.seed, 'device':args.device, 'steps':steps,
                  'checkpoint_sha256':ck_hash, 'schedule_sha256':e1b.canonical_hash(schedule),
                  'code_sha256':e1b.sha256(__file__), 'e1b_runner_sha256':e1b.sha256(e1b.__file__),
                  'e2_runner_sha256':e1b.sha256(e2.__file__), 'initial_states':states}
    if args.mode == 'preflight':
        print(json.dumps({'status':'PREFLIGHT_PASS', **provenance}, indent=2), flush=True)
        return
    run_id = f'{args.pair}_seed{args.seed}'
    output = ROOT / ('smoke' if args.mode == 'smoke-test' else 'full') / args.branch / run_id
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite: {output}')
    if args.mode == 'run':
        gate_path = ROOT / 'smoke' / args.branch / run_id / 'gate.json'
        if not gate_path.is_file():
            p.error(f'Missing smoke-test gate: {gate_path}')
        gate = json.loads(gate_path.read_text())
        for key in ('checkpoint_sha256','code_sha256','e1b_runner_sha256','e2_runner_sha256','device','branch','config'):
            if gate.get(key) != provenance[key]:
                p.error(f'Smoke-test gate mismatch: {key}')
        if gate.get('status') != 'PASS' or gate.get('replay_passed') is not True:
            p.error('Smoke-test replay was not verified')
    output.mkdir(parents=True)
    try:
        val = e1b.dataset_for(args.pair, 'validation')
        limit = 128 if args.mode == 'smoke-test' else None
        result = train(ck, args.pair, args.branch, schedule, data, val, device, limit)
        if args.mode == 'smoke-test':
            again = train(ck, args.pair, args.branch, schedule, data, val, device, limit)
            if e1b.canonical_hash(result) != e1b.canonical_hash(again):
                raise RuntimeError('Deterministic smoke-test replay mismatch')
        record = {'status':'PASS', 'mode':args.mode, **provenance, 'validation_limit':limit,
                  'replay_passed':args.mode == 'smoke-test', 'result':result}
        (output / 'results.json').write_text(json.dumps(record, indent=2) + '\n')
        if args.mode == 'smoke-test':
            gate = {k:v for k,v in record.items() if k not in ('result','initial_states')}
            (output / 'gate.json').write_text(json.dumps(gate, indent=2) + '\n')
        print(f'{VERSION} {args.mode} PASS: {output} | Top-1={result["validation"]["Top-1"]:.6f}', flush=True)
    except Exception as exc:
        (output / 'FAILED.txt').write_text(f'{type(exc).__name__}: {exc}\n')
        raise

if __name__ == '__main__':
    main()
