"""E2: preregistered heuristic reliability gates, one-epoch E0 continuation.

No validation or test examples are used for gate decisions. The gates are
computed from detached representations of the CURRENT TRAINING BATCH.
This is an exploratory controller, not a validated predictor of benefit.
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
from thesis.experiments.e0_config import BATCH_SIZE

ROOT = Path('thesis/results/e2/reliability_gated_v1')
CONTROLLER_VERSION = 'e2_heuristic_v1'
CONFIG = dict(ema_decay=0.9, warmup_steps=20, k=5,
              geometry_low=0.05, geometry_high=0.20,
              overlap_low=0.15, overlap_high=0.45,
              maturity_low=0.00, maturity_high=0.30)

def ramp(value, low, high):
    if not high > low: raise ValueError('Invalid ramp interval')
    return max(0., min(1., (float(value)-low)/(high-low)))

def diagnostics(a, b, k=5):
    """Detached batch-only diagnostics, bounded and reproducible."""
    with torch.no_grad():
        a = torch.nn.functional.normalize(a.detach().float(), dim=-1)
        b = torch.nn.functional.normalize(b.detach().float(), dim=-1)
        n = a.shape[0]
        if n < k+2: raise ValueError('Insufficient batch for neighborhood diagnostic')
        positive = (a*b).sum(-1).mean()
        sim = a @ b.T
        sim.fill_diagonal_(-float('inf'))
        hard = sim.max(dim=1).values.mean()
        margin = float((positive-hard).item())
        concentration = float(((a.mean(0).square().sum() + b.mean(0).square().sum())/2).item())
        aa = a @ a.T
        bb = b @ b.T
        aa.fill_diagonal_(-float('inf'))
        bb.fill_diagonal_(-float('inf'))
        ia = aa.topk(k, dim=1).indices
        ib = bb.topk(k, dim=1).indices
        overlap = float((ia.unsqueeze(2)==ib.unsqueeze(1)).any(dim=2).float().mean().item())
        return {'alignment_margin':margin,'concentration':concentration,'neighborhood_overlap':overlap}

def weights(d, step, pair):
    """Alignment maturity is a common prerequisite; independent structure gates."""
    c=CONFIG
    maturity=ramp(d['alignment_margin'],c['maturity_low'],c['maturity_high'])
    geometry_need=ramp(d['concentration'],c['geometry_low'],c['geometry_high'])
    neighborhood_reliability=ramp(d['neighborhood_overlap'],c['overlap_low'],c['overlap_high'])
    warmup=min(1.,step/c['warmup_steps'])
    geo_gate=maturity*geometry_need*warmup
    topo_gate=maturity*neighborhood_reliability*warmup
    base_geo,base_topo=e1b.coefficients(pair,'full')
    return base_geo*geo_gate,base_topo*topo_gate,geo_gate,topo_gate

def train_gated(ck,pair,schedule,train_data,val_data,device,validation_limit):
    model,opt=e1b.construct_model_optimizer(ck,device)
    e1b.restore_rng(ck)
    model.train()
    trace=[]
    ema=None
    for step,indices in enumerate(schedule,1):
        x1,x2=e1b.collate_indices(train_data,indices,device)
        a=model(**x1); b=model(**x2)
        raw=diagnostics(a,b,k=CONFIG['k'])
        ema=raw if ema is None else {key: CONFIG['ema_decay']*ema[key]+(1-CONFIG['ema_decay'])*raw[key] for key in raw}
        lg,lt,gg,tg=weights(ema,step,pair)
        la=journal.alignment_loss(a,b)
        geo=journal.geometry_loss(a,b) if lg>0 else None
        topo=journal.topology_loss(a,b) if lt>0 else None
        total=la+(lg*geo if geo is not None else 0)+(lt*topo if topo is not None else 0)
        if not bool(torch.isfinite(total).item()): raise FloatingPointError(f'Nonfinite loss at {step}')
        opt.zero_grad();total.backward()
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        if not bool(torch.isfinite(grad).item()): raise FloatingPointError(f'Nonfinite gradient at {step}')
        opt.step()
        row={'step':step,**{'raw_'+k:v for k,v in raw.items()},
             **{'ema_'+k:v for k,v in ema.items()},'lambda_geo':lg,'lambda_topo':lt,
             'geo_gate':gg,'topo_gate':tg,'alignment':float(la.detach()),
             'geometry':float(geo.detach()) if geo is not None else 0.,
             'topology':float(topo.detach()) if topo is not None else 0.,
             'total':float(total.detach()),'grad_norm':float(grad.detach())}
        if not all(math.isfinite(v) for v in row.values()): raise FloatingPointError(f'Nonfinite trace at {step}')
        trace.append(row)
    result={'optimizer_steps':len(trace),'model_sha256':e1b.canonical_hash(model.state_dict()),
            'optimizer_sha256':e1b.canonical_hash(opt.state_dict()),
            'validation':e1b.evaluate(model,val_data,device,validation_limit),'trace':trace}
    del model,opt
    if device.type=='cuda': torch.cuda.empty_cache()
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',required=True,choices=('preflight','smoke-test','run'))
    p.add_argument('--pair',required=True,choices=e1b.PAIRS)
    p.add_argument('--seed',required=True,type=int,choices=range(1,6))
    p.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    p.add_argument('--approve-full',action='store_true')
    args=p.parse_args()
    if args.mode=='run' and not args.approve_full: p.error('Full run requires --approve-full')
    if args.approve_full and args.mode!='run': p.error('--approve-full only for full run')
    if args.device=='cuda' and not torch.cuda.is_available(): p.error('CUDA unavailable')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    device=torch.device('cuda:0' if args.device=='cuda' else 'cpu')
    journal.DEVICE=device
    ck,ck_hash=e1b.load_verified_checkpoint(args.pair,args.seed)
    train_data=e1b.dataset_for(args.pair,'train')
    steps=2 if args.mode!='run' else math.ceil(len(train_data)/BATCH_SIZE)
    ck,ck_hash,train_data,schedule,states=e1b.preflight(args.pair,args.seed,device,steps)
    sh=e1b.canonical_hash(schedule)
    code_sha=e1b.sha256(__file__)
    provenance={'controller_version':CONTROLLER_VERSION,'config':CONFIG,'mode':args.mode,
        'pair':args.pair,'seed':args.seed,'device':args.device,'steps':steps,
        'checkpoint_sha256':ck_hash,'schedule_sha256':sh,'code_sha256':code_sha,
        'e1b_runner_sha256':e1b.sha256(e1b.__file__),'initial_states':states}
    print(json.dumps({'status':'PREFLIGHT_PASS',**provenance},indent=2),flush=True)
    if args.mode=='preflight': return
    output=ROOT/('smoke' if args.mode=='smoke-test' else 'full')/f'{args.pair}_seed{args.seed}'
    if output.exists(): raise FileExistsError(f'Refusing overwrite: {output}')
    if args.mode=='run':
        gatepath=ROOT/'smoke'/f'{args.pair}_seed{args.seed}'/'gate.json'
        if not gatepath.is_file(): p.error(f'Missing smoke gate: {gatepath}')
        gate=json.loads(gatepath.read_text())
        for key in ('checkpoint_sha256','code_sha256','e1b_runner_sha256','device'):
            if gate.get(key)!=provenance[key]: p.error(f'Smoke gate mismatch: {key}')
        if gate.get('status')!='PASS' or gate.get('replay_passed') is not True: p.error('Smoke replay gate failed')
    output.mkdir(parents=True)
    try:
        val=e1b.dataset_for(args.pair,'validation')
        limit=128 if args.mode=='smoke-test' else None
        result=train_gated(ck,args.pair,schedule,train_data,val,device,limit)
        if args.mode=='smoke-test':
            again=train_gated(ck,args.pair,schedule,train_data,val,device,limit)
            if e1b.canonical_hash(result)!=e1b.canonical_hash(again):
                raise RuntimeError('Deterministic smoke replay mismatch')
        record={'status':'PASS',**provenance,'validation_limit':limit,
                'replay_passed':args.mode=='smoke-test','result':result}
        (output/'results.json').write_text(json.dumps(record,indent=2)+'\n')
        if args.mode=='smoke-test':
            (output/'gate.json').write_text(json.dumps({k:v for k,v in record.items() if k not in ('result','initial_states')},indent=2)+'\n')
        print(f'E2 {args.mode} PASS: {output} | Top-1={result["validation"]["Top-1"]:.6f}',flush=True)
    except Exception as exc:
        (output/'FAILED.txt').write_text(f'{type(exc).__name__}: {exc}\n')
        raise

if __name__=='__main__': main()
