"""E1-B matched, fixed-coefficient continuation of frozen E0 baseline checkpoints.

Run from repository root. Never writes to thesis/results/e0.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import XLMRobertaTokenizer

import train_structure_preserving_multilingual as journal
from thesis.data.dataset import ThesisParallelDataset
from thesis.evaluation.retrieval import compute_retrieval_stats_chunked
from thesis.experiments.e0_config import BATCH_SIZE, LEARNING_RATE, MAX_LENGTH, TOPOLOGY_WEIGHTS

PAIRS = ('en-fr','en-ru','en-it','en-es')
BRANCHES = ('control','geo','topo','full')
ROOT = Path('thesis/results/e1/b_interventions')
GEO_WEIGHT = 0.0005

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def canonical_hash(value):
    """Digest tensors and primitive nested checkpoint structures without pickle."""
    h=hashlib.sha256()
    def walk(v):
        if isinstance(v,torch.Tensor):
            a=v.detach().cpu().contiguous()
            h.update(b'T');h.update(str(a.dtype).encode());h.update(str(tuple(a.shape)).encode())
            h.update(a.numpy().tobytes())
        elif isinstance(v,dict):
            h.update(b'D')
            for k in sorted(v,key=str): walk(str(k));walk(v[k])
        elif isinstance(v,(tuple,list)):
            h.update(b'L');h.update(str(len(v)).encode())
            for item in v: walk(item)
        elif isinstance(v,np.ndarray):
            h.update(b'A');h.update(str(v.dtype).encode());h.update(str(v.shape).encode());h.update(v.tobytes())
        elif isinstance(v,np.generic): walk(v.item())
        elif isinstance(v,(int,float,str,bool,type(None))):
            h.update((type(v).__name__+':'+repr(v)).encode())
        else: raise TypeError(f'Unsupported state type: {type(v)}')
    walk(value)
    return h.hexdigest()

def coefficients(pair,branch):
    if pair not in PAIRS or branch not in BRANCHES: raise ValueError('Unknown pair/branch')
    return (GEO_WEIGHT if branch in ('geo','full') else 0.,
            TOPOLOGY_WEIGHTS[pair] if branch in ('topo','full') else 0.)

def validate_output(path):
    p=Path(path).resolve(); base=ROOT.resolve()
    if p == base or base not in p.parents: raise ValueError('Output must be a child of E1-B results root')
    if p.exists(): raise FileExistsError(f'Refusing overwrite: {p}')
    return p

def source_checkpoint(pair,seed):
    return Path('thesis/results/e0')/f'{pair}_seed{seed}_baseline'/'last.pt'

def load_verified_checkpoint(pair,seed):
    p=source_checkpoint(pair,seed)
    if not p.is_file(): raise FileNotFoundError(p)
    ck=torch.load(p,map_location='cpu',weights_only=False) # trusted local E0 checkpoint only
    assert ck['epoch']==10 and ck['run']['run_id']==f'{pair}_seed{seed}_baseline'
    assert ck['run']['variant']=='baseline'
    assert ck['run']['batch_size']==BATCH_SIZE and ck['run']['max_length']==MAX_LENGTH
    assert ck['run']['learning_rate']==LEARNING_RATE
    assert not ck['run']['experiment']['use_geo'] and not ck['run']['experiment']['use_topo']
    for key in ('model','optimizer','rng_python','rng_numpy','rng_torch','rng_cuda','loader_generator'):
        if key not in ck or ck[key] is None: raise RuntimeError(f'Missing checkpoint state: {key}')
    for key,filename in [('journal_sha256','train_structure_preserving_multilingual.py'),
                         ('evaluator_sha256','thesis/evaluation/retrieval.py'),
                         ('runner_sha256','thesis/experiments/run_e0.py')]:
        actual_hash = sha256(filename)
        expected_hash = ck['metadata'][key]

        if actual_hash != expected_hash:
            if key != 'runner_sha256':
                raise RuntimeError(
                    f'Frozen E0 source mismatch: {filename}'
                )

            import subprocess

            revision = ck['metadata'].get('git_revision')
            if not revision:
                raise RuntimeError(
                    'Missing historical E0 Git revision'
                )

            try:
                historical_source = subprocess.check_output(
                    ['git', 'show', f'{revision}:{filename}'],
                    stderr=subprocess.PIPE,
                )
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(
                    'Cannot recover historical E0 runner'
                ) from exc

            historical_hash = hashlib.sha256(
                historical_source
            ).hexdigest()

            if historical_hash != expected_hash:
                raise RuntimeError(
                    'Historical E0 runner hash mismatch'
                )

            print(
                'PROVENANCE VERIFIED: historical E0 runner '
                f'{revision}; current runner SHA256={actual_hash}',
                flush=True,
            )
    manifest=Path('thesis/data/processed')/pair/'manifest.json'
    if sha256(manifest)!=ck['metadata']['manifest_sha256']:
        raise RuntimeError('Frozen dataset manifest mismatch')
    return ck,sha256(p)

def schedule_from_generator(n,batch_size,generator_state,steps):
    """Reproduce E0 DataLoader shuffle, including iterator base-seed draw."""
    gen=torch.Generator();gen.set_state(generator_state.clone())
    loader=DataLoader(range(n),batch_size=batch_size,shuffle=True,generator=gen,num_workers=0)
    batches=[]
    for batch in loader:
        batches.append(batch.tolist())
        if len(batches)>=steps: break
    if len(batches)!=steps: raise ValueError('Requested steps exceed one training epoch')
    return batches

def restore_rng(ck):
    random.setstate(ck['rng_python']);np.random.set_state(ck['rng_numpy'])
    torch.set_rng_state(ck['rng_torch'])
    torch.cuda.set_rng_state_all(ck['rng_cuda'])

def construct_model_optimizer(ck,device):
    model=journal.GeoTransformer().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=LEARNING_RATE)
    import copy
    model.load_state_dict(copy.deepcopy(ck['model']),strict=True)
    optimizer.load_state_dict(copy.deepcopy(ck['optimizer']))
    if len(optimizer.state_dict()['state']) != len(ck['optimizer']['state']):
        raise RuntimeError('Optimizer state count mismatch')
    return model,optimizer

def check_initial_states(ck,device):
    expected_model=canonical_hash(ck['model']);expected_opt=canonical_hash(ck['optimizer'])
    checks=[]
    for branch in BRANCHES:
        model,opt=construct_model_optimizer(ck,device)
        mh=canonical_hash(model.state_dict());oh=canonical_hash(opt.state_dict())
        if (mh,oh)!=(expected_model,expected_opt): raise RuntimeError(f'Initial-state mismatch: {branch}')
        checks.append({'branch':branch,'model_sha256':mh,'optimizer_sha256':oh})
        del model,opt
    return checks

def dataset_for(pair,split):
    tokenizer=XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    return ThesisParallelDataset(tokenizer=tokenizer,lang_pair=pair,split=split,max_length=MAX_LENGTH)

def collate_indices(dataset,indices,device):
    examples=[dataset[i] for i in indices]
    x1={k:torch.stack([a[k] for a,b in examples]).to(device) for k in examples[0][0]}
    x2={k:torch.stack([b[k] for a,b in examples]).to(device) for k in examples[0][1]}
    return x1,x2

def evaluate(model,validation,device,limit):
    # Validation is fixed, deterministic, and uses frozen E0 argsort retrieval semantics.
    from torch.utils.data import Subset
    if limit is not None: validation=Subset(validation,range(min(limit,len(validation))))
    loader=DataLoader(validation,batch_size=BATCH_SIZE,shuffle=False,num_workers=0)
    model.eval();left=[];right=[]
    with torch.inference_mode():
        for x1,x2 in loader:
            x1={k:v.to(device) for k,v in x1.items()};x2={k:v.to(device) for k,v in x2.items()}
            left.append(model(**x1).cpu());right.append(model(**x2).cpu())
    return compute_retrieval_stats_chunked(torch.cat(left),torch.cat(right),chunk_size=256)

def train_branch(ck,branch,pair,schedule,train_data,val_data,device,validation_limit):
    model,optimizer=construct_model_optimizer(ck,device)
    restore_rng(ck)  # after model construction, before stochastic forward pass
    model.train()
    lg,lt=coefficients(pair,branch)
    losses=[]
    for step,indices in enumerate(schedule,1):
        x1,x2=collate_indices(train_data,indices,device)
        z1=model(**x1);z2=model(**x2)
        la=journal.alignment_loss(z1,z2)
        geo=journal.geometry_loss(z1,z2) if lg else None
        topo=journal.topology_loss(z1,z2) if lt else None
        total=la+(lg*geo if geo is not None else 0)+(lt*topo if topo is not None else 0)
        if not bool(torch.isfinite(total).item()): raise FloatingPointError(f'{branch} nonfinite loss at step {step}')
        if not total.requires_grad: raise RuntimeError(f'{branch} missing gradient at step {step}')
        optimizer.zero_grad();total.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0,error_if_nonfinite=True)
        if not bool(torch.isfinite(norm).item()): raise FloatingPointError(f'{branch} nonfinite gradients at step {step}')
        optimizer.step()
        losses.append({'step':step,'total':float(total.detach()),'alignment':float(la.detach()),
                       'geometry':float(geo.detach()) if geo is not None else 0.,
                       'topology':float(topo.detach()) if topo is not None else 0.,
                       'grad_norm':float(norm.detach())})
    result={'branch':branch,'lambda_geo':lg,'lambda_topo':lt,'optimizer_steps':len(losses),
            'model_sha256':canonical_hash(model.state_dict()),
            'optimizer_sha256':canonical_hash(optimizer.state_dict()),
            'validation':evaluate(model,val_data,device,validation_limit), 'losses':losses}
    del model,optimizer
    if device.type=='cuda': torch.cuda.empty_cache()
    return result

def replay_signature(result):
    return canonical_hash({k:v for k,v in result.items() if k!='losses'}) , canonical_hash(result['losses'])

def preflight(pair,seed,device,steps):
    ck,ckhash=load_verified_checkpoint(pair,seed)
    train_data=dataset_for(pair,'train')
    schedule=schedule_from_generator(len(train_data),BATCH_SIZE,ck['loader_generator'],steps)
    if len(set(i for batch in schedule for i in batch))!=sum(map(len,schedule)):
        raise RuntimeError('Repeated sample indices within epoch')
    states=check_initial_states(ck,device)
    return ck,ckhash,train_data,schedule,states

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('preflight','smoke-test','run'),required=True)
    parser.add_argument('--pair',choices=PAIRS,required=True)
    parser.add_argument('--seed',type=int,choices=range(1,6),required=True)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--steps',type=int,default=None)
    parser.add_argument('--validation-limit',type=int,default=None)
    parser.add_argument('--approve-full',action='store_true',help='Explicit authorization after smoke audit')
    args=parser.parse_args()
    if args.device=='cuda' and not torch.cuda.is_available(): parser.error('CUDA unavailable')
    if args.mode=='run' and not args.approve_full: parser.error('Full continuation locked: --approve-full required')
    if args.mode!='run' and args.approve_full: parser.error('--approve-full is only valid with --mode run')
    if args.steps is None: args.steps=2 if args.mode!='run' else 0
    if args.steps<1 and args.mode!='run': parser.error('--steps must be positive')
    if args.validation_limit is None: args.validation_limit=128 if args.mode=='smoke-test' else None
    if args.mode=='run' and args.validation_limit is not None: parser.error('Full run must use full validation')
    if args.validation_limit is not None and args.validation_limit<2: parser.error('Validation limit must be >=2')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    device=torch.device('cuda:0' if args.device=='cuda' else 'cpu');journal.DEVICE=device
    ck,ckhash=load_verified_checkpoint(args.pair,args.seed)
    if args.mode=='run' and args.steps==0:
        train_data=dataset_for(args.pair,'train')
        args.steps=math.ceil(len(train_data)/BATCH_SIZE)
    if args.mode=='run':
        gate=ROOT/'smoke'/f'{args.pair}_seed{args.seed}'/'smoke_gate.json'
        if not gate.is_file(): parser.error(f'No passed group smoke gate: {gate}')
        data=json.loads(gate.read_text())
        if data.get('status')!='PASS' or data.get('checkpoint_sha256')!=ckhash:
            parser.error('Smoke gate failed or checkpoint changed')
        if data.get('device')!=args.device: parser.error('Smoke gate device mismatch')
        if data.get('code_sha256')!=sha256(__file__): parser.error('Runner changed since smoke gate; rerun smoke test')
        if data.get('replay_passed') is not True: parser.error('Deterministic replay not verified')
    # Preflight itself writes nothing.
    ck,ckhash,train_data,schedule,states=preflight(args.pair,args.seed,device,args.steps)
    schedule_hash=canonical_hash(schedule)
    print(json.dumps({'phase':'E1-B','mode':args.mode,'run':f'{args.pair}_seed{args.seed}',
                      'checkpoint_sha256':ckhash,'initial_states':states,
                      'schedule_sha256':schedule_hash,'steps':len(schedule),
                      'branch_coefficients':{b:coefficients(args.pair,b) for b in BRANCHES},
                      'status':'PREFLIGHT_PASS'},indent=2),flush=True)
    if args.mode=='preflight': return
    output=validate_output(ROOT/('smoke' if args.mode=='smoke-test' else 'full')/f'{args.pair}_seed{args.seed}')
    # No output directory is created before preflight passes.
    output.mkdir(parents=True)
    try:
        validation=dataset_for(args.pair,'validation')
        all_results={}
        for branch in BRANCHES:
            first=train_branch(ck,branch,args.pair,schedule,train_data,validation,device,args.validation_limit)
            if first['optimizer_steps']!=len(schedule): raise RuntimeError('Step count mismatch')
            if args.mode=='smoke-test':
                second=train_branch(ck,branch,args.pair,schedule,train_data,validation,device,args.validation_limit)
                if replay_signature(first)!=replay_signature(second):
                    raise RuntimeError(f'Deterministic replay mismatch: {branch}')
            all_results[branch]=first
            print(f'PASS {branch}: {len(schedule)} steps, Top-1={first["validation"]["Top-1"]:.6f}',flush=True)
        manifest={'status':'PASS','mode':args.mode,'pair':args.pair,'seed':args.seed,
                  'device':args.device,'checkpoint_sha256':ckhash,'schedule_sha256':schedule_hash,
                  'steps':len(schedule),'validation_limit':args.validation_limit,
                  'code_sha256':sha256(__file__),
                  'replay_passed':args.mode=='smoke-test',
                  'results':all_results,
                  'paired_top1_deltas':{b:all_results[b]['validation']['Top-1']-all_results['control']['validation']['Top-1']
                                        for b in BRANCHES if b!='control'}}
        (output/'results.json').write_text(json.dumps(manifest,indent=2)+'\n')
        if args.mode=='smoke-test':
            gate={k:manifest[k] for k in ('status','mode','pair','seed','device','checkpoint_sha256','schedule_sha256','steps','code_sha256','replay_passed')}
            (output/'smoke_gate.json').write_text(json.dumps(gate,indent=2)+'\n')
        print(f'=== E1-B {args.mode.upper()} PASS ===\nOutput: {output}',flush=True)
    except Exception as exc:
        (output/'FAILED.txt').write_text(type(exc).__name__+': '+str(exc)+'\n')
        raise

if __name__=='__main__': main()
