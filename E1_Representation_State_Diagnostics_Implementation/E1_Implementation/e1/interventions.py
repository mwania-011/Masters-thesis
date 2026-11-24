"""Matched-intervention bookkeeping and branch-state verification.

Training is deliberately not implemented without the original E0 model and optimizer APIs.
"""
import hashlib
import json
from pathlib import Path
import torch

VARIANTS={'control':(0.,0.),'geo':(0.0005,0.),'topo':(0.,0.0005),'full':(0.0005,0.0005)}

def state_digest(state):
    h=hashlib.sha256()
    def visit(obj):
        if isinstance(obj,torch.Tensor):
            v=obj.detach().cpu().contiguous()
            h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode());h.update(v.numpy().tobytes())
        elif isinstance(obj,dict):
            for key in sorted(obj,key=str):
                h.update(str(key).encode());visit(obj[key])
        elif isinstance(obj,(list,tuple)):
            for item in obj: visit(item)
        elif isinstance(obj,(str,int,float,bool,type(None))):
            h.update(repr(obj).encode())
        else:
            raise TypeError(f'Unsupported checkpoint state type {type(obj)}')
    visit(state)
    return h.hexdigest()

def verify_equal_initial_states(states):
    if len(states)<2: raise ValueError('Need >=2 matched branches')
    digests=[state_digest(s) for s in states]
    if len(set(digests))!=1: raise ValueError('Matched branch initial states differ')
    return digests[0]

def paired_deltas(control_score,branch_scores):
    if 'control' in branch_scores: raise ValueError('Pass non-control branches only')
    return {name:float(score-control_score) for name,score in branch_scores.items()}

def safe_output_dir(path):
    p=Path(path).resolve()
    if 'e0' in [part.lower() for part in p.parts]:
        raise ValueError('Refusing to write into E0')
    p.mkdir(parents=True,exist_ok=False)
    return p

def create_plan(language_pairs=('en-fr','en-ru','en-it','en-es'),seeds=range(1,6),
                variants=('control','geo','topo','full')):
    rows=[]
    for pair in language_pairs:
        for seed in seeds:
            for variant in variants:
                rows.append({'experiment_id':f'e1_{pair}_s{seed}_{variant}',
                             'language_pair':pair,'seed':seed,'variant':variant,
                             'lambda_geo':VARIANTS[variant][0],
                             'lambda_topo':VARIANTS[variant][1],
                             'source_checkpoint':'UNRESOLVED: requires E0 repository inspection',
                             'start_epoch':'UNRESOLVED', 'continuation_epochs':'TO_BE_APPROVED',
                             'matched_control':f'e1_{pair}_s{seed}_control',
                             'status':'BLOCKED_PENDING_CHECKPOINT_AUDIT'})
    return rows
