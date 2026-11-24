import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np
from .diagnostics import evaluate
from .interventions import create_plan

def main():
    p=argparse.ArgumentParser(description='E1: offline diagnostics on paired VALIDATION embeddings')
    sub=p.add_subparsers(dest='command',required=True)
    d=sub.add_parser('diagnose');d.add_argument('--npz',required=True);d.add_argument('--output',required=True)
    d.add_argument('--k',type=int,default=10);d.add_argument('--seed',type=int,default=0)
    x=sub.add_parser('plan');x.add_argument('--output',required=True)
    args=p.parse_args()
    out=Path(args.output)
    if out.exists():p.error('Output already exists; refusing overwrite')
    if 'e0' in [s.lower() for s in out.resolve().parts]:p.error('E0 is frozen')
    out.parent.mkdir(parents=True,exist_ok=True)
    if args.command=='diagnose':
        inp=Path(args.npz)
        with np.load(inp,allow_pickle=False) as z:
            if set(z.files)!={'source','target'}:p.error('NPZ must contain exactly source,target arrays')
            result=evaluate(z['source'],z['target'],k=args.k,seed=args.seed)
        result['provenance']={'source_file':str(inp),'sha256':hashlib.sha256(inp.read_bytes()).hexdigest(),
                              'seed':args.seed,'split':'validation (must be verified by operator)'}
        with out.open('x') as f:json.dump(result,f,indent=2)
    else:
        rows=create_plan()
        with out.open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('Created',out)
if __name__=='__main__':main()
