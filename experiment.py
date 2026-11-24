"""Run matched-budget multilingual sentence retrieval experiments.
Input JSONL: {"source":"...","target":"..."}.
GPU requirements: pip install torch transformers pandas numpy scikit-learn matplotlib
"""
import argparse,csv,json,os,random,time,sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from rg_core import Controller,alignment_loss,structural_losses,diagnostics,retrieval_metrics

class Pairs(Dataset):
    def __init__(self,path):
        with open(path,encoding='utf8') as f:self.rows=[json.loads(x) for x in f if x.strip()]
        if not self.rows or any('source' not in r or 'target' not in r for r in self.rows):raise ValueError('Expected source and target keys')
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):return self.rows[i]

class Encoder(nn.Module):
    def __init__(self,base):
        super().__init__()
        from transformers import AutoModel
        self.base=AutoModel.from_pretrained(base)
        h=self.base.config.hidden_size
        self.proj=nn.Sequential(nn.Linear(h,h),nn.GELU(),nn.Linear(h,256),nn.LayerNorm(256))
    def forward(self,tokens):
        out=self.base(**tokens).last_hidden_state[:,0,:]
        return F.normalize(self.proj(out),dim=-1)

def evaluate(model,tokenizer,loader,device):
    model.eval();aa=[];bb=[]
    with torch.no_grad():
        for batch in loader:
            a=tokenizer(batch['source'],padding=True,truncation=True,max_length=64,return_tensors='pt').to(device)
            b=tokenizer(batch['target'],padding=True,truncation=True,max_length=64,return_tensors='pt').to(device)
            aa.append(model(a).cpu());bb.append(model(b).cpu())
    return retrieval_metrics(torch.cat(aa),torch.cat(bb))

def train(args):
    from transformers import AutoTokenizer
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed)
    device='cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    train_ds=Pairs(args.train);val_ds=Pairs(args.validation)
    if len(val_ds)<5:raise ValueError('Validation needs at least 5 parallel pairs')
    tokenizer=AutoTokenizer.from_pretrained(args.model)
    model=Encoder(args.model).to(device)
    if args.checkpoint:
        state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        weights = state['model'] if 'model' in state else state
        remapped = {}
        for key, value in weights.items():
            if key.startswith('encoder.'):
                key = 'base.' + key[len('encoder.'):]
            elif key.startswith('projection.'):
                key = 'proj.' + key[len('projection.'):]
            remapped[key] = value
        model.load_state_dict(remapped, strict=True)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr)
    loader=DataLoader(train_ds,batch_size=args.batch,shuffle=True,drop_last=True)
    val_loader=DataLoader(val_ds,batch_size=args.eval_batch,shuffle=False)
    controller=Controller(method=args.method,pair=args.pair,threshold_g=args.gate_g,threshold_n=args.gate_n,
      disable_margin=args.ablate=='margin',disable_geometry=args.ablate=='geometry',
      disable_neighborhood=args.ablate=='neighborhood',disable_ema=args.ablate=='ema',disable_selection=args.ablate=='selection')
    history=[];traces=[];step=0;start=time.perf_counter()
    for epoch in range(1,args.epochs+1):
        model.train();losses=[]
        for batch in loader:
            step+=1
            a=tokenizer(batch['source'],padding=True,truncation=True,max_length=64,return_tensors='pt').to(device)
            b=tokenizer(batch['target'],padding=True,truncation=True,max_length=64,return_tensors='pt').to(device)
            za,zb=model(a),model(b)
            raw=diagnostics(za,zb)
            wg,wn,meta=controller.weights(raw,step)
            align=alignment_loss(za,zb)
            geo,topo=structural_losses(za,zb)
            loss=align+wg*geo+wn*topo
            if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite loss at step {step}')
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            losses.append(loss.item())
            traces.append({'step':step,'epoch':epoch,'margin':raw['margin'],'concentration':raw['concentration'],
                           'overlap':raw['overlap'],'lambda_geometry':wg,'lambda_neighborhood':wn,**meta})
        metrics=evaluate(model,tokenizer,val_loader,device)
        record={'epoch':epoch,'seed':args.seed,'pair':args.pair,'method':args.method,'ablate':args.ablate or 'none',
                'loss':float(np.mean(losses)),'elapsed_seconds':time.perf_counter()-start,
                'peak_vram_gb':torch.cuda.max_memory_allocated()/1e9 if device=='cuda' else 0.,**metrics}
        history.append(record);print(json.dumps(record),flush=True)
        (out/'history.json').write_text(json.dumps(history,indent=2))
        (out/'traces.json').write_text(json.dumps(traces,indent=2))
        if args.save_checkpoint:torch.save({'model':model.state_dict(),'epoch':epoch,'optimizer':opt.state_dict()},out/'latest.pt')
    (out/'run_config.json').write_text(json.dumps(vars(args),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--train',required=True);p.add_argument('--validation',required=True);p.add_argument('--output',required=True)
    p.add_argument('--model',default='xlm-roberta-base');p.add_argument('--checkpoint',default='')
    p.add_argument('--pair',default='en-fr',choices=['en-fr','en-ru','en-it','en-es'])
    p.add_argument('--method',default='calibrated',choices=['alignment','fixed','original','calibrated','selective'])
    p.add_argument('--ablate',default='',choices=['','margin','geometry','neighborhood','ema','selection'])
    p.add_argument('--gate-g',type=float,default=.55);p.add_argument('--gate-n',type=float,default=.55)
    p.add_argument('--seed',type=int,default=1);p.add_argument('--epochs',type=int,default=1)
    p.add_argument('--batch',type=int,default=32);p.add_argument('--eval-batch',type=int,default=64)
    p.add_argument('--lr',type=float,default=2e-5);p.add_argument('--cpu',action='store_true');p.add_argument('--save-checkpoint',action='store_true')
    args=p.parse_args();train(args)
