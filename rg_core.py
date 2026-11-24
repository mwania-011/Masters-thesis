"""Thesis-aligned losses, diagnostics and calibrated selective controller."""
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F


def diagnostics(a, b, k=5):
    with torch.no_grad():
        a, b = F.normalize(a.detach(),dim=-1), F.normalize(b.detach(),dim=-1)
        n=a.shape[0]
        s=a@b.T
        margin=(s.diag()-s.masked_fill(torch.eye(n,device=a.device,dtype=torch.bool),-1e4).max(1).values).mean().item() if n>1 else 0.
        concentration=((a.mean(0).square().sum()+b.mean(0).square().sum())/2).item()
        overlap=float('nan')
        if n>=k+2:
            eye=torch.eye(n,device=a.device,dtype=torch.bool)
            ia=(a@a.T).masked_fill(eye,-1e4).topk(k,dim=1).indices
            ib=(b@b.T).masked_fill(eye,-1e4).topk(k,dim=1).indices
            overlap=(ia[:,:,None]==ib[:,None,:]).any(-1).float().mean().item()
        return {'margin':margin,'concentration':concentration,'overlap':overlap}


def structural_losses(a,b,temperature=.07):
    n=a.shape[0]
    a,b=F.normalize(a,dim=-1),F.normalize(b,dim=-1)
    allz=torch.cat([a,b],dim=0)
    gram=allz@allz.T
    mask=torch.ones((2*n,2*n),device=a.device,dtype=torch.bool)
    mask.fill_diagonal_(False)
    idx=torch.arange(n,device=a.device)
    mask[idx,idx+n]=False; mask[idx+n,idx]=False
    geo=gram[mask].square().mean() if n>1 else gram.sum()*0
    la=F.log_softmax((a@a.T)/temperature,dim=-1)
    lb=F.log_softmax((b@b.T)/temperature,dim=-1)
    pa=la.exp().clamp_min(1e-8); pb=lb.exp().clamp_min(1e-8)
    topo=(pa*(la-lb)+pb*(lb-la)).sum(dim=1).mean()/2
    return geo,topo.clamp(0,10)


def alignment_loss(a,b,temp=.05,label_smoothing=.05,margin=.3,hard_weight=.5):
    n=a.shape[0]; s=F.normalize(a,dim=-1)@F.normalize(b,dim=-1).T
    labels=torch.arange(n,device=a.device)
    contrast=(F.cross_entropy(s/temp,labels,label_smoothing=label_smoothing)+F.cross_entropy(s.T/temp,labels,label_smoothing=label_smoothing))/2
    if n<2:return contrast
    eye=torch.eye(n,device=a.device,dtype=torch.bool)
    hardest_row=s.masked_fill(eye,-1e4).max(1).values
    hardest_col=s.masked_fill(eye,-1e4).max(0).values
    hard=(F.relu(margin-s.diag()+hardest_row).mean()+F.relu(margin-s.diag()+hardest_col).mean())/2
    return contrast+hard_weight*hard


def ramp(x,lo,hi):return max(0.,min(1.,(x-lo)/(hi-lo)))

@dataclass
class Controller:
    method:str='calibrated'
    pair:str='en-fr'
    ema_beta:float=.9
    warmup:int=20
    threshold_g:float=.55
    threshold_n:float=.55
    disable_margin:bool=False
    disable_geometry:bool=False
    disable_neighborhood:bool=False
    disable_ema:bool=False
    disable_selection:bool=False
    def __post_init__(self):
        self.ema=None
        self.lambda_g=.0005
        self.lambda_n={'en-fr':.0005,'en-ru':.0004,'en-it':.0002,'en-es':.0002}.get(self.pair,.0002)
    def weights(self,raw,step):
        if not np.isfinite(list(raw.values())).all():return 0.,0.,{'eligible':False}
        if self.ema is None or self.disable_ema:self.ema=dict(raw)
        else:self.ema={k:self.ema_beta*self.ema[k]+(1-self.ema_beta)*v for k,v in raw.items()}
        m=ramp(self.ema['margin'],0,.30) if not self.disable_margin else 1.
        g=ramp(self.ema['concentration'],.05,.20) if not self.disable_geometry else 1.
        r=ramp(self.ema['overlap'],.15,.45) if not self.disable_neighborhood else 1.
        w=min(1.,step/self.warmup)
        if self.method=='alignment':return 0.,0.,{'eligible':True}
        if self.method=='fixed':return self.lambda_g,self.lambda_n,{'eligible':True}
        if self.method=='original':return self.lambda_g*w*m*g,self.lambda_n*w*m*r,{'eligible':True}
        if self.method=='selective':
            # Exploratory raw-diagnostic thresholds; not predictive calibration.
            eg=(self.ema['concentration']>=self.threshold_g) or self.disable_selection
            en=(self.ema['overlap']>=self.threshold_n) or self.disable_selection
            return (
                self.lambda_g*w*m*g*float(eg),
                self.lambda_n*w*m*r*float(en),
                {'eligible':True,'geometry_on':eg,'neighborhood_on':en}
            )
        if self.method=='calibrated':
            # Thresholds apply to bounded diagnostic factors. Tune on calibration split ONLY.
            eg=(g>=self.threshold_g) or self.disable_selection
            en=(r>=self.threshold_n) or self.disable_selection
            return self.lambda_g*w*m*g*float(eg),self.lambda_n*w*m*r*float(en),{'eligible':True,'geometry_on':eg,'neighborhood_on':en}
        raise ValueError(self.method)


def retrieval_metrics(a,b):
    s=F.normalize(a,dim=-1)@F.normalize(b,dim=-1).T
    order=s.argsort(dim=1,descending=True)
    ranks=(order==torch.arange(s.shape[0],device=s.device)[:,None]).int().argmax(1)+1
    return {'top1':float((ranks==1).float().mean()*100),'top5':float((ranks<=5).float().mean()*100),'mrr':float((1/ranks.float()).mean()),'mean_rank':float(ranks.float().mean())}
