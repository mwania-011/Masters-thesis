"""Generate Figures 3,4,5,7 from REAL experiment logs, never synthetic data.
Figure 3 needs probes.csv columns: margin,concentration,overlap,delta_top1,split
where delta_top1 = intervention validation Top-1 minus matched no-intervention Top-1.
The calibration model is fit ONLY on rows with split=calibration, tested on split=test.
"""
import argparse,json,csv,glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def save(fig,out,name):
    fig.tight_layout();fig.savefig(out/(name+'.png'),dpi=250,bbox_inches='tight');fig.savefig(out/(name+'.pdf'),bbox_inches='tight');plt.close(fig)

def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',required=True);p.add_argument('--out',required=True);p.add_argument('--probes',default='')
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    paths=list(Path(args.runs).rglob('history.json'))
    if not paths:raise SystemExit('No history.json found. Run actual training first.')
    rows=[];traces=[]
    for path in paths:
        h=json.loads(path.read_text());rows.extend(h)
        trace=path.with_name('traces.json')
        if trace.exists():
            for r in json.loads(trace.read_text()):traces.append({**r,'method':h[0]['method'],'ablate':h[0]['ablate'],'pair':h[0]['pair'],'seed':h[0]['seed']})
    df=pd.DataFrame(rows);df.to_csv(out/'measured_epoch_results.csv',index=False)
    final=df.sort_values('epoch').groupby(['method','ablate','pair','seed'],as_index=False).tail(1)
    final.to_csv(out/'measured_final_results.csv',index=False)
    # Figure 4: mean accuracy +/- between-seed SEM (only if repeated seeds)
    fig,ax=plt.subplots(figsize=(8,4.8))
    for (method,ablate),g in df.groupby(['method','ablate']):
        s=g.groupby('epoch')['top1'].agg(['mean','std','count']).reset_index()
        label=method if ablate=='none' else method+' / -'+ablate
        ax.plot(s.epoch,s['mean'],marker='o',markersize=3,label=label)
        if s['count'].min()>1:
            err=1.96*s['std'].fillna(0)/np.sqrt(s['count'])
            ax.fill_between(s.epoch,s['mean']-err,s['mean']+err,alpha=.12)
    ax.set(xlabel='Epoch',ylabel='Validation Top-1 (%)',title='Figure 4 | Measured convergence')
    ax.legend(fontsize=8);ax.grid(alpha=.2);save(fig,out,'Figure_4_Convergence')
    # Figure 5: measured controller diagnostic and coefficient dynamics
    if traces:
        td=pd.DataFrame(traces)
        sel=td[(td.method=='calibrated') & (td.ablate=='none')]
        if not sel.empty:
            sel=sel.groupby('step',as_index=False)[['margin','concentration','overlap','lambda_geometry','lambda_neighborhood']].mean()
            fig,axes=plt.subplots(2,1,figsize=(8,6.5),sharex=True)
            for col in ['margin','concentration','overlap']:axes[0].plot(sel.step,sel[col],label=col)
            for col in ['lambda_geometry','lambda_neighborhood']:axes[1].plot(sel.step,sel[col],label=col)
            axes[0].set_ylabel('Diagnostic value');axes[1].set_ylabel('Effective loss coefficient');axes[1].set_xlabel('Optimization step')
            for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
            fig.suptitle('Figure 5 | Measured controller behavior');save(fig,out,'Figure_5_Controller_Behavior')
    # Figure 7: paired performance versus measured runtime
    agg=final.groupby(['method','ablate'],as_index=False).agg(top1=('top1','mean'),elapsed_seconds=('elapsed_seconds','mean'))
    fig,ax=plt.subplots(figsize=(8,5))
    for _,r in agg.iterrows():
        ax.scatter(r.elapsed_seconds/3600,r.top1,s=80)
        ax.annotate(r.method+(' / -'+r.ablate if r.ablate!='none' else ''),(r.elapsed_seconds/3600,r.top1),xytext=(5,5),textcoords='offset points',fontsize=8)
    ax.set(xlabel='Training time (GPU hours, measured)',ylabel='Mean validation Top-1 (%)',title='Figure 7 | Measured accuracy-efficiency')
    ax.grid(alpha=.2);save(fig,out,'Figure_7_Accuracy_Efficiency')
    if args.probes:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_curve,roc_auc_score,brier_score_loss
        from sklearn.calibration import calibration_curve
        probes=pd.read_csv(args.probes)
        fields=['margin','concentration','overlap'];probes=probes.dropna(subset=fields+['delta_top1','split'])
        train=probes[probes.split=='calibration'];test=probes[probes.split=='test']
        if len(train)<20 or len(test)<20:raise ValueError('Need >=20 calibration and >=20 held-out test probes')
        y=(train.delta_top1>0).astype(int);yt=(test.delta_top1>0).astype(int)
        if y.nunique()<2 or yt.nunique()<2:raise ValueError('Both outcome classes required in each split')
        fig,axes=plt.subplots(1,2,figsize=(10,4.3));report=[]
        for name,cols in [('margin',['margin']),('geometry',['concentration']),('neighborhood',['overlap']),('combined',fields)]:
            clf=LogisticRegression(max_iter=1000).fit(train[cols],y)
            pr=clf.predict_proba(test[cols])[:,1]
            auc=roc_auc_score(yt,pr);fpr,tpr,_=roc_curve(yt,pr)
            axes[0].plot(fpr,tpr,label=f'{name}: {auc:.3f}')
            obs,pred=calibration_curve(yt,pr,n_bins=6,strategy='quantile')
            axes[1].plot(pred,obs,marker='o',label=name)
            report.append({'predictor':name,'heldout_auroc':auc,'heldout_brier':brier_score_loss(yt,pr)})
        for ax in axes:ax.plot([0,1],[0,1],'k--',alpha=.4);ax.legend(fontsize=8);ax.grid(alpha=.2)
        axes[0].set(xlabel='False positive rate',ylabel='True positive rate',title='Held-out ROC')
        axes[1].set(xlabel='Predicted benefit probability',ylabel='Observed benefit frequency',title='Held-out calibration')
        fig.suptitle('Figure 3 | Predicting beneficial interventions');save(fig,out,'Figure_3_Predictive_Validity')
        pd.DataFrame(report).to_csv(out/'measured_predictive_validity.csv',index=False)
    print('Wrote measured results and available figures to',out)
if __name__=='__main__':main()
