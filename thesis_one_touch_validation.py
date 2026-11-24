#!/usr/bin/env python3
"""Non-destructive, one-command thesis convergence audit and frozen next-stage protocol.
Runs locally on the GPU project; does not launch unsupported experimental variants.
"""
import csv, hashlib, json, platform, sys
from pathlib import Path
from datetime import datetime, timezone
from statistics import mean

ROOT=Path.cwd(); RUNS=ROOT/'thesis/results/e0'; AUDIT=ROOT/'thesis/audits/final_validation'
AUDIT.mkdir(parents=True,exist_ok=True)
PAIRS=['en-fr','en-ru','en-it','en-es']; SEEDS=range(1,6); VARIANTS=['baseline','geo','full']
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def num(v):
 try:return float(v)
 except (ValueError,TypeError):return None

def convergence(rows):
 vals=[num(r.get('Top-1')) for r in rows]
 if not vals or any(v is None for v in vals):return {'status':'missing_validation_top1'}
 best=max(vals); epoch=vals.index(best)+1
 # diagnostic, not a formal convergence determination
 tail=vals[-5:]; slope=(tail[-1]-tail[0])/(len(tail)-1) if len(tail)>1 else None
 return {'observed_epochs':len(vals),'best_epoch':epoch,'best_validation_top1':best,
         'final_validation_top1':vals[-1],'last_five_epoch_slope':slope,
         'best_at_last_epoch':epoch==len(vals),'convergence_certified':False,
         'note':'Convergence requires longer-horizon validation with a frozen stopping rule.'}

rows=[]; missing=[]; suspicious=[]
for pair in PAIRS:
 for seed in SEEDS:
  for variant in VARIANTS:
   name=f'{pair}_seed{seed}_{variant}'; d=RUNS/name; ep=d/'epochs.csv'; sm=d/'summary.json'
   if not ep.exists():missing.append(str(ep));continue
   with ep.open(newline='') as f: hist=list(csv.DictReader(f))
   r={'run':name,'pair':pair,'seed':seed,'variant':variant,'epochs_csv_sha256':sha(ep),
      'summary_exists':sm.is_file(),'last_checkpoint_exists':(d/'last.pt').is_file(),
      'best_checkpoint_exists':(d/'best.pt').is_file(),**convergence(hist)}
   rows.append(r)
   if r.get('best_at_last_epoch'):suspicious.append(name)
source={}
for f in ['thesis/experiments/run_e0.py','thesis/experiments/e0_config.py',
          'thesis/experiments/run_e1b.py','thesis/experiments/run_e2_gated.py',
          'train_structure_preserving_multilingual.py','thesis/evaluation/retrieval.py',
          'thesis/data/dataset.py','thesis/data/prepare_datasets.py']:
 p=ROOT/f;source[f]=sha(p) if p.is_file() else None
protocol='''# Frozen prospective thesis-validation plan (proposed, NOT executed)

## Primary questions
1. Does a pre-intervention diagnostic predict the *held-out* paired benefit of the corresponding structural objective?
2. Does representation-state-dependent control outperform constant and time-matched regularization at equivalent training budgets?
3. Do any effects persist under convergence-based training and across random seeds?

## Training and selection
- Use the same processed train/validation/test partitions, model architecture, tokenization, and five seeds for every comparator.
- Train independent baselines and controller variants under a common maximum budget of 40 epochs (a ceiling, not a claim that 40 are necessary).
- Prespecify validation Top-1 as the primary early-stopping endpoint, with minimum 5 epochs, patience 5 complete validations, and a minimum meaningful improvement of 0.0001 in fractional accuracy.
- Save per-epoch curves, best checkpoint, optimizer state, coefficient trace, wall time, and GPU memory; test only once after validation-based selection.
- Also report fixed-update-budget comparisons, to separate training-time differences from final quality.
- For long-horizon matched continuations, freeze the same initial checkpoint, shuffled-data schedule, and training budget; evaluate at 1, 5, 10 epochs and at stopping. These are distinct from independently trained baselines.

## Prospective intervention-prediction test
- Prespecify diagnostic features and benefit thresholds using only a development partition of pair-seed units.
- Validate predictive association and direction on untouched held-out units; report correlation, sign-prediction performance, calibration and uncertainty.
- Five seeds and four language pairs provide limited independent units: avoid treating per-batch measurements as independent replications.

## Controller mechanism comparison
- Alignment only; geometry only; neighborhood only; combined fixed coefficients.
- Existing diagnostic-modulated controller; fixed coefficients matched to its average exposure; time-dependent diagnostic-independent schedule; shuffled diagnostic control; calibrated selective controller.
- The last three require implementing new runners; they do not exist in the verified archive.
- Match regularization exposure and optimization steps where applicable; freeze all tuning on validation data.

## Analysis
- Primary: paired Top-1 difference in percentage points with 95% CIs and five-seed distribution.
- Secondary: Top-5, mean rank, alignment margin, uniformity, effective rank, neighborhood overlap, perturbation stability, runtime and GPU memory.
- Apply a prespecified multiple-comparison correction to confirmatory tests; distinguish exploratory results.
- Analyze convergence curves, coefficient activation, thresholds and failure conditions.

## Go/no-go
DO NOT claim convergence or controller superiority based on the existing one-epoch continuation alone.
DO NOT rerun or overwrite archived reference, diagnostic, intervention, or controller results.
DO NOT launch unimplemented comparators or rewrite frozen runners in place.
'''
report={'created_utc':datetime.now(timezone.utc).isoformat(),'python':sys.version,'platform':platform.platform(),
        'root':str(ROOT),'reference_runs_expected':60,'reference_runs_found':len(rows),
        'missing_epoch_logs':missing,'runs_best_at_final_epoch':suspicious,
        'source_sha256':source,'reference_convergence_audit':rows,
        'next_stage_experiments_executed':False,
        'readiness':'NEEDS_IMPLEMENTATION_AND_FROZEN_PROTOCOL_FOR_NEW_LONG_HORIZON_EXPERIMENTS'}
(AUDIT/'convergence_audit.json').write_text(json.dumps(report,indent=2)+'\n')
(AUDIT/'prospective_validation_protocol.md').write_text(protocol)
with (AUDIT/'convergence_summary.csv').open('w',newline='') as f:
 fields=['run','pair','seed','variant','observed_epochs','best_epoch','best_validation_top1','final_validation_top1','last_five_epoch_slope','best_at_last_epoch','convergence_certified']
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows([{k:r.get(k) for k in fields} for r in rows])
print('THESIS ONE-TOUCH AUDIT COMPLETE')
print('Reference logs:',len(rows),'/60; missing:',len(missing))
print('Best at final epoch:',len(suspicious))
print('Convergence certified: NO (requires longer training and frozen stopping rules)')
print('New GPU experiments launched: NO (not yet supported by verified runners)')
print('Artifacts:',AUDIT/'convergence_audit.json',AUDIT/'convergence_summary.csv',AUDIT/'prospective_validation_protocol.md',sep='\n  ')
if missing:sys.exit(2)
