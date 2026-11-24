"""Run with: pytest -q tests/test_e2_gates.py"""
import torch
from thesis.experiments.run_e2_gated import diagnostics, weights, ramp, CONFIG

def test_ramp_bounds():
    assert ramp(-10,0,1)==0
    assert ramp(10,0,1)==1
    assert ramp(.5,0,1)==.5

def test_independent_gates_and_bounds():
    a={'alignment_margin':.3,'concentration':.2,'neighborhood_overlap':0.0}
    lg,lt,gg,tg=weights(a,20,'en-it')
    assert lg==.0005 and lt==0 and gg==1 and tg==0
    b={'alignment_margin':.3,'concentration':0.,'neighborhood_overlap':.45}
    lg,lt,gg,tg=weights(b,20,'en-it')
    assert lg==0 and lt==.0002 and gg==0 and tg==1

def test_alignment_maturity_suppresses_both():
    lg,lt,gg,tg=weights({'alignment_margin':-.2,'concentration':1.,'neighborhood_overlap':1.},20,'en-ru')
    assert (lg,lt,gg,tg)==(0,0,0,0)

def test_warmup_and_pair_coefficients():
    d={'alignment_margin':.3,'concentration':.2,'neighborhood_overlap':.45}
    assert weights(d,1,'en-fr')[:2]==(.0005/20,.0005/20)
    assert weights(d,20,'en-ru')[:2]==(.0005,.0004)

def test_diagnostics_finite_and_nonnegative_overlap():
    g=torch.Generator().manual_seed(10)
    a=torch.randn(16,32,generator=g)
    b=a+.01*torch.randn(16,32,generator=g)
    d=diagnostics(a,b,5)
    assert set(d)=={'alignment_margin','concentration','neighborhood_overlap'}
    assert 0<=d['neighborhood_overlap']<=1
    assert 0<=d['concentration']<=1.000001
    assert all(torch.isfinite(torch.tensor(x)) for x in d.values())
