"""Run: PYTHONPATH=. pytest -q tests/test_e1b.py (from repository root after install)."""
import pytest
import torch
from thesis.experiments.run_e1b import (
    BRANCHES, coefficients, canonical_hash, schedule_from_generator, validate_output,
)

def test_branch_coefficients():
    expected={'en-fr':.0005,'en-ru':.0004,'en-it':.0002,'en-es':.0002}
    for pair,topo in expected.items():
        assert coefficients(pair,'control')==(0.,0.)
        assert coefficients(pair,'geo')==(.0005,0.)
        assert coefficients(pair,'topo')==(0.,topo)
        assert coefficients(pair,'full')==(.0005,topo)

def test_all_four_branches():
    assert set(BRANCHES)=={'control','geo','topo','full'}

def test_schedule_replay_is_exact():
    gen=torch.Generator().manual_seed(101)
    state=gen.get_state()
    a=schedule_from_generator(101,32,state,3)
    b=schedule_from_generator(101,32,state,3)
    assert a==b and len(a)==3
    assert len({i for batch in a for i in batch})==96
    assert canonical_hash(a)==canonical_hash(b)

def test_schedule_refuses_more_than_one_epoch():
    gen=torch.Generator().manual_seed(101)
    with pytest.raises(ValueError):
        schedule_from_generator(10,4,gen.get_state(),4)

def test_state_digest_detects_mutation():
    a={'model':{'x':torch.arange(4,dtype=torch.float32)},'optimizer':{'step':2}}
    b={'model':{'x':torch.arange(4,dtype=torch.float32)},'optimizer':{'step':2}}
    assert canonical_hash(a)==canonical_hash(b)
    b['model']['x'][0]=99
    assert canonical_hash(a)!=canonical_hash(b)

def test_output_guard():
    with pytest.raises(ValueError): validate_output('thesis/results/e0/anything')
    with pytest.raises(ValueError): validate_output('thesis/results/e1/b_interventions')

def test_output_guard_no_overwrite(tmp_path,monkeypatch):
    import thesis.experiments.run_e1b as e1b
    monkeypatch.setattr(e1b,'ROOT',tmp_path)
    p=tmp_path/'smoke'/'run'
    assert e1b.validate_output(p)==p.resolve()
    p.mkdir(parents=True)
    with pytest.raises(FileExistsError): e1b.validate_output(p)
