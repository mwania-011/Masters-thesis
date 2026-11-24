import unittest
import numpy as np
import torch
from e1.diagnostics import alignment,geometry,neighborhood,evaluate,_topk_neighbors
from e1.interventions import verify_equal_initial_states,paired_deltas,safe_output_dir,create_plan

class E1Tests(unittest.TestCase):
    def setUp(self):
        self.a=np.eye(12,dtype=float)
        self.b=self.a.copy()
    def test_alignment(self):
        self.assertEqual(alignment(self.a,self.b)['top1_source_to_target'],1.)
    def test_diagnostics_reproducible(self):
        self.assertEqual(evaluate(self.a,self.b,k=3,seed=42),evaluate(self.a,self.b,k=3,seed=42))
    def test_self_neighbors_excluded(self):
        nbr=_topk_neighbors(self.a,3)
        self.assertTrue(all(i not in row for i,row in enumerate(nbr)))
    def test_bilingual_overlap(self):
        self.assertEqual(neighborhood(self.a,self.b,k=3)['cross_lingual_neighbor_overlap_at_k'],1.)
    def test_geometry_degenerate(self):
        x=np.tile([1.,0.],(12,1))
        self.assertTrue(np.isfinite(list(geometry(x).values())).all())
    def test_bad_vectors(self):
        with self.assertRaises(ValueError):alignment(np.zeros((12,4)),self.b)
    def test_matched_states(self):
        state={'model':{'weight':torch.arange(5.)},'epoch':3}
        self.assertTrue(verify_equal_initial_states([state,state]))
        with self.assertRaises(ValueError):
            verify_equal_initial_states([state,{'model':{'weight':torch.ones(5)},'epoch':3}])
    def test_delta(self):
        self.assertAlmostEqual(paired_deltas(.8,{'geo':.9})['geo'],.1)
    def test_plan(self):
        self.assertEqual(len(create_plan()),80)
        self.assertTrue(all(r['status'].startswith('BLOCKED') for r in create_plan()))
    def test_e0_guard(self):
        with self.assertRaises(ValueError):safe_output_dir('/tmp/e0/forbidden')

if __name__=='__main__':unittest.main()
