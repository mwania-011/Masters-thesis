import unittest
import torch
from rg_core import Controller,diagnostics,structural_losses,alignment_loss,retrieval_metrics
class TestCore(unittest.TestCase):
 def setUp(self):
  torch.manual_seed(3);self.a=torch.nn.functional.normalize(torch.randn(16,256),dim=-1);self.b=torch.nn.functional.normalize(self.a+.03*torch.randn(16,256),dim=-1)
 def test_metrics(self):
  m=retrieval_metrics(self.a,self.b);self.assertEqual(m['top1'],100.)
 def test_losses(self):
  g,n=structural_losses(self.a,self.b);self.assertTrue(torch.isfinite(g+n+alignment_loss(self.a,self.b)))
 def test_diagnostics(self):
  d=diagnostics(self.a,self.b);self.assertGreaterEqual(d['overlap'],0);self.assertLessEqual(d['overlap'],1)
 def test_gating(self):
  c=Controller();w=c.weights({'margin':.2,'concentration':.02,'overlap':.1},25)
  self.assertEqual(w[0],0);self.assertEqual(w[1],0)
 def test_undersized(self):
  d=diagnostics(self.a[:5],self.b[:5]);self.assertTrue(torch.isnan(torch.tensor(d['overlap'])))
  self.assertEqual(Controller().weights(d,1)[:2],(0.,0.))
if __name__=='__main__':unittest.main()
