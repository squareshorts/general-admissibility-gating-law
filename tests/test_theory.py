import math
import unittest

from src.theory.gating import Losses, Parameters, analytical_metrics, simulate


class TheoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.zero_baselines = Losses(false_positive=10.0,false_negative=3.0,withheld_true=2.0,withheld_false=1.0)
    def test_loss_difference_identity(self):
        p=Parameters(0.08,0.90,0.12,0.85,0.30); r=analytical_metrics(p,self.zero_baselines); self.assertAlmostEqual(r['delta_loss'],r['delta_loss_identity'])
    def test_positive_delta_q_improves_ppv(self):
        p=Parameters(0.10,0.80,0.20,0.90,0.40); r=analytical_metrics(p,self.zero_baselines); self.assertGreater(r['ppv_gated'],r['ppv_ungated'])
    def test_negative_delta_q_reduces_ppv(self):
        p=Parameters(0.10,0.80,0.20,0.20,0.40); r=analytical_metrics(p,self.zero_baselines); self.assertLess(r['ppv_gated'],r['ppv_ungated'])
    def test_truth_independent_gate_preserves_ppv(self):
        p=Parameters(0.10,0.80,0.20,0.35,0.35); r=analytical_metrics(p,self.zero_baselines); self.assertAlmostEqual(r['ppv_gated'],r['ppv_ungated'])
    def test_ppv_sign_criterion_over_parameter_grid(self):
        for pi in (0.01,0.2,0.8):
            for s in (0.1,0.7,1.0):
                for f in (0.05,0.4,1.0):
                    for q1 in (0.1,0.5,0.9):
                        for q0 in (0.1,0.5,0.9):
                            r=analytical_metrics(Parameters(pi,s,f,q1,q0),self.zero_baselines); observed=r['ppv_gated']-r['ppv_ungated']; expected=q1-q0
                            if expected>0:self.assertGreater(observed,0)
                            elif expected<0:self.assertLess(observed,0)
                            else:self.assertAlmostEqual(observed,0)
    def test_fdr_is_one_minus_ppv(self):
        r=analytical_metrics(Parameters(0.12,0.77,0.19,0.81,0.24),self.zero_baselines); self.assertAlmostEqual(r['fdr_ungated'],1-r['ppv_ungated']); self.assertAlmostEqual(r['fdr_gated'],1-r['ppv_gated'])
    def test_false_actions_and_sensitivity_identities(self):
        p=Parameters(0.07,0.83,0.13,0.72,0.21); r=analytical_metrics(p,self.zero_baselines); self.assertAlmostEqual(r['sensitivity_gated'],.83*.72); self.assertAlmostEqual(r['false_actions_ungated']-r['false_actions_gated'],(1-.07)*.13*(1-.21))
    def test_delta_q_not_sufficient_for_loss(self):
        p=Parameters(.5,.5,.5,.9,.8); r=analytical_metrics(p,Losses(1,2,10,10)); self.assertGreater(p.retain_true_candidate-p.retain_false_candidate,0); self.assertGreater(r['delta_loss'],0)
    def test_delta_q_not_necessary_for_loss(self):
        p=Parameters(.001,.5,.5,.2,.3); r=analytical_metrics(p,Losses(100,1,1,0)); self.assertLess(p.retain_true_candidate-p.retain_false_candidate,0); self.assertLess(r['delta_loss'],0)
    def test_no_actions_has_undefined_ppv(self):
        self.assertTrue(math.isnan(analytical_metrics(Parameters(.1,.8,.2,0,0),self.zero_baselines)['ppv_gated']))
    def test_monte_carlo_matches_analytical_loss(self):
        p=Parameters(.12,.78,.18,.82,.25); exact=analytical_metrics(p,self.zero_baselines); sampled=simulate(p,self.zero_baselines,observations=250000,seed=314159); self.assertAlmostEqual(sampled['delta_loss'],exact['delta_loss'],delta=.02)
    def test_invalid_probability_rejected(self):
        with self.assertRaises(ValueError): Parameters(1.1,.8,.1,.9,.2)

if __name__=='__main__': unittest.main()
