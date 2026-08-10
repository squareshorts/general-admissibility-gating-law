import math, unittest
import numpy as np
from src.simulation.analytical import acquisition_cost_delta, complete_metrics, convention_losses, prevalence_boundary
from src.simulation.finite_sample import run_finite_sample, wilson_interval
from src.simulation.score_level import ScoreModel,bayes_three_actions,gate_parameters,generate_scores,known_model_posterior,posterior_expected_risk,same_score_equivalence
from src.theory.gating import Losses,Parameters,analytical_metrics

class AnalyticalSimulationTests(unittest.TestCase):
    def test_three_withholding_conventions(self):
        a=convention_losses('pure_abstention',c_fp=8,c_fn=5,c_a=2); b=convention_losses('abstention_plus_missed',c_fp=8,c_fn=5,c_a=2); c=convention_losses('treated_as_negative',c_fp=8,c_fn=5,c_a=2); self.assertEqual((a.withheld_true,a.withheld_false),(2,2)); self.assertEqual((b.withheld_true,b.withheld_false),(7,2)); self.assertEqual((c.withheld_true,c.withheld_false),(5,0))
    def test_complete_metrics_direct_probabilities(self):
        m=complete_metrics(Parameters(.1,.8,.2,.75,.25),Losses(10,4,5,1)); self.assertAlmostEqual(m['operational_sensitivity'],.6); self.assertAlmostEqual(m['false_action_rate'],.05)
    def test_exact_prevalence_boundary(self):
        losses=Losses(10,4,5,1); t=Parameters(.2,.85,.12,.85,.30); pi=prevalence_boundary(t,losses); self.assertAlmostEqual(analytical_metrics(Parameters(pi,.85,.12,.85,.30),losses)['delta_loss'],0,places=12)
    def test_acquisition_cost_extension(self):
        p=Parameters(.05,.85,.12,.85,.30); losses=Losses(10,4,5,1); base=analytical_metrics(p,losses)['delta_loss']; self.assertAlmostEqual(acquisition_cost_delta(p,losses,2),base+2*(.05*.85+.95*.12))
    def test_boundary_cases(self):
        losses=Losses(10,4,5,1); self.assertEqual(complete_metrics(Parameters(.1,.8,.2,.8,0),losses)['false_action_rate'],0); self.assertEqual(complete_metrics(Parameters(.1,.8,.2,0,.3),losses)['operational_sensitivity'],0); self.assertAlmostEqual(complete_metrics(Parameters(.1,.8,.2,.4,.4),losses)['ppv_change'],0)

class ScoreSimulationTests(unittest.TestCase):
    def test_fixed_seed_reproducibility(self):
        a=generate_scores(2000,ScoreModel(),123); b=generate_scores(2000,ScoreModel(),123); [np.testing.assert_array_equal(a[k],b[k]) for k in a]
    def test_same_score_equivalence_observationwise(self):
        s=np.linspace(-4,4,10001); self.assertTrue(same_score_equivalence(s,.3,1.1)); self.assertTrue(same_score_equivalence(s,1.1,.3))
    def test_dependence_does_not_break_binary_identity(self):
        d=generate_scores(250000,ScoreModel(rho0=.8,rho1=-.5),987); gp=gate_parameters(d,.55,.35); truth=d['truth']; candidate=d['s']>=.55; p=Parameters(float(np.mean(truth)),float(np.mean(candidate[truth])),float(np.mean(candidate[~truth])),gp['q1'],gp['q0']); accepted=candidate&(d['z']>=.35); self.assertAlmostEqual(np.mean(accepted[truth]),p.sensitivity*p.retain_true_candidate,places=12)
    def test_known_model_bayes_policy_minimizes_conditional_risk(self):
        model=ScoreModel(rho0=.2,rho1=.2); d=generate_scores(20000,model,159); prob=known_model_posterior(d['s'],d['z'],model); losses=Losses(5,1,1.2,.2); a,w=bayes_three_actions(prob,losses); opt=posterior_expected_risk(prob,a,w,losses); neg=posterior_expected_risk(prob,np.zeros(len(a),bool),np.zeros(len(a),bool),losses); self.assertLessEqual(opt,neg)
    def test_truth_independent_gate_thins_equally(self):
        d=generate_scores(400000,ScoreModel(),456); rng=np.random.default_rng(789); c=d['s']>=.55; a=c&(rng.random(len(c))<.4); n1=c[d['truth']].sum(); n0=c[~d['truth']].sum(); q1=a[d['truth']].sum()/n1; q0=a[~d['truth']].sum()/n0; se=math.sqrt(.4*.6*(1/n1+1/n0)); self.assertLessEqual(abs(q1-q0),5*se)
    def test_finite_sample_reproducibility(self):
        p=Parameters(.01,.85,.12,.85,.3); losses=Losses(10,4,5,1); self.assertTrue(run_finite_sample(p,losses,n=1000,repeats=10,seed=42).equals(run_finite_sample(p,losses,n=1000,repeats=10,seed=42)))
    def test_wilson_zero_not_zero_width(self):
        lo,hi=wilson_interval(0,20); self.assertEqual(lo,0); self.assertGreater(hi,0)

if __name__=='__main__': unittest.main()
