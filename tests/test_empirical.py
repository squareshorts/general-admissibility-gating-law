import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from src.empirical.datasets import deterministic_split
from src.empirical.evaluation import binary_metrics,matched_random_gate,select_youden_threshold

class EmpiricalEvaluationTests(unittest.TestCase):
    def test_metrics_match_gate_identities(self):
        m=binary_metrics(np.array([1,1,0,0,0],bool),np.ones(5,bool),np.array([1,0,1,0,0],bool)); self.assertAlmostEqual(m['q1'],.5); self.assertAlmostEqual(m['q0'],1/3); self.assertAlmostEqual(m['action_ppv'],.5); self.assertAlmostEqual(m['false_candidates_removed'],2/3)
    def test_actions_must_be_candidates(self):
        with self.assertRaises(ValueError): binary_metrics(np.array([1,0]),np.array([0,1]),np.array([1,0]))
    def test_outcome_independent_split_is_deterministic(self):
        self.assertEqual(deterministic_split('a103l'),deterministic_split('a103l')); self.assertIn(deterministic_split('v100s'),{'train','validation','test'})
    def test_youden_threshold_separates_ordered_scores(self):
        scores=np.array([.05,.10,.15,.85,.90,.95]); truth=np.array([0,0,0,1,1,1],bool); cut=select_youden_threshold(scores,truth); self.assertTrue(np.all((scores>=cut)==truth))
    def test_matched_random_gate_reproducible(self):
        c=np.array([1,0,1,1,0],bool); a=matched_random_gate(c,.5,20,9); b=matched_random_gate(c,.5,20,9); np.testing.assert_array_equal(a,b); self.assertTrue(np.all(~a|c[None,:]))

ROOT=Path(__file__).resolve().parents[1]
@unittest.skipUnless((ROOT/'results'/'empirical'/'main_metrics.csv').exists(),'empirical results not present')
class EmpiricalResultIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.metrics=pd.read_csv(ROOT/'results'/'empirical'/'main_metrics.csv'); cls.counts=pd.read_csv(ROOT/'results'/'empirical'/'candidate_counts.csv')
    def test_ppv_identity_from_reported_gate_counts(self):
        for r in self.metrics[self.metrics.strategy=='Hard gate'].itertuples():
            v=(r.candidate_ppv*r.q1)/(r.candidate_ppv*r.q1+(1-r.candidate_ppv)*r.q0); self.assertAlmostEqual(r.action_ppv,v,places=12)
    def test_strategies_use_same_test_n(self):
        for _,g in self.metrics.groupby('dataset'): self.assertEqual(g.n.nunique(),1)

if __name__=='__main__': unittest.main()
