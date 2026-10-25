import unittest

from legalagent.analysis.bad_case_sampler import compute_weights, sample_curriculum

LABELS = [{"task_id": "A"}, {"task_id": "B"}, {"task_id": "C"}]
PROMPTS = [{"task_id": k, "prompt": k} for k in 'ABC']
GROUPS = [
    {"task_id": "A", "policy_version": "s10", "rewards": [0, .2, .8, 1],
     "successes": [0, 0, 1, 1], "error_types": [["citation_error"], ["citation_error"], [], []]},
    {"task_id": "B", "policy_version": "s10", "rewards": [0, 0, 0, 0],
     "successes": [0, 0, 0, 0], "error_types": [["retrieval_error"]] * 4},
    {"task_id": "C", "policy_version": "s10", "rewards": [1, 1, 1, 1],
     "successes": [1, 1, 1, 1], "error_types": [[]] * 4},
]

class TestSampler(unittest.TestCase):
    def test_prefer_learnable_groups(self):
        p, audit = compute_weights(LABELS, GROUPS)
        self.assertAlmostEqual(sum(p.values()), 1)
        self.assertGreater(p['A'], p['B'])
        self.assertGreater(p['B'], p['C'])
        self.assertGreaterEqual(min(p.values()), .3 / 3)
        self.assertEqual(audit['num_homogeneous_reward_groups'], 2)

    def test_sample_has_no_labels(self):
        p, _ = compute_weights(LABELS, GROUPS)
        rows = sample_curriculum(PROMPTS, p, 100, 7)
        self.assertEqual(len(rows), 100)
        self.assertTrue(all('gold_article_ids' not in x for x in rows))
        self.assertEqual(rows, sample_curriculum(PROMPTS, p, 100, 7))

    def test_reject_stale_policy_mix(self):
        groups = [dict(g) for g in GROUPS]
        groups[1]['policy_version'] = 's11'
        with self.assertRaises(ValueError):
            compute_weights(LABELS, groups)

    def test_reject_nontrain_groups(self):
        groups = [dict(g) for g in GROUPS]
        groups[1]['task_id'] = 'TEST_001'
        with self.assertRaises(ValueError):
            compute_weights(LABELS, groups)

if __name__ == '__main__':
    unittest.main()
