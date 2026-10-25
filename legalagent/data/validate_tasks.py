"""Validation for the active task set (600 tasks: 300 / 100 / 150 / 50); run after building tasks."""
import collections, hashlib, json, unittest

from ..paths import DATA, TASKS

BASE=TASKS

def read(path):
    return [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]

class DatasetValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus=read(DATA/'labor_subset.jsonl')
        cls.task=read(BASE/'tasks_all_reviewed.jsonl')
        cls.idx={(x['law_id'],x['article_number']):x for x in cls.corpus}
    def test_record_count_types_unique(self):
        self.assertEqual(len(self.task),600)
        self.assertEqual(len({t['task_id'] for t in self.task}),600)
        self.assertEqual(len({t['prompt'] for t in self.task}),600)
        self.assertEqual(dict(collections.Counter(t['task_type'] for t in self.task)),{'direct_retrieval':127,'fact_application':237,'multi_article':173,'insufficient_information':63})
        self.assertEqual(dict(collections.Counter(t['split'] for t in self.task)),{'train':300,'dev':100,'test':150,'badcase':50})
    def test_all_sources_present_hash_correct(self):
        for t in self.task:
            with self.subTest(tid=t['task_id']):
                self.assertEqual(t['applicable_date'],'2024-06-01')
                ids=[]
                for e in t['source_evidence']:
                    law,num=e['article_id'].rsplit(':',1)
                    row=self.idx[(law,int(num))]
                    self.assertEqual(row['text_sha256'],hashlib.sha256(row['text'].encode('utf-8')).hexdigest())
                    self.assertEqual(row['text_sha256'],e['text_sha256'])
                    self.assertEqual(row['version_id'],e['version_id'])
                    self.assertIn(e['supporting_quote'],row['text'])
                    self.assertLessEqual(row['effective_from'],t['applicable_date'])
                    self.assertTrue(row['effective_to'] is None or t['applicable_date'] < row['effective_to'])
                    ids.append(e['article_id'])
                self.assertEqual(set(ids),set(t['gold_article_ids']))
                self.assertEqual(len(t['gold_article_ids']),len(set(t['gold_article_ids'])))
                self.assertTrue(t['expected_answer_points'])
                self.assertTrue(t['gold_issues'])
                self.assertEqual(t['review']['independent_human_legal_expert_review'],'pending')
    def test_fixed_splits(self):
        ids=set()
        for group,expected in [('train',300),('dev',100),('test',150),('badcase',50)]:
            labeled=read(BASE/f'tasks_{group}_labeled.jsonl')
            prompts=read(BASE/f'tasks_{group}_prompts_only.jsonl')
            self.assertEqual(len(labeled),expected)
            self.assertEqual(len(prompts),expected)
            self.assertEqual({x['task_id'] for x in labeled},{x['task_id'] for x in prompts})
            self.assertFalse(ids.intersection({x['task_id'] for x in labeled}))
            ids.update(x['task_id'] for x in labeled)
            for x in prompts:
                self.assertEqual(set(x),{'task_id','prompt','jurisdiction','applicable_date'})
                self.assertNotIn('gold_',json.dumps(x))
        self.assertEqual(ids,{t['task_id'] for t in self.task})
        badcase={x['task_id'] for x in read(BASE/'tasks_badcase_labeled.jsonl')}
        train_dev_test={x['task_id'] for x in self.task if x['split']!='badcase'}
        self.assertFalse(badcase & train_dev_test)
    def test_unsafe_corpus_article_not_labelled(self):
        # These older provisions require cross-law/higher-ranked-law qualification; avoid as hard gold.
        used=set(e['article_id'] for t in self.task for e in t['source_evidence'])
        self.assertFalse({'labor_law:82','labor_law:36','labor_law:62'} & used)
    def test_insufficient_facts_explicit(self):
        cases=[t for t in self.task if t['task_type']=='insufficient_information']
        self.assertEqual(len(cases),63)
        self.assertTrue(all(t['missing_facts'] for t in cases))

if __name__=='__main__':
    unittest.main(verbosity=2)
