"""Prompt isolation matters: no reference leakage and one controlled intervention."""
import unittest
from evaluation import BENCHMARK, answer_messages

class EvaluationPrompts(unittest.TestCase):
    def test_references_are_not_included_in_no_context_answers(self):
        for case in BENCHMARK['cases']:
            messages = answer_messages(case, [], 'json')
            self.assertEqual(messages[-1]['content'], 'Sources:\n(none supplied)\n\nQuestion: ' + case['question'])
            self.assertNotIn('reference', messages[-1]['content'])

    def test_context_is_only_paired_prompt_change(self):
        case = BENCHMARK['cases'][0]
        a=answer_messages(case, [], 'json')
        b=answer_messages(case, BENCHMARK['sources'], 'json')
        self.assertEqual(a[0],b[0])
        self.assertEqual(a[1]['content'].split('Question: ')[1],b[1]['content'].split('Question: ')[1])
        for source in BENCHMARK['sources']:
            self.assertIn(source['text'], b[1]['content'])
        self.assertEqual(len(BENCHMARK['cases']),4)
        self.assertEqual(len({c['id'] for c in BENCHMARK['cases']}),4)

if __name__ == '__main__':
    unittest.main()
