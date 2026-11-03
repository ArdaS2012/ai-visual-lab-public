import test from 'node:test';
import assert from 'node:assert/strict';
import { EVALUATION_CASES, evaluateCase, summarizeEvaluation } from '../src/lib/evaluation.ts';
test('paired fixtures preserve correctness-format tradeoff and true recall denominator',()=>{
 const baseline=summarizeEvaluation(EVALUATION_CASES.baseline,2);
 assert.equal(baseline.exact,.25);assert.equal(baseline.schema,1);assert.equal(baseline.recall,1/6);assert.equal(baseline.recallN,3);
 const changed=summarizeEvaluation(EVALUATION_CASES.changed,2);
 assert.equal(changed.exact,1);assert.equal(changed.schema,.75);assert.equal(changed.recall,1);
});
test('cutoff changes recall without changing saved answer scores',()=>{
 const a=summarizeEvaluation(EVALUATION_CASES.baseline,2),b=summarizeEvaluation(EVALUATION_CASES.baseline,3);
 assert.equal(b.recall,1);assert.equal(a.exact,b.exact);assert.equal(a.schema,b.schema);
});
test('duplicates count once, empty relevant sets remain undefined',()=>{
 assert.equal(evaluateCase({...EVALUATION_CASES.baseline[2],ranked:['a','a','x']},2).recall,.5);
 assert.equal(evaluateCase(EVALUATION_CASES.baseline[3],2).recall,null);
 assert.throws(()=>evaluateCase(EVALUATION_CASES.baseline[0],0));
});
