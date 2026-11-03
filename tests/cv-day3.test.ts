import {test} from 'node:test'
import assert from 'node:assert/strict'
import {regionOverlap,relativeScores,semanticRegion} from '../src/lib/cv-day3.ts'
test('overlap handles empty regions, symmetry and partial agreement',()=>{
 assert.deepEqual(regionOverlap([false],[false]),{intersection:0,union:0,areaA:0,areaB:0,iou:1,dice:1})
 assert.equal(regionOverlap([true],[false]).iou,0)
 const a=[true,true,false],b=[false,true,true];assert.equal(regionOverlap(a,b).iou,1/3);assert.equal(regionOverlap(a,b).dice,.5);assert.equal(regionOverlap(a,b).dice,regionOverlap(b,a).dice);assert.throws(()=>regionOverlap([true],[]))
})
test('candidate scoring keeps denominator and remains stable at large scale',()=>{
 const a=relativeScores([.6,.2],5);assert.ok(Math.abs(a[0]-.88079707798)<1e-10)
 const b=relativeScores([.6,.2,.6],5);assert.ok(b[0]<a[0]);assert.ok(Math.abs(b.reduce((x,y)=>x+y,0)-1)<1e-12)
 assert.deepEqual(relativeScores([1,1],10000),[.5,.5]);assert.throws(()=>relativeScores([1],0));assert.throws(()=>relativeScores([NaN],1))
})
test('semantic selection combines class identity with inclusive cutoff',()=>{
 assert.deepEqual(semanticRegion([[1,0,1]],[[[.2,.7,.1]],[[.8,.3,.9]]],1,.8),[true,false,true])
})
