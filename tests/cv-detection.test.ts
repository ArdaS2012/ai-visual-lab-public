import test from 'node:test'
import assert from 'node:assert/strict'
import fs from 'node:fs'
import { iou, filterDetections, matchReferences, type Detection, type DetectionAssets } from '../src/lib/cv-detection.ts'
const data=JSON.parse(fs.readFileSync(new URL('../src/lib/data/cv-detection.json',import.meta.url),'utf8')) as DetectionAssets
test('continuous box IoU handles overlap, identity, disjoint and zero area',()=>{
 assert.equal(iou([0,0,4,4],[2,0,6,4]),1/3);assert.equal(iou([0,0,4,4],[0,0,4,4]),1);assert.equal(iou([0,0,1,1],[2,2,3,3]),0);assert.equal(iou([0,0,0,0],[0,0,0,0]),0)
})
test('confidence boundary is inclusive and NMS respects classes and overlap',()=>{
 const rows:Detection[]=[{id:'a',label:'person',score:.8,box:[0,0,4,4]},{id:'b',label:'person',score:.7,box:[0,0,4,4]},{id:'c',label:'bus',score:.7,box:[0,0,4,4]}]
 assert.deepEqual(filterDetections(rows,.7,.5).kept.map(d=>d.id),['a','c']);assert.equal(filterDetections(rows,.7,1).kept.length,3);assert.equal(filterDetections(rows,1,.5).kept.length,0)
 assert.equal(filterDetections([{...rows[0],box:[1,1,0,0]}],0,.5).invalid,1)
})
test('reference matches are class-aware and one-to-one',()=>{
 const d:Detection={id:'a',label:'person',score:.8,box:[0,0,4,4]};const refs=['r1','r2'].map(id=>({id,label:'person',box:[0,0,4,4] as [number,number,number,number],source:'authored test'}));
 assert.equal(matchReferences([d],refs).filter(r=>r.matched).length,1);assert.equal(matchReferences([{...d,label:'bus'}],refs)[0].matched,false)
})
test('real detector assets are finite and higher confidence cannot increase retained count at fixed NMS',()=>{
 assert.equal(data.images.length,2);assert.equal(Object.keys(data.models).length,2)
 for(const image of data.images)for(const rows of Object.values(image.outputs)){
  assert.ok(rows.length>0);assert.equal(new Set(rows.map(d=>d.id)).size,rows.length);assert.ok(rows.every(d=>Number.isFinite(d.score)&&d.score>=0&&d.score<=1&&d.box.every(Number.isFinite)))
  const counts=[0,.25,.5,.75,1].map(c=>filterDetections(rows,c,.5).kept.length);for(let i=1;i<counts.length;i++)assert.ok(counts[i]<=counts[i-1]);assert.equal(counts.at(-1),0)
 }
})
