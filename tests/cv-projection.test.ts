import test from 'node:test'
import assert from 'node:assert/strict'
import fs from 'node:fs'
import { project, type ProjectionAssets } from '../src/lib/cv-projection.ts'
const data=JSON.parse(fs.readFileSync(new URL('../src/lib/data/cv-projection.json',import.meta.url),'utf8')) as ProjectionAssets

test('full rank reconstructs every original image; rank errors decrease',()=>{
 const results=[1,8,32,64].map(k=>project(data,k))
 assert.ok(results[3].mse<1e-20)
 for(let i=1;i<results.length;i++){assert.ok(results[i].mse<=results[i-1].mse);assert.ok(results[i].retained>=results[i-1].retained)}
 assert.ok(Math.abs(results[3].retained-1)<1e-12)
})
test('all optimized views preserve fixed row identity and finite coordinates',()=>{
 assert.equal(new Set(data.ids).size,300)
 for(const view of Object.values(data.views))for(const coords of Object.values(view)){
  assert.equal(coords.length,300);assert.ok(coords.every(p=>p.length===2&&p.every(Number.isFinite)))
 }
 assert.throws(()=>project(data,0)); assert.throws(()=>project(data,65))
})
