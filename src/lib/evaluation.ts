/** Authored fixtures with deterministic metric computation; no inference claims. */
export type EvalCase = { id:string; question:string; reference:string; source:string; relevant:string[]; ranked:string[]; raw:string; judge:number; human:number };
export const EVALUATION_CASES: Record<string,EvalCase[]> = {
 baseline:[
  {id:'t1',question:'Participants in B?',reference:'18',source:'Study B enrolled 18 participants.',relevant:['b'],ranked:['x','y','b'],raw:'{"answer":"18","sources":["b"]}',judge:2,human:2},
  {id:'t2',question:'Weeks in B?',reference:'8',source:'Study B lasted 8 weeks.',relevant:['b'],ranked:['x','y','b'],raw:'{"answer":"6","sources":["b"]}',judge:2,human:0},
  {id:'t3',question:'Studies using consent?',reference:'A and B',source:'Study A and study B both obtained consent.',relevant:['a','b'],ranked:['a','x','b'],raw:'{"answer":"A","sources":["a"]}',judge:1,human:1},
  {id:'t4',question:'Funding amount?',reference:'UNKNOWN',source:'No funding amount is recorded.',relevant:[],ranked:['a','b'],raw:'{"answer":"1000","sources":[]}',judge:0,human:0},
 ],
 changed:[
  {id:'t1',question:'Participants in B?',reference:'18',source:'Study B enrolled 18 participants.',relevant:['b'],ranked:['b','x'],raw:'{"answer":"18","sources":["b"]}',judge:2,human:2},
  {id:'t2',question:'Weeks in B?',reference:'8',source:'Study B lasted 8 weeks.',relevant:['b'],ranked:['b','x'],raw:'8',judge:2,human:2},
  {id:'t3',question:'Studies using consent?',reference:'A and B',source:'Study A and study B both obtained consent.',relevant:['a','b'],ranked:['a','b','x'],raw:'{"answer":"A and B","sources":["a","b"]}',judge:2,human:2},
  {id:'t4',question:'Funding amount?',reference:'UNKNOWN',source:'No funding amount is recorded.',relevant:[],ranked:['a','b'],raw:'{"answer":"UNKNOWN","sources":[]}',judge:2,human:2},
 ],
};
export function evaluateCase(row:EvalCase,k:number){
 if(!Number.isInteger(k)||k<1)throw Error('k must be a positive integer');
 let answer=row.raw, schema=false;
 try {const obj=JSON.parse(row.raw);schema=!!obj && !Array.isArray(obj) && typeof obj==='object' && Object.keys(obj).sort().join(',')==='answer,sources' && typeof obj.answer==='string' && Array.isArray(obj.sources) && obj.sources.every((x:unknown)=>typeof x==='string');if(schema)answer=obj.answer;}catch{}
 const normalize=(text:string)=>text.toLowerCase().trim().replace(/\s+/g,' ');
 const relevant=new Set(row.relevant),top=new Set(row.ranked.slice(0,k));
 const recall=relevant.size?[...relevant].filter(id=>top.has(id)).length/relevant.size:null;
 return {answer,schema,exact:normalize(answer)===normalize(row.reference),recall};
}
export function summarizeEvaluation(rows:EvalCase[],k:number){
 const scores=rows.map(r=>evaluateCase(r,k));const recalls=scores.flatMap(r=>r.recall===null?[]:[r.recall]);
 return {n:rows.length,exact:rows.length?scores.filter(r=>r.exact).length/rows.length:null,schema:rows.length?scores.filter(r=>r.schema).length/rows.length:null,recall:recalls.length?recalls.reduce((a,b)=>a+b,0)/recalls.length:null,recallN:recalls.length};
}
