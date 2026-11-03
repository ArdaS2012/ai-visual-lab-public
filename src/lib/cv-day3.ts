export type Grid=number[][]
export function regionOverlap(a:boolean[],b:boolean[]){
 if(a.length!==b.length)throw Error('Aligned masks must have equal size')
 let intersection=0,union=0,areaA=0,areaB=0
 for(let i=0;i<a.length;i++){if(a[i])areaA++;if(b[i])areaB++;if(a[i]&&b[i])intersection++;if(a[i]||b[i])union++}
 return {intersection,union,areaA,areaB,iou:union?intersection/union:1,dice:areaA+areaB?2*intersection/(areaA+areaB):1}
}
export function relativeScores(cosines:number[],scale:number){
 if(!cosines.length||!cosines.every(Number.isFinite)||!Number.isFinite(scale)||scale<=0)throw Error('Finite cosines and positive scale required')
 const top=Math.max(...cosines),weights=cosines.map(s=>Math.exp((s-top)*scale)),total=weights.reduce((a,b)=>a+b,0)
 return weights.map(w=>w/total)
}
export function semanticRegion(labels:Grid,probabilities:number[][][],classId:number,threshold:number){
 return labels.flatMap((row,y)=>row.map((v,x)=>v===classId&&probabilities[classId][y][x]>=threshold))
}
