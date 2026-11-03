export type Box = [number,number,number,number]
export type Detection = {id:string;label:string;score:number;box:Box}
export type Reference = {id:string;label:string;box:Box;source:string}
export type DetectionImage = {id:string;image:string;width:number;height:number;outputs:Record<string,Detection[]>;references:Reference[]}
export type DetectionAssets = {description:string;annotation_note:string;models:Record<string,{weights:string;source:string;preprocessing:string;license:string}>;image_source:string;images:DetectionImage[]}
export function area(b:Box){return Math.max(0,b[2]-b[0])*Math.max(0,b[3]-b[1])}
export function iou(a:Box,b:Box){
 const intersection=Math.max(0,Math.min(a[2],b[2])-Math.max(a[0],b[0]))*Math.max(0,Math.min(a[3],b[3])-Math.max(a[1],b[1]))
 const union=area(a)+area(b)-intersection
 return union>0?intersection/union:0
}
export function filterDetections(candidates:Detection[],confidence:number,nms:number){
 const valid=candidates.filter(d=>Number.isFinite(d.score)&&d.box.every(Number.isFinite)&&area(d.box)>0)
 const sorted=valid.filter(d=>d.score>=confidence).sort((a,b)=>b.score-a.score||a.id.localeCompare(b.id))
 const kept:Detection[]=[]
 const suppressed:{detection:Detection;by:Detection;overlap:number}[]=[]
 for(const detection of sorted){
  const by=kept.find(k=>k.label===detection.label&&iou(k.box,detection.box)>nms)
  if(by)suppressed.push({detection,by,overlap:iou(by.box,detection.box)})
  else kept.push(detection)
 }
 return {kept,suppressed,belowConfidence:valid.length-sorted.length,aboveConfidence:sorted.length,invalid:candidates.length-valid.length}
}
export function matchReferences(kept:Detection[],references:Reference[],threshold=.5){
 const used=new Set<string>()
 return references.map(reference=>{
  const ranked=kept.filter(d=>d.label===reference.label&&!used.has(d.id)).map(d=>({detection:d,overlap:iou(d.box,reference.box)})).sort((a,b)=>b.overlap-a.overlap)
  const best=ranked[0]
  if(best&&best.overlap>=threshold){used.add(best.detection.id);return {reference,detection:best.detection,overlap:best.overlap,matched:true}}
  return {reference,detection:null,overlap:best?.overlap??0,matched:false}
 })
}
