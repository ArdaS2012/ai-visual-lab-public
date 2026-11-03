export type ProjectionAssets = {
 inputs: number[][]; labels: number[]; ids: number[]; mean: number[];
 directions: number[][]; singular: number[];
 views: Record<string, {tsne: number[][]; umap: number[][]}>;
 provenance: {source_url: string; seed: number; normalization: string; nonlinear_execution: string; [key:string]: unknown};
}
export function project(assets: ProjectionAssets, k: number) {
 if(!Number.isInteger(k)||k<1||k>assets.directions.length) throw new Error('Invalid retained component count')
 const basis=assets.directions.slice(0,k)
 const scores=assets.inputs.map(row=>basis.map(v=>row.reduce((sum,x,j)=>sum+(x-assets.mean[j])*v[j],0)))
 const reconstruction=scores.map(z=>assets.mean.map((mean,j)=>mean+z.reduce((sum,value,i)=>sum+value*basis[i][j],0)))
 const mse=assets.inputs.reduce((sum,row,i)=>sum+row.reduce((s,x,j)=>s+(x-reconstruction[i][j])**2,0),0)/(assets.inputs.length*assets.mean.length)
 const total=assets.singular.reduce((s,v)=>s+v*v,0)
 const retained=assets.singular.slice(0,k).reduce((s,v)=>s+v*v,0)/total
 return {scores,reconstruction,mse,retained}
}
export function squaredDistance(a: number[],b: number[]) {return a.reduce((sum,x,j)=>sum+(x-b[j])**2,0)}
