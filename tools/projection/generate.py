"""Reproduce actual SVD and optimized neighborhood views; no authored coordinates."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
from umap import UMAP

data=load_digits()
ids=np.concatenate([np.flatnonzero(data.target==c)[:30] for c in range(10)])
x=data.data[ids].astype(float)/16
mean=x.mean(0); centered=x-mean
u,s,vt=np.linalg.svd(centered,full_matrices=False)
start=time.monotonic(); views={}; timings={}
for count in [5,15,30]:
    t=time.monotonic()
    tsne=TSNE(n_components=2,perplexity=count,random_state=42,max_iter=500,init='pca').fit_transform(x)
    umap=UMAP(n_components=2,n_neighbors=count,min_dist=0.1,random_state=42,n_jobs=1).fit_transform(x)
    views[str(count)]={'tsne':tsne.tolist(),'umap':umap.tolist()}
    timings[str(count)]=round(time.monotonic()-t,3)
    print(f'Computed actual nonlinear views for {count}: {timings[str(count)]}s',flush=True)
payload={'inputs':x.tolist(),'labels':data.target[ids].tolist(),'ids':ids.tolist(),'mean':mean.tolist(),'directions':vt.tolist(),'singular':s.tolist(),'views':views,'provenance':{'source':'sklearn.datasets.load_digits','source_url':'https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html','sample_rule':'first 30 of each digit class in original order','sha256_float64':hashlib.sha256(x.tobytes()).hexdigest(),'seed':42,'tsne_iterations':500,'umap_min_dist':0.1,'normalization':'intensity / 16','fit_seconds':timings,'total_seconds':round(time.monotonic()-start,3),'nonlinear_execution':'Precomputed real optimizers; browser recomputes PCA projection and reconstruction.'}}
out=Path(__file__).resolve().parents[2]/'src/lib/data/cv-projection.json'
out.write_text(json.dumps(payload,separators=(',',':'))+'\n')
print(out)
