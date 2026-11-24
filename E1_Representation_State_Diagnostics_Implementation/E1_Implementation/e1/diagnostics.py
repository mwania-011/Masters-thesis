"""Deterministic, validation-only bilingual embedding diagnostics.

Input: paired rows of source and target embeddings; same row index = translation.
All cosine measurements use L2-normalized embeddings, no centering.
Covariance metrics center the normalized embeddings before eigendecomposition.
"""
import numpy as np


def normalize(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all() or x.shape[0] < 3:
        raise ValueError('Expected >=3 finite embedding rows in a 2D array')
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if (norms <= 1e-12).any():
        raise ValueError('Zero-norm embeddings are not permitted')
    return x / norms


def _validate_pair(a,b):
    a,b = normalize(a),normalize(b)
    if a.shape != b.shape: raise ValueError('Paired embeddings must have equal shape')
    return a,b


def _topk_neighbors(x, k, chunk=256):
    n = len(x)
    if not 1 <= k < n: raise ValueError('k must be in [1,n-1]')
    out = np.empty((n,k),dtype=np.int64)
    for start in range(0,n,chunk):
        stop=min(n,start+chunk)
        sims=x[start:stop]@x.T
        sims[np.arange(stop-start),np.arange(start,stop)]=-np.inf
        # stable ties by original index
        out[start:stop]=np.argsort(-sims,axis=1,kind='stable')[:,:k]
    return out


def alignment(a,b,chunk=256):
    a,b=_validate_pair(a,b)
    n=len(a)
    pos=np.sum(a*b,axis=1)
    top_correct=0
    negatives=[]
    for start in range(0,n,chunk):
        stop=min(n,start+chunk)
        sim=a[start:stop]@b.T
        sim[np.arange(stop-start),np.arange(start,stop)]=-np.inf
        negatives.extend(sim.max(axis=1).tolist())
        # top1 includes positives; ties prefer smallest index
        full=a[start:stop]@b.T
        top_correct+=np.sum(np.argmax(full,axis=1)==np.arange(start,stop))
    neg=np.asarray(negatives)
    return {'top1_source_to_target':float(top_correct/n),'positive_cosine':float(pos.mean()),
            'hard_negative_cosine':float(neg.mean()),'positive_hard_negative_margin':float(np.mean(pos-neg))}


def geometry(x, uniformity_pairs=20000, seed=0):
    x=normalize(x)
    n,d=x.shape
    mean_cos=float((np.linalg.norm(x.sum(axis=0))**2-n)/(n*(n-1)))
    covariance=np.cov(x,rowvar=False)
    covariance=np.atleast_2d(covariance)
    eigen=np.clip(np.linalg.eigvalsh(covariance),0,None)
    total=float(eigen.sum())
    if total <= 1e-15:
        eff_rank=0.0; concentration=1.0
    else:
        p=eigen/total
        nz=p[p>0]
        eff_rank=float(np.exp(-np.sum(nz*np.log(nz))))
        concentration=float(p.max())
    # unbiased Monte Carlo for distinct ordered pairs; fixed seed
    rng=np.random.default_rng(seed)
    i=rng.integers(0,n,size=uniformity_pairs)
    j=rng.integers(0,n-1,size=uniformity_pairs)
    j+=j>=i
    dist2=np.sum((x[i]-x[j])**2,axis=1)
    # Wang-Isola uniformity log E exp(-2 ||xi-xj||²)
    uniformity=float(np.log(np.mean(np.exp(-2*dist2))))
    return {'mean_pairwise_cosine':mean_cos,'anisotropy_mean_vector_norm':float(np.linalg.norm(x.mean(axis=0))),
            'covariance_spectral_concentration':concentration,'covariance_effective_rank':eff_rank,
            'uniformity_logexp_t2':uniformity,'n':n,'dimension':d}


def neighborhood(a,b,k=10,noise_std=0.01,seed=0,chunk=256):
    a,b=_validate_pair(a,b)
    na=_topk_neighbors(a,k,chunk); nb=_topk_neighbors(b,k,chunk)
    # index sets refer to aligned bilingual sentence IDs, not vector locations
    overlap=np.mean([len(set(u)&set(v))/k for u,v in zip(na,nb)])
    rng=np.random.default_rng(seed)
    a2=normalize(a+rng.normal(0,noise_std,a.shape))
    b2=normalize(b+rng.normal(0,noise_std,b.shape))
    na2=_topk_neighbors(a2,k,chunk); nb2=_topk_neighbors(b2,k,chunk)
    stability=np.mean([(len(set(u)&set(v))+len(set(w)&set(z)))/(2*k)
                       for u,v,w,z in zip(na,na2,nb,nb2)])
    # retained-neighbor order consistency; rank correlation on intersection omitted
    return {'cross_lingual_neighbor_overlap_at_k':float(overlap),
            'perturbation_neighbor_stability_at_k':float(stability),
            'neighborhood_sensitivity':float(1-stability),'k':k,'noise_std':noise_std}


def evaluate(a,b,k=10,seed=0):
    a,b=_validate_pair(a,b)
    return {'alignment':alignment(a,b), 'source_geometry':geometry(a,seed=seed),
            'target_geometry':geometry(b,seed=seed+1),
            'neighborhood':neighborhood(a,b,k=k,seed=seed)}
