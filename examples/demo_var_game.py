#!/usr/bin/env python
"""
Minimal end-to-end demo:
- synthesize X (N x T)
- seasonal information-sharing game -> payoffs
- NMF strategies -> estimate A (ridge) -> ESS
"""
import argparse, numpy as np
import ts2eg as gm
from ts2eg import extensions as ext

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=5)
    ap.add_argument("--T", type=int, default=360)
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0, help="PRNG seed for reproducibility")
    args = ap.parse_args()


    import random
    random.seed(args.seed)

    rng = np.random.default_rng(args.seed)
    t = np.arange(args.T)
    season = 0.6*np.sin(2*np.pi*t/12)
    common = 0.3*np.sin(2*np.pi*t/60)
    contrasts = np.vstack([
        0.5*np.sin(2*np.pi*(t+3)/40),
        -0.4*np.cos(2*np.pi*(t+5)/36),
        0.3*np.sin(2*np.pi*(t+7)/24),
        -0.2*np.cos(2*np.pi*(t+11)/30),
        0.25*np.sin(2*np.pi*(t+13)/48),
    ])[:args.N]
    X = season + common + contrasts + 0.15*rng.standard_normal((args.N, args.T))

    _ = ext.var_information_sharing_game_seasonal(
        X, p=args.p, ridge=1e-3, seasonal_period=12,
        seasonal_multiples=[1,2], include_self_always=True
    )

    X0 = (X - X.min(axis=1, keepdims=True)) / (np.ptp(X,axis=1, keepdims=True) + 1e-9)
    S, H = gm.nmf_on_X(X0, k=args.k, iters=200, seed=args.seed, normalize="l2")

    X_std = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-12)
    A = ext.estimate_A_from_series_weighted(S, X_std, X_std, k=args.k, ridge=args.ridge)["A"]

    res = gm.find_ESS(A, tol=1e-8, max_support=args.k)
    print("Nash:", sum(r["is_nash"] for r in res), " ESS:", sum(r["is_ess"] for r in res))

if __name__ == "__main__":
    main()
