#!/usr/bin/env python
"""
Minimal legacy payoff-field demonstration:
- synthesize X (N x T)
- construct a seasonal profile-payoff map as a separate descriptive example
- fit an NMF reconstruction basis on nonnegative X0
- re-infer mixtures on the same X0 scale
- regress an explicitly stipulated payoff field on those mixtures

Nash and ESS-certificate results describe the fitted operator only. They are
not hypothesis tests, motion recovery, or evidence about the data-generating
process.  This legacy pipeline has no validated detection capability.
The replacement motion estimator specified in the v2 design is not implemented
in this demonstration.
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

    sharing_game = ext.var_information_sharing_game_seasonal(
        X, p=args.p, ridge=1e-3, seasonal_period=12,
        seasonal_multiples=[1,2], include_self_always=True
    )
    print("Descriptive seasonal sharing profiles:", sharing_game.meta["profiles"])

    X0 = (X - X.min(axis=1, keepdims=True)) / (np.ptp(X,axis=1, keepdims=True) + 1e-9)
    S, _H = gm.nmf_on_X(X0, k=args.k, iters=200, seed=args.seed, normalize="l2")

    # This stipulated field is only a smoke-test input.  It is not produced by
    # the profile-payoff map above and has no causal interpretation.
    X_std = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-12)
    stipulated_payoffs = X_std
    est = ext.estimate_A_from_series_weighted(
        S, X0, stipulated_payoffs, k=args.k, ridge=args.ridge
    )
    A = est["A"]

    res = gm.find_ESS(A, tol=1e-8, max_support=args.k)
    print(
        "Fitted-operator Nash candidates:", sum(r["is_nash"] for r in res),
        " fitted-operator ESS certificates:",
        sum(r["ess_status"] == "certified" for r in res),
        " in-sample zero-field R2:", round(est["R2"], 3),
        " (no validated data-level detection)",
    )

if __name__ == "__main__":
    main()
