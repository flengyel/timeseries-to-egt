#!/usr/bin/env python
from __future__ import annotations
from pathlib import Path
import nbformat as nbf, re

NB = Path("notebooks/FIRM_DEMO.ipynb")

NEW_CELL = """from pathlib import Path
import os, pandas as pd, numpy as np
from ts2eg.core import nmf_on_X, value_gradient_payoffs, estimate_A_from_series, find_ESS

path = Path("artifacts/data/firm_quarterly.csv")
if not path.exists():
    path.parent.mkdir(parents=True, exist_ok=True)
    tmpl = pd.DataFrame({
        "date": pd.period_range("2010Q1", periods=12, freq="Q").astype(str),
        "LaborComp":  [45,46,47,48,47,46,45,44,45,46,46,47],
        "CapEx":      [20,21,22,21,22,23,22,21,20,20,21,22],
        "RnD":        [8,8,9,9,9,9,10,10,9,9,9,9],
        "SGA":        [17,16,16,16,16,16,16,17,17,17,17,17],
        "Payouts":    [10,9,8,8,8,8,7,8,9,8,7,5],
        "VA":         [100,102,104,103,105,106,108,109,110,112,113,115],
        "W":          [58,59,60,61,60,59,58,57,58,59,59,60],
        "P":          [42,43,44,42,45,47,50,52,52,53,54,55],
        "ROIC":       [0.08,0.081,0.082,0.081,0.083,0.084,0.085,0.084,0.083,0.082,0.082,0.083],
    })
    tmpl.to_csv(path, index=False)
    print("Template written:", str(path))

df = pd.read_csv(path)
cols = ["LaborComp","CapEx","RnD","SGA","Payouts"]
X_real = df[cols].to_numpy(dtype=float).T
X_real = np.clip(X_real, 1e-9, None)
X_real = X_real / (X_real.sum(axis=0, keepdims=True) + 1e-12)
VA = df["VA"].to_numpy(float); W = df["W"].to_numpy(float); P = df["P"].to_numpy(float)
ROIC = df.get("ROIC", pd.Series([np.nan]*len(df))).to_numpy(float)

# Example payoff: value-gradient to future ROIC (H=4 quarters lead)
H = 4
J_future = np.roll(ROIC, -H)
v_real = value_gradient_payoffs(X_real, J_future, ridge=1e-2, standardize=True)

# Strategies & operator
S_r, H_r = nmf_on_X(X_real, k=3, iters=500, seed=2, normalize='l2')
est_r = estimate_A_from_series(S_r, X_real, v_real, k=3, ridge=1e-2)
A_r = est_r["A"]
ess_r = [r for r in find_ESS(A_r, tol=1e-8, max_support=3) if r["is_ess"]]
print("Real-data stub: R2=", round(est_r["R2"],3), "  ESS count=", len(ess_r))
"""

def main():
    nb = nbf.read(NB, as_version=4)
    done = False
    for c in nb.cells:
        if c.get("cell_type") != "code":
            continue
        src = c.get("source","") or ""
        if "firm_quarterly.csv" in src:
            c["source"] = NEW_CELL
            done = True
            break
    if not done:
        print("[warn] target cell not found; no changes")
        return
    nbf.write(nb, NB)
    print("[patched] FIRM_DEMO: data cell now writes under artifacts/data and always reads CSV")

if __name__ == "__main__":
    main()
