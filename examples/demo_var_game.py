# Create examples/demo_var_game.py, patch README to remove shims remark,
# and add a helper script to execute notebooks one-by-one.
import os, re, json, textwrap
from pathlib import Path

BASE = Path("/mnt/data")
EXDIR = BASE / "examples"
EXDIR.mkdir(exist_ok=True, parents=True)

demo_code = r'''#!/usr/bin/env python
"""
examples/demo_var_game.py

Minimal end-to-end demo:
- synthesize X (N x T)
- seasonal information-sharing game
- learn strategies (NMF)
- fit replicator operator A
- find ESS and (optionally) surrogate significance
"""
import argparse
import numpy as np

import ts2eg as gm
from ts2eg import extensions as ext

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=5, help="players")
    ap.add_argument("--T", type=int, default=360, help="time steps")
    ap.add_argument("--p", type=int, default=2, help="VAR lags")
    ap.add_argument("--k", type=int, default=3, help="strategies")
    ap.add_argument("--ridge", type=float, default=1e-3, help="ridge for VAR/regression")
    ap.add_argument("--seasonal-period", type=int, default=12, help="seasonal period")
    ap.add_argument("--seasonal-multiples", type=int, nargs="*", default=[1, 2], help="seasonal harmonics")
    ap.add_argument("--surrogates", type=int, default=0, help="num IAAFT surrogates for ESS rate (0 = skip)")
    ap.add_argument("--iaaft-iters", type=int, default=100, help="IAAFT iterations per surrogate")
    ap.add_argument("--seed", type=int, default=1, help="RNG seed")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # X: N x T, centered-ish synthetic series
    X = rng.standard_normal((args.N, args.T))

    # Weighted seasonal VAR info-sharing game
    w = np.ones(args.N)
    res = ext.var_information_sharing_game_seasonal(
        X, p=args.p, ridge=args.ridge,
        seasonal_period=args.seasonal_period,
        seasonal_multiples=args.seasonal_multiples,
        include_self_always=True, lambda_common=0.0, weights=w
    )

    # Strategy basis via NMF
    X0 = (X - X.min(axis=1, keepdims=True)) / (X.ptp(axis=1, keepdims=True) + 1e-9)
    S, H = gm.nmf_on_X(X0, k=args.k, iters=200, seed=args.seed, normalize="l2")

    # Fit replicator operator A (on standardized series)
    Xz = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-12)
    A = ext.estimate_A_from_series_weighted(S, Xz, Xz, k=args.k, lambda_=args.ridge, weights=w)["A"]

    # ESS search
    ess = [e for e in gm.find_ESS(A, tol=1e-8, max_support=args.k) if e["is_ess"]]

    print("=== Demo summary ===")
    print(f"N={args.N}, T={args.T}, p={args.p}, k={args.k}, ridge={args.ridge}")
    print("Profiles (seasonal):", res.meta.get("profiles"))
    print("A shape:", A.shape)
    print("ESS supports:", [e["support"] for e in ess])

    # Optional surrogate significance
    if args.surrogates > 0:
        sig = ext.surrogate_ess_frequency(
            S, Xz, Xz, k=args.k, lambda_=args.ridge,
            num_surrogates=args.surrogates, iaaft_iters=args.iaaft_iters,
            weights=w, seed=args.seed
        )
        print("Surrogate ESS rate:", sig["ess_rate"])

if __name__ == "__main__":
    main()
'''
(EXDIR / "demo_var_game.py").write_text(demo_code)
os.chmod(EXDIR / "demo_var_game.py", 0o755)

# Remove shims remark from README.md
readme_path = BASE / "README.md"
if readme_path.exists():
    txt = readme_path.read_text()
    # Remove any line that mentions "shims" or "Deprecated shims"
    lines = [ln for ln in txt.splitlines() if "shim" not in ln.lower() and "deprecated shims" not in ln.lower()]
    new_txt = "\n".join(lines)
    readme_path.write_text(new_txt)

# Add a simple notebook execution helper (optional)
SCRIPTS = BASE / "scripts"
SCRIPTS.mkdir(exist_ok=True)

nb_runner = r'''#!/usr/bin/env python
"""
scripts/execute_notebooks.py
Execute each .ipynb in notebooks/ with nbclient. Fails on error.
Usage:
  python scripts/execute_notebooks.py --timeout 600
"""
import argparse, sys, os, glob, time
from pathlib import Path

def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()

    try:
        import nbformat
        from nbclient import NotebookClient, CellExecutionError
    except Exception as e:
        print("Missing deps: pip install nbclient nbformat", file=sys.stderr)
        sys.exit(2)

    nb_dir = Path("notebooks")
    if not nb_dir.exists():
        print("No notebooks/ directory; nothing to run.")
        return 0

    notebooks = sorted(glob.glob(str(nb_dir / "*.ipynb")))
    if not notebooks:
        print("No notebooks found.")
        return 0

    rc_total = 0
    out_dir = Path("artifacts/notebooks")
    out_dir.mkdir(parents=True, exist_ok=True)

    for nb_path in notebooks:
        print(f"\n=== Executing {nb_path} ===")
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        client = NotebookClient(nb, timeout=args.timeout, kernel_name="python3")
        try:
            client.execute()
        except CellExecutionError as e:
            print(f"ERROR in {nb_path}:\n{e}", file=sys.stderr)
            rc_total = 1
        finally:
            out_file = out_dir / (Path(nb_path).stem + "_executed.ipynb")
            with open(out_file, "w", encoding="utf-8") as f:
                nbformat.write(nb, f)
            print(f"Wrote {out_file}")

    return rc_total

if __name__ == "__main__":
    sys.exit(run())
'''
(SCRIPTS / "execute_notebooks.py").write_text(nb_runner)
os.chmod(SCRIPTS / "execute_notebooks.py", 0o755)

print("Created examples/demo_var_game.py, patched README.md, and added scripts/execute_notebooks.py")
