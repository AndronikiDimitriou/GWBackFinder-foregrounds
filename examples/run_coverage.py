"""Recompute the per-parameter coverage (P-P plots, paper Figs. 6 and 7) for both networks.

    python examples/run_coverage.py --n-test 500 --out results/

Saves results/coverage_gamma_<model>.npz with the coverage statistic of every parameter
and the test parameters. Takes ~10-30 min per model on a CPU.
"""
import argparse
import os
import time
import warnings

import numpy as np

from gwbf_foregrounds import coverage, get_model, posterior

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", default=["nofg", "full"])
ap.add_argument("--n-test", type=int, default=500)
ap.add_argument("--n-post", type=int, default=500)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--posterior-dir", default=None)
ap.add_argument("--out", default="results")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

for name in args.models:
    m = get_model(name)
    t0 = time.time()
    src = os.path.join(args.posterior_dir, m.posterior_file) if args.posterior_dir else m.name
    post = posterior.load_posterior(src, m.param_names)
    print(f"[{name}] simulating {args.n_test} test datasets from the prior...", flush=True)
    thetas, xs = coverage.build_test_set(m, n_test=args.n_test, seed=args.seed)
    print(f"[{name}] done in {time.time() - t0:.0f}s; computing coverage...", flush=True)
    g = coverage.gamma_coverage_all(post, thetas, xs, m.param_names, n_post_samples=args.n_post,
                                    verbose=True)
    out = os.path.join(args.out, f"coverage_gamma_{name}.npz")
    np.savez(out, thetas=thetas.numpy(), param_names=np.array(m.param_names),
             n_post_samples=args.n_post, seed=args.seed, **g)
    print(f"[{name}] saved {out} ({time.time() - t0:.0f}s)", flush=True)
