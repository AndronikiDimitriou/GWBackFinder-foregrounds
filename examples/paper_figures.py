"""Make the paper-2 figures with gwbf_foregrounds.

Run from anywhere after ``pip install -e ".[all]"``:

    python examples/paper_figures.py --summaries <folder with sbi_*_summary_with_errors.npz>
                                     --mcmc <folder with mcmc_logGmu_*.npz and dataset_logGmu_*.npz>
                                     --out figures/

Steps that need the trained posteriors download them on first use (or pass --posterior-dir
with posterior_full.pkl / posterior_gbexb.pkl / posterior_nofg.pkl). The expensive steps
(precision scans, coverage) are off by default: see the flags.
"""
import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")

from gwbf_foregrounds import FULL, data, mcmc, posterior, reconstruct, scan  # noqa: E402
from gwbf_foregrounds.plotting import precision as P  # noqa: E402
from gwbf_foregrounds.plotting.corner import plot_1d_posteriors, plot_mcmc_vs_sbi  # noqa: E402
from gwbf_foregrounds.plotting.spectra import plot_noise_and_pls, plot_reconstruction  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--summaries", default=".")
ap.add_argument("--mcmc", default=None)
ap.add_argument("--posterior-dir", default=None)
ap.add_argument("--out", default="figures")
ap.add_argument("--logGmu", type=float, default=-14.0, help="injection for reconstruction/corner")
ap.add_argument("--run-scans", action="store_true", help="recompute the SBI precision scans (slow)")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
S = lambda name: os.path.join(args.summaries, name)  # noqa: E731
O = lambda name: os.path.join(args.out, name)        # noqa: E731


def post_path(model):
    if args.posterior_dir:
        return os.path.join(args.posterior_dir, f"posterior_{model}.pkl")
    return model                                       # -> download from the release


# 1. precision scans (optional; otherwise the existing summary files are used)
if args.run_scans:
    for model, sign, name in (("full", 0, "sbi_fg"), ("full", -1, "sbi_lowfg"),
                              ("full", 1, "sbi_highfg"), ("nofg", 0, "sbi_nofg"),
                              ("gbexb", 0, "sbi_gbexb")):
        scan.run_sbi_scan(model, posterior=post_path(model), fg_sign=sign,
                          output_dir=args.summaries, summary_name=name)

# 2. precision bar plots
mcmc_pts = None
if args.mcmc:
    x, y = scan.mcmc_precision(sorted(glob.glob(os.path.join(args.mcmc, "mcmc_logGmu_*.npz"))))
    keep = (x >= -16) & (x <= -10)          # -9 is the prior edge: not comparable
    mcmc_pts = (x[keep], y[keep])
P.plot_fg_vs_nofg(S("sbi_fg_summary_with_errors.npz"), S("sbi_nofg_summary_with_errors.npz"),
                  out=O("sbi_precision_fg_vs_nofg.pdf"))
P.plot_with_validation(S("sbi_fg_summary_with_errors.npz"), S("sbi_nofg_summary_with_errors.npz"),
                       mcmc=mcmc_pts, fisher=S("fisher_95_intervals.npz"),
                       fisher_nofg=S("fisher_nofg_95_intervals.npz"),
                       out=O("sbi_precision_normal_with_mcmc_fisher.pdf"))
P.plot_low_normal_high(S("sbi_lowfg_summary_with_errors.npz"), S("sbi_fg_summary_with_errors.npz"),
                       S("sbi_highfg_summary_with_errors.npz"),
                       out=O("sbi_precision_low_normal_high.pdf"))
if os.path.exists(S("sbi_gbexb_summary_with_errors.npz")):
    P.plot_nofg_gbexb_full(S("sbi_nofg_summary_with_errors.npz"),
                           S("sbi_gbexb_summary_with_errors.npz"),
                           S("sbi_fg_summary_with_errors.npz"),
                           out=O("sbi_precision_nofg_gbexb_fullfg.pdf"))

# 3. noise + PLS
plot_noise_and_pls(out=O("lisa_noise_pls_AA_TT.pdf"))

# 4. reconstruction and MCMC vs SBI on the SAME dataset as the MCMC run
tag = data.tag_from_logGmu(args.logGmu)
ds_file = args.mcmc and os.path.join(args.mcmc, f"dataset_{tag}.npz")
ds = data.Dataset.load(ds_file) if ds_file and os.path.exists(ds_file) else \
    data.simulate(FULL, logGmu=args.logGmu, seed=42)
post = posterior.load_posterior(post_path("full"), FULL.param_names)
s_sbi = posterior.sample(post, ds, n=10_000, seed=0)
rec = reconstruct.reconstruct(s_sbi, FULL, ds.f_obs, truth=ds.params, max_samples=1000)
plot_reconstruction(rec, show_noise=True, ylim=(5e-14, 3e-10),
                    out=O(f"reconstruction_all_{tag}.pdf"))

chain_file = args.mcmc and os.path.join(args.mcmc, f"mcmc_{tag}.npz")
if chain_file and os.path.exists(chain_file):
    chain, names, truth = mcmc.load_chain(chain_file)
    plot_mcmc_vs_sbi(chain, s_sbi, names, truth, title=f"MCMC vs SBI | logGmu = {args.logGmu}",
                     out=O(f"corner_hdi_{tag}.pdf"))
    plot_mcmc_vs_sbi(chain, s_sbi, names, truth, keep=[n for n in names if n not in ("A_p", "A_acc")],
                     out=O(f"corner_hdi_{tag}_filtered.pdf"))
    plot_1d_posteriors(chain, s_sbi, names, truth, out=O(f"1d_posteriors_{tag}.pdf"))
print(f"figures in {args.out}/")
