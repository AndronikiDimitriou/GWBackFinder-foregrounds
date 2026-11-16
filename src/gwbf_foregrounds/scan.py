"""Precision on G mu versus the injected G mu, from SBI and from MCMC chains.

The summary files have the same keys as the ``sbi_*_summary_with_errors.npz``
files of the analysis scripts, so old and new results plot the same way.
"""
import os

import numpy as np

from . import data, posterior as post
from .models import get_model
from .stats import precision_from_samples

DEFAULT_GRID = np.arange(-17.0, -9.0 + 0.125, 0.25)


def run_sbi_scan(
    model,
    posterior=None,
    logGmu_values=DEFAULT_GRID,
    fg_sign=0,
    n_datasets=20,
    n_samples=10_000,
    output_dir=".",
    file_prefix=None,
    summary_name=None,
    verbose=True,
):
    """For each injected log10 G mu: simulate ``n_datasets`` noise realisations
    (seeds 0..n-1), sample the posterior, and record the 95% HDI precision.

    Saves one ``<file_prefix>_<tag>.npz`` per injection and
    ``<summary_name>_summary_with_errors.npz``. Returns the summary dict.
    """
    m = get_model(model)
    suffix = {-1: "low", 0: "", 1: "high"}[fg_sign]
    file_prefix = file_prefix or f"sbi_combined_{m.name}{('_' + suffix) if suffix else ''}"
    summary_name = summary_name or f"sbi_{m.name}{suffix}"
    os.makedirs(output_dir, exist_ok=True)
    posterior = post.load_posterior(posterior or m.name, m.param_names)

    rows = []
    for logGmu_inj in np.asarray(logGmu_values, float):
        per_seed, prec, lo, hi = [], [], [], []
        for seed in range(n_datasets):
            ds = data.simulate(m, logGmu=logGmu_inj, seed=seed, fg_sign=fg_sign)
            s = post.sample(posterior, ds, n=n_samples)[:, 0]
            p_i, lo_i, hi_i = precision_from_samples(s, logGmu_inj)
            per_seed.append(s)
            prec.append(p_i)
            lo.append(lo_i)
            hi.append(hi_i)
        combined = np.concatenate(per_seed)
        prec = np.asarray(prec)
        comb_prec, comb_lo, comb_hi = precision_from_samples(combined, logGmu_inj)
        row = dict(
            logGmu_inj=logGmu_inj,
            precision_mean=float(np.nanmean(prec)),
            precision_std=float(np.nanstd(prec, ddof=1)) if len(prec) > 1 else 0.0,
            precision_p16=float(np.nanpercentile(prec, 16)),
            precision_p50=float(np.nanpercentile(prec, 50)),
            precision_p84=float(np.nanpercentile(prec, 84)),
            combined_precision=comb_prec,
            lo_logGmu=comb_lo, hi_logGmu=comb_hi,
            lo_Gmu=10 ** comb_lo, hi_Gmu=10 ** comb_hi,
            median_logGmu=float(np.median(combined)),
            mean_logGmu=float(np.mean(combined)),
        )
        np.savez(
            os.path.join(output_dir, f"{file_prefix}_{data.tag_from_logGmu(logGmu_inj)}.npz"),
            logGmu_samples=combined,
            logGmu_samples_by_seed=np.asarray(per_seed),
            seed_precisions=prec,
            seed_lo_logGmu=np.asarray(lo),
            seed_hi_logGmu=np.asarray(hi),
            n_datasets=n_datasets,
            n_sbi_samples=n_samples,
            **row,
        )
        rows.append(row)
        if verbose:
            print(f"[{summary_name}] logGmu={logGmu_inj:6.2f}  precision = "
                  f"{row['precision_mean']:.3g} +- {row['precision_std']:.3g} %")

    summary = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    summary["Gmu_inj"] = 10 ** summary["logGmu_inj"]
    out = os.path.join(output_dir, f"{summary_name}_summary_with_errors.npz")
    np.savez(out, **summary)
    if verbose:
        print(f"Summary saved -> {out}")
    return summary


def load_summary(path):
    """Read a precision summary (.npz) into a plain dict of arrays."""
    with np.load(path) as d:
        return {k: d[k] for k in d.files}


def mcmc_precision(paths, burn_fraction=0.0):
    """Precision from MCMC chains (``mcmc_logGmu_*.npz`` from ``mcmc.run_mcmc`` or
    ``mcmc_paper2.py``). Returns (logGmu_inj array, precision % array), sorted."""
    pts = []
    for p in paths:
        with np.load(p, allow_pickle=True) as d:
            names = [str(n) for n in d["param_names"]]
            i = names.index("logGmu")
            s = d["samples"][:, i]
            s = s[int(burn_fraction * len(s)):]
            inj = float(d["true_params"][i])
        pts.append((inj, precision_from_samples(s, inj)[0]))
    pts.sort()
    return np.array([a for a, _ in pts]), np.array([b for _, b in pts])
