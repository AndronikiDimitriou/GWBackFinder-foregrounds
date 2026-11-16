"""Calibration checks of the trained posteriors: per-parameter coverage, TARP and SBC."""
import numpy as np
import scipy.stats

from . import data
from .models import get_model


def sample_prior(model, n, seed=0):
    """Draw ``n`` parameter vectors from the model's training prior, shape (n, ndim)."""
    m = get_model(model)
    rng = np.random.default_rng(seed)
    out = np.empty((n, m.ndim))
    for i, name in enumerate(m.param_names):
        kind, a, b = m.prior[name]
        out[:, i] = rng.uniform(a, b, n) if kind == "uniform" else rng.normal(a, b, n)
    return out


def build_test_set(model, n_test=300, seed=0):
    """(thetas tensor, xs tensor) of ``n_test`` prior draws and their simulated data."""
    import torch

    thetas = sample_prior(model, n_test, seed=seed)
    xs = data.simulate_x(thetas, model, seed=seed)
    return torch.tensor(thetas, dtype=torch.float32), xs


# --------------------------------------------------------------------------- #
# Per-parameter coverage (HPD rank statistic via 1D KDE)
# --------------------------------------------------------------------------- #

def _gamma(marg, true_val, kde_points):
    from scipy.stats import gaussian_kde

    kde = gaussian_kde(marg, bw_method="silverman")
    yy = kde(np.linspace(marg.min(), marg.max(), kde_points))
    at_true = kde(float(true_val))[0]
    return yy[yy >= at_true].sum() / yy.sum()


def gamma_coverage_all(posterior, thetas, xs, param_names, n_post_samples=500, kde_points=5000,
                       verbose=False):
    """For each test case and parameter: posterior mass of the 1D HPD region that just
    reaches the true value. Uniform on [0, 1] for a calibrated posterior.

    The posterior is sampled once per test case and used for every parameter.
    Returns {name: array of length n_test}.
    """
    thetas = np.asarray(thetas.detach().cpu() if hasattr(thetas, "detach") else thetas)
    out = {name: np.empty(len(thetas)) for name in param_names}
    for i in range(len(thetas)):
        s = posterior.sample((n_post_samples,), x=xs[i:i + 1], show_progress_bars=False)
        s = s.detach().cpu().numpy()
        for j, name in enumerate(param_names):
            out[name][i] = _gamma(s[:, j], thetas[i, j], kde_points)
        if verbose and (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(thetas)} test cases done", flush=True)
    return out


def gamma_coverage(posterior, thetas, xs, param_idx, n_post_samples=500, kde_points=5000):
    """Coverage statistic for one parameter (see ``gamma_coverage_all``)."""
    names = [f"p{k}" for k in range(np.shape(thetas)[1])]
    return gamma_coverage_all(posterior, thetas, xs, names, n_post_samples, kde_points)[
        names[param_idx]]


_probit = scipy.stats.norm.ppf


def z_from_alpha(alpha):
    return _probit(1 - alpha / 2)


def alpha_from_z(z):
    return 2 - scipy.stats.norm.cdf(z) * 2


def jeffreys_interval(k, n, z=1):
    alpha = alpha_from_z(z)
    lo = scipy.stats.beta.ppf(alpha / 2, k + 0.5, n - k + 0.5)
    hi = scipy.stats.beta.ppf(1 - alpha / 2, k + 0.5, n - k + 0.5)
    return np.array([np.where(k > 0, lo, 0.0), np.where(k < n, hi, 1.0)]).T


def estimate_hat_z(masses, nbins=50, zmax=4, z_band=1):
    """Empirical coverage hat-z against nominal z, with a Jeffreys band."""
    n = len(masses)
    z = np.linspace(0, zmax, nbins)
    t = 1 - alpha_from_z(z)
    k = np.array([np.sum(masses < ti) for ti in t])
    band = jeffreys_interval(k, n, z=z_band)
    zb = z_from_alpha(1 - band)
    return dict(z=z, mean=z_from_alpha(1 - k / n), lower=zb[:, 0], upper=zb[:, 1])


# --------------------------------------------------------------------------- #
# TARP and SBC (thin wrappers around sbi)
# --------------------------------------------------------------------------- #

def run_tarp(posterior, thetas, xs, num_posterior_samples=1000):
    """(ecp, alpha, atc, ks_pval) from sbi's TARP test."""
    from sbi.diagnostics import check_tarp, run_tarp as _run_tarp

    ecp, alpha = _run_tarp(thetas, xs, posterior, references=None,
                           num_posterior_samples=num_posterior_samples,
                           use_batched_sampling=False)
    atc, ks = check_tarp(ecp, alpha)
    return np.asarray(ecp), np.asarray(alpha), float(atc), float(ks)


def run_sbc(posterior, thetas, xs, num_posterior_samples=1000):
    """(ranks, check_stats) from sbi's SBC."""
    from sbi.diagnostics import check_sbc, run_sbc as _run_sbc

    ranks, dap = _run_sbc(thetas, xs, posterior, num_posterior_samples=num_posterior_samples,
                          use_batched_sampling=False)
    stats = check_sbc(ranks, thetas, dap, num_posterior_samples=num_posterior_samples)
    return np.asarray(ranks), {k: np.asarray(v) for k, v in stats.items()}
