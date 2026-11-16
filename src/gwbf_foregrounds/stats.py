"""Credible intervals and the precision on G mu used throughout the paper."""
import numpy as np

LN10 = np.log(10.0)


def hdi(samples, prob=0.95):
    """Highest-density interval of 1D samples (same as ``arviz.hdi`` for unimodal samples)."""
    x = np.sort(np.asarray(samples, float).ravel())
    n = len(x)
    k = int(np.floor(prob * n))
    if k < 1 or k >= n:
        return float(x[0]), float(x[-1])
    widths = x[k:] - x[: n - k]
    i = int(np.argmin(widths))
    return float(x[i]), float(x[i + k])


def precision(lo_logGmu, hi_logGmu, logGmu_inj):
    """Width of the interval in G mu, as a percentage of the injected G mu:

        100 * (10**hi - 10**lo) / 10**inj
    """
    return (np.exp(LN10 * (lo_logGmu - logGmu_inj))
            * (np.exp(LN10 * (hi_logGmu - lo_logGmu)) - 1.0) * 100.0)


def precision_from_samples(logGmu_samples, logGmu_inj, prob=0.95):
    """(precision %, lo, hi) from the 95% HDI of log10 G mu samples."""
    lo, hi = hdi(logGmu_samples, prob)
    return float(precision(lo, hi, logGmu_inj)), lo, hi


def sigma_to_precision(sigma_logGmu, logGmu_inj, z=1.96):
    """Fisher: Gaussian 95% interval logGmu_inj +- 1.96 sigma -> (precision %, lo, hi)."""
    lo = logGmu_inj - z * sigma_logGmu
    hi = logGmu_inj + z * sigma_logGmu
    return float(precision(lo, hi, logGmu_inj)), float(lo), float(hi)


def bands(stack, levels=(0.95,)):
    """Pointwise median and equal-tailed bands of an (n_samples, n_f) array.

    Returns {'median': ..., 0.95: (lo, hi), ...}.
    """
    stack = np.asarray(stack)
    out = {"median": np.percentile(stack, 50.0, axis=0)}
    for lev in levels:
        q = 100 * (1 - lev) / 2
        out[lev] = (np.percentile(stack, q, axis=0), np.percentile(stack, 100 - q, axis=0))
    return out


def crossing_logGmu(log_grid, prec_grid, threshold):
    """log10 G mu where the precision curve crosses ``threshold`` % (log-linear interpolation)."""
    order = np.argsort(log_grid)
    x = np.asarray(log_grid)[order]
    logy = np.log10(np.asarray(prec_grid)[order])
    logt = np.log10(threshold)
    for i in range(len(x) - 1):
        if (logy[i] - logt) * (logy[i + 1] - logt) <= 0 and logy[i] != logy[i + 1]:
            frac = (logt - logy[i]) / (logy[i + 1] - logy[i])
            return float(x[i] + frac * (x[i + 1] - x[i]))
    return None
