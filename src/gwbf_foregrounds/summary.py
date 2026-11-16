"""Numbers for the text: precision at chosen G mu, threshold crossings, method comparison."""
import numpy as np

from .scan import load_summary
from .stats import crossing_logGmu


def _load(s):
    return load_summary(s) if isinstance(s, str) or hasattr(s, "__fspath__") else s


def precision_table(summaries, points=(-17, -15, -12, -11, -10, -9)):
    """{name: summary} -> rows of (logGmu, {name: precision %}) at the nearest grid point."""
    summaries = {k: _load(v) for k, v in summaries.items()}
    rows = []
    for t in points:
        row = {}
        for name, s in summaries.items():
            i = int(np.argmin(np.abs(s["logGmu_inj"] - t)))
            row[name] = float(s["precision_mean"][i])
        rows.append((t, row))
    return rows


def crossings(summaries, thresholds=(100, 10)):
    """{name: summary} -> {(name, threshold): log10 G mu where precision = threshold %}."""
    out = {}
    for name, s in summaries.items():
        s = _load(s)
        for thr in thresholds:
            out[(name, thr)] = crossing_logGmu(s["logGmu_inj"], s["precision_mean"], thr)
    return out


def compare_methods(values, grid=None):
    """Pairwise agreement between precision curves on a common grid.

    values : {name: (logGmu array, precision array)}, e.g. SBI, MCMC, Fisher
    Returns {(a, b): {'msd', 'rms', 'msd_log10'}} using the points common to all.
    """
    if grid is None:
        sets = [set(np.round(x, 6)) for x, _ in values.values()]
        grid = np.array(sorted(set.intersection(*sets)))
    p = {}
    for name, (x, y) in values.items():
        x = np.round(np.asarray(x), 6)
        p[name] = np.array([y[np.where(x == g)[0][0]] for g in grid])
    names = list(p)
    out = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            d2 = (p[a] - p[b]) ** 2
            out[(a, b)] = dict(msd=float(d2.mean()), rms=float(np.sqrt(d2.mean())),
                               msd_log10=float(np.mean((np.log10(p[a]) - np.log10(p[b])) ** 2)))
    return grid, p, out


def print_report(summaries, points=(-17, -15, -12, -11, -10, -9)):
    print("Precision (%) at selected log10 G mu")
    names = list(summaries)
    print(f"{'logGmu':>8} " + " ".join(f"{n:>12}" for n in names))
    for t, row in precision_table(summaries, points):
        print(f"{t:8.1f} " + " ".join(f"{row[n]:12.4g}" for n in names))
    print("\nlog10 G mu where the precision crosses 100% and 10%")
    for (name, thr), x in crossings(summaries).items():
        print(f"  {name:12s} {thr:4d}%  " + (f"{x:.3f}  (G mu = {10 ** x:.3e})" if x is not None
                                             else "never in range"))
