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


def table_ii(full, nofg, mcmc=None, fisher=None, grid=(-16, -15, -14, -13, -12, -11, -10)):
    """Paper Table II: precision % for SBI (with fg), MCMC, Fisher, and SBI without fg.

    mcmc   : (logGmu, precision) arrays (``scan.mcmc_precision``)
    fisher : path/dict with 'logGmu_inj', 'precision'
    Returns a list of row dicts (NaN where a method has no value).
    """
    full, nofg = _load(full), _load(nofg)
    fis = _load(fisher) if fisher is not None else None

    def at(x, y, g):
        x = np.asarray(x)
        i = np.where(np.isclose(x, g))[0]
        return float(np.asarray(y)[i[0]]) if len(i) else np.nan

    rows = []
    for g in grid:
        rows.append(dict(
            logGmu=g,
            sbi=at(full["logGmu_inj"], full["precision_mean"], g),
            mcmc=at(*mcmc, g) if mcmc is not None else np.nan,
            fisher=at(fis["logGmu_inj"], fis["precision"], g) if fis is not None else np.nan,
            sbi_nofg=at(nofg["logGmu_inj"], nofg["precision_mean"], g),
        ))
    return rows


def format_table_ii(rows, latex=False):
    def f(v):
        if not np.isfinite(v):
            return "--"
        return f"{v:.2e}" if v >= 1e4 else f"{v:.2f}"

    if latex:
        lines = [r"\begin{tabular}{c|ccc|c}", r"$G\mu_{\rm inj}$ & SBI & MCMC & Fisher & SBI (no fg) \\ \hline"]
        lines += [f"$10^{{{r['logGmu']}}}$ & {f(r['sbi'])} & {f(r['mcmc'])} & {f(r['fisher'])} & "
                  f"{f(r['sbi_nofg'])} \\\\" for r in rows]
        return "\n".join(lines + [r"\end{tabular}"])
    head = f"{'Gmu_inj':>8} | {'SBI':>9} {'MCMC':>9} {'Fisher':>10} | {'SBI no fg':>9}"
    return "\n".join([head, "-" * len(head)] + [
        f"{'1e' + str(r['logGmu']):>8} | {f(r['sbi']):>9} {f(r['mcmc']):>9} {f(r['fisher']):>10} | "
        f"{f(r['sbi_nofg']):>9}" for r in rows])
