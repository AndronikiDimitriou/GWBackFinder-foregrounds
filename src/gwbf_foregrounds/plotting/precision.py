"""Precision on G mu versus injected G mu: the bar plots of the paper.

Every function takes summaries as ``.npz`` paths or dicts (``scan.load_summary``),
returns the matplotlib figure, and saves it if ``out`` is given.
"""
import numpy as np
import matplotlib.pyplot as plt

from ..scan import load_summary

BLUE, BLUE_EDGE = "#5B9BD5", "#2E5A8A"
ORANGE, ORANGE_EDGE = "#FFA500", "#B86E00"
GREEN, GREEN_EDGE = "#5BB473", "#3E7536"


def _load(s):
    return load_summary(s) if isinstance(s, (str, bytes)) or hasattr(s, "__fspath__") else s


def _select(summary, lo=-17, hi=-10, integers_only=False):
    x = summary["logGmu_inj"]
    mask = (x >= lo - 1e-9) & (x <= hi + 1e-9)
    if integers_only:
        mask &= np.isclose(x, np.round(x))
    return mask


def _axis_style(ax, log_centers, ylim, fontsize=20, legend_fontsize=18):
    ax.set_yscale("log")
    ax.set_xlabel(r"$G\mu_{\rm inj}$", fontsize=fontsize)
    ax.set_ylabel(r"$\delta G\mu~\%$", fontsize=fontsize)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_ylim(*ylim)
    ticks = np.arange(np.ceil(log_centers.min()), np.floor(log_centers.max()) + 1)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"$10^{{{int(v)}}}$" for v in ticks], fontsize=fontsize)
    ax.tick_params(axis="both", which="major", labelsize=fontsize, length=8, width=1.5)
    ax.tick_params(axis="y", which="minor", length=4, width=1.0)
    ax.legend(fontsize=legend_fontsize, loc="upper right")


def _arrows(ax, xs, color="black"):
    """Arrows above the bars that leave the plot (precision above the y range)."""
    for x in xs:
        ax.annotate("", xy=(x, 1.08), xytext=(x, 0.90),
                    xycoords=ax.get_xaxis_transform(), textcoords=ax.get_xaxis_transform(),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2, mutation_scale=14),
                    annotation_clip=False, clip_on=False)


def _off_scale(log_centers, values, ymax):
    return log_centers[values > ymax]


def _save(fig, out):
    if out:
        fig.savefig(out, dpi=150, bbox_inches="tight")


def plot_fg_vs_nofg(fg, nofg, out=None, logGmu_range=(-17, -10), ylim=(1e0, 1e3),
                    errorbar_lw=3, errorbar_cap=4, errorbar_cthk=2):
    """Blue = no foregrounds; orange stacked on top = extra width from the foregrounds."""
    fg, nofg = _load(fg), _load(nofg)
    m = _select(fg, *logGmu_range)
    x = fg["logGmu_inj"][m]
    p_n, e_n = nofg["precision_mean"][m], nofg["precision_std"][m]
    p_f, e_f = fg["precision_mean"][m], fg["precision_std"][m]
    dx = abs(x[1] - x[0])
    ekw = dict(elinewidth=errorbar_lw, capthick=errorbar_cthk)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x, np.clip(p_f - p_n, 0, None), yerr=np.sqrt(e_f ** 2 + e_n ** 2), bottom=p_n,
           capsize=errorbar_cap, error_kw=ekw, width=0.52 * dx, color=ORANGE,
           edgecolor=ORANGE_EDGE, linewidth=1.2, label="with foregrounds")
    ax.bar(x, p_n, yerr=e_n, capsize=errorbar_cap, error_kw=ekw, width=0.8 * dx, color=BLUE,
           edgecolor=BLUE_EDGE, linewidth=0.8, label="no foregrounds")
    _axis_style(ax, x, ylim)
    ax.spines[["top", "right"]].set_visible(False)
    _arrows(ax, _off_scale(x, p_f, ylim[1]))
    fig.tight_layout()
    _save(fig, out)
    return fig


def plot_with_validation(fg, nofg, mcmc=None, fisher=None, fisher_nofg=None, out=None,
                         ylim=(3e-1, 3e4), errorbar_lw=3, errorbar_cap=4, errorbar_cthk=2):
    """Stacked SBI bars (no-fg + fg excess) with MCMC dots and Fisher markers.

    mcmc        : (logGmu array, precision array), e.g. from ``scan.mcmc_precision``
    fisher      : path/dict with 'logGmu_inj' and 'precision' (``fisher.run_fisher``)
    fisher_nofg : same, for the no-foreground model
    """
    fg, nofg = _load(fg), _load(nofg)
    x = fg["logGmu_inj"]
    p_n, e_n = nofg["precision_mean"], nofg["precision_std"]
    p_f, e_f = fg["precision_mean"], fg["precision_std"]
    w = 0.6 * abs(x[1] - x[0])
    ekw = dict(elinewidth=errorbar_lw, capthick=errorbar_cthk)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x, p_n, yerr=e_n, capsize=errorbar_cap, error_kw=ekw, width=w, color=BLUE,
           edgecolor=BLUE_EDGE, linewidth=0.8, label="No foregrounds")
    ax.bar(x, np.clip(p_f - p_n, 0, None), yerr=np.sqrt(e_f ** 2 + e_n ** 2), bottom=p_n,
           capsize=errorbar_cap, error_kw=ekw, width=w, color=ORANGE, edgecolor=ORANGE_EDGE,
           linewidth=1.2, label="Astrophysical foregrounds")
    if mcmc is not None:
        ax.scatter(*mcmc, marker="o", s=200, color="red", edgecolor="white", linewidth=1,
                   label="MCMC", zorder=6)
    if fisher is not None:
        f = _load(fisher)
        ax.scatter(f["logGmu_inj"], f["precision"], marker="*", s=190, color="#149331",
                   edgecolor="black", linewidth=0.6, label="Fisher", zorder=5)
    if fisher_nofg is not None:
        f = _load(fisher_nofg)
        ax.scatter(f["logGmu_inj"], f["precision"], marker="^", s=170, color="#0A5C36",
                   edgecolor="black", linewidth=0.6, label="Fisher (no fg)", zorder=5)
    _axis_style(ax, x, ylim)
    fig.tight_layout()
    _save(fig, out)
    return fig


def _arrow_positions(x, values, ymax, arrows):
    """arrows=None: bars above the y range; arrows=n: the first n bars."""
    return x[:arrows] if isinstance(arrows, int) else _off_scale(x, values, ymax)


def plot_low_normal_high(low, normal, high, out=None, logGmu_range=(-17, -10),
                         ylim=(4e0, 1e3), reference_lines=(10, 100), ax=None, arrows=None,
                         colors=("#555555", ORANGE, "#B5B5B5"),
                         edges=("#222222", ORANGE_EDGE, "#6E6E6E"), arrow_color="#DD3327",
                         fontsize=20, legend_fontsize=18):
    """Fiducial +1 sigma / fiducial / -1 sigma foregrounds, stacked
    (paper Fig. 3, lower panel: dark grey / orange / light grey)."""
    low, normal, high = _load(low), _load(normal), _load(high)
    m = _select(normal, *logGmu_range)
    x = normal["logGmu_inj"][m]
    p_l, p_m, p_h = (s["precision_mean"][m] for s in (low, normal, high))
    dx = abs(x[1] - x[0])
    d_lm = np.clip(p_m - p_l, 0, None)

    fig, ax = (plt.subplots(figsize=(13, 5)) if ax is None else (ax.figure, ax))
    ax.bar(x, np.clip(p_h - p_m, 0, None), bottom=p_l + d_lm, width=0.6 * dx, color=colors[0],
           edgecolor=edges[0], linewidth=1.0,
           label=r"with foregrounds $(\mathrm{fiducial} +1\sigma)$")
    ax.bar(x, d_lm, bottom=p_l, width=0.6 * dx, color=colors[1], edgecolor=edges[1],
           linewidth=1.0, label=r"with foregrounds $(\mathrm{fiducial})$")
    ax.bar(x, p_l, width=0.6 * dx, color=colors[2], edgecolor=edges[2], linewidth=1.0,
           label=r"with foregrounds $(\mathrm{fiducial} -1\sigma)$")
    for y in reference_lines:
        ax.axhline(y, color="black", linestyle="--", linewidth=1.5, alpha=0.6)
    _axis_style(ax, x, ylim, fontsize=fontsize, legend_fontsize=legend_fontsize)
    ax.spines[["top", "right"]].set_visible(False)
    _arrows(ax, _arrow_positions(x, p_h, ylim[1], arrows), color=arrow_color)
    if out:
        fig.tight_layout()
        _save(fig, out)
    return fig


def plot_nofg_gbexb_full(nofg, gbexb, full, out=None, logGmu_range=(-17, -10),
                         ylim=(1e0, 1e3), reference_lines=(10, 100), ax=None, arrows=None,
                         fontsize=20, legend_fontsize=18, errorbar_lw=1.2, errorbar_cap=2,
                         errorbar_cthk=1.2, arrow_color="#DD3327"):
    """No foregrounds (blue) < WD + SOBHB only (green) < all foregrounds (orange)
    (paper Fig. 3, upper panel). Error bars: std over the 20 noise realisations."""
    nofg, gbexb, full = _load(nofg), _load(gbexb), _load(full)
    m = _select(gbexb, *logGmu_range)
    x = gbexb["logGmu_inj"][m]
    p_n, e_n = nofg["precision_mean"][m], nofg["precision_std"][m]
    p_g, e_g = gbexb["precision_mean"][m], gbexb["precision_std"][m]
    p_f, e_f = full["precision_mean"][m], full["precision_std"][m]
    dx = abs(x[1] - x[0])
    ekw = dict(elinewidth=errorbar_lw, capthick=errorbar_cthk, zorder=5)

    fig, ax = (plt.subplots(figsize=(13, 5)) if ax is None else (ax.figure, ax))
    w = 0.6 * dx
    ax.bar(x, np.clip(p_f - p_g, 0, None), yerr=e_f, bottom=p_g, capsize=errorbar_cap,
           error_kw=ekw, width=w, color=ORANGE, edgecolor=ORANGE_EDGE, linewidth=1.0,
           label="with all foregrounds", zorder=3)
    ax.bar(x, np.clip(p_g - p_n, 0, None), yerr=e_g, bottom=p_n, capsize=errorbar_cap,
           error_kw=ekw, width=w, color=GREEN, edgecolor=GREEN_EDGE, linewidth=1.0,
           label="with WD + SOBHB foregrounds", zorder=3)
    ax.bar(x, p_n, yerr=e_n, capsize=errorbar_cap, error_kw=ekw, width=w, color=BLUE,
           edgecolor=BLUE_EDGE, linewidth=1.0, label="no foregrounds", zorder=3)
    for y in reference_lines:
        ax.axhline(y, color="black", linestyle="--", linewidth=1.5, alpha=0.6, zorder=2)
    _axis_style(ax, x, ylim, fontsize=fontsize, legend_fontsize=legend_fontsize)
    ax.spines[["top", "right"]].set_visible(False)
    _arrows(ax, _arrow_positions(x, p_f, ylim[1], arrows), color=arrow_color)
    if out:
        fig.tight_layout()
        _save(fig, out)
    return fig


def plot_paper_fig3(nofg, gbexb, full, low, high, out=None, arrows=4):
    """Paper Fig. 3: upper = foreground budgets, lower = foreground amplitudes +-1 sigma."""
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 10))
    plot_nofg_gbexb_full(nofg, gbexb, full, ax=a1, arrows=arrows)
    plot_low_normal_high(low, full, high, ax=a2, arrows=arrows)
    fig.tight_layout(h_pad=2.5)
    _save(fig, out)
    return fig


def plot_paper_fig4(full, mcmc=None, fisher=None, out=None, logGmu_range=(-16, -10),
                    ylim=(3e0, 2e3), figsize=(7, 6.5)):
    """Paper Fig. 4: SBI (orange bars, std over realisations) vs MCMC (red circles) and
    Fisher (green stars), integer injections only.

    mcmc   : (logGmu, precision) arrays, e.g. ``scan.mcmc_precision(chain_files)``
    fisher : path/dict with 'logGmu_inj', 'precision'
    """
    full = _load(full)
    m = _select(full, *logGmu_range, integers_only=True)
    x, p, e = full["logGmu_inj"][m], full["precision_mean"][m], full["precision_std"][m]
    lo, hi = logGmu_range

    def _in(xs):
        xs = np.asarray(xs)
        return (xs >= lo - 1e-9) & (xs <= hi + 1e-9) & np.isclose(xs, np.round(xs))

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x, p, width=0.4, color=ORANGE, edgecolor=ORANGE_EDGE, linewidth=1.2, label="SBI",
           zorder=1)
    if mcmc is not None:
        mx, my = map(np.asarray, mcmc)
        k = _in(mx)
        ax.scatter(mx[k], my[k], marker="o", s=30, color="red", label="MCMC", zorder=3)
    if fisher is not None:
        f = _load(fisher)
        k = _in(f["logGmu_inj"])
        ax.scatter(f["logGmu_inj"][k], f["precision"][k], marker="*", s=90, color="#149331",
                   edgecolor="black", linewidth=0.5, label="Fisher", zorder=4)
    ax.errorbar(x, p, yerr=e, fmt="none", ecolor="black", elinewidth=1.2, capsize=3,
                capthick=1.2, zorder=10)
    ax.set_yscale("log")
    ax.set_ylim(*ylim)
    ax.set_xticks(x)
    ax.set_xticklabels([f"$10^{{{int(v)}}}$" for v in x], fontsize=11)
    ax.tick_params(axis="y", labelsize=11)
    ax.set_xlabel(r"$G\mu_{\rm inj}$", fontsize=13)
    ax.set_ylabel(r"$\delta G\mu~\%$", fontsize=13)
    ax.grid(axis="y", linestyle="-", alpha=0.3)
    ax.legend(fontsize=11, loc="upper right")
    fig.tight_layout()
    _save(fig, out)
    return fig
