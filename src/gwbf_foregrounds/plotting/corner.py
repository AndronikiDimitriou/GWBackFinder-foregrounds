"""MCMC vs SBI posteriors: overlaid corner plot with HDI shading, and 1D marginals."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from ..stats import hdi

SBI_COLOR = "#048929"
MCMC_COLOR = "black"

# default plot range per parameter: SBI mean +- factor * SBI std
DEFAULT_ZOOM = {"logGmu": 1.5, "A_p": 5, "A_acc": 9.0, "logA_Gal": 0.5,
                "logA_ExB": 4, "logA_ExWB": 4.0, "logA_EMRI": 4.0}

FONTS = dict(axis_label=35, tick=27, hdi_title=25, legend=30, suptitle=35)


def _fmt_interval(lo, hi):
    diff = abs(hi - lo)
    if diff == 0:
        return f"$[{lo:.4f},\\ {hi:.4f}]$"
    n = max(2, int(np.ceil(-np.log10(diff))) + 1)
    return f"$[{lo:.{n}f},\\ {hi:.{n}f}]$"


def _shade_hdi(hist_ax, x):
    """Diagonal histogram bars: opaque inside 68%, then 95%, 99.7%, faint outside."""
    lims = [hdi(x, p) for p in (0.6827, 0.9545, 0.9973)]
    for patch in hist_ax.patches:
        c = patch.get_x() + 0.5 * patch.get_width()
        alpha = 0.1
        for (lo, hi), a in zip(lims, (1.0, 0.5, 0.25)):
            if lo < c < hi:
                alpha = a
                break
        patch.set_alpha(alpha)


def _corner_layer(samples, fig, color, labels, limits, truths, annotate_hdi, fonts):
    import corner

    ndim = samples.shape[1]
    fig = corner.corner(
        samples, labels=labels, truths=truths, truth_color="red", bins=100,
        label_kwargs={"fontsize": fonts["axis_label"]},
        levels=(1.0 - np.exp(-1.15), 1.0 - np.exp(-3.09)),
        plot_contours=True, fill_contours=False, smooth=True, smooth1d=True,
        plot_datapoints=False, color=color, labelpad=0.35, fig=fig,
        hist_kwargs={"linewidth": 2.5}, range=limits, show_titles=False,
        contour_kwargs={"linewidths": 1.8},
    )
    ax = fig.get_axes()
    for j in range(ndim):
        for patch in ax[ndim * j + j].patches:
            patch.set_fill(False)
            patch.set_edgecolor(color)
            patch.set_linewidth(2.5)
    for j in range(1, ndim):
        for i in range(j):
            ax[i + j * ndim].grid(linestyle=":", zorder=0, alpha=0.4)
    for axi in ax:
        axi.tick_params(axis="both", labelsize=fonts["tick"], rotation=0, length=5, width=1.2,
                        pad=8)
        axi.xaxis.set_major_locator(plt.MaxNLocator(4))
        axi.yaxis.set_major_locator(plt.MaxNLocator(4))
        for lab in axi.get_xticklabels():
            lab.set_rotation(40)
            lab.set_ha("right")
    if annotate_hdi:
        for j in range(ndim):
            _shade_hdi(ax[ndim * j + j], samples[:, j])
            lo, hi = hdi(samples[:, j], 0.9545)
            ax[ndim * j + j].set_title(_fmt_interval(lo, hi), fontsize=fonts["hdi_title"],
                                       y=1.08, color=color)
    return fig


def plot_mcmc_vs_sbi(mcmc_samples, sbi_samples, param_names, truths=None, labels=None,
                     keep=None, offsets=None, zoom=None, limits=None, n_plot=10_000, title=None,
                     out=None, fonts=None, figsize=(26, 26)):
    """Overlaid corner plot: MCMC in black, SBI in green with 95% HDI titles.

    keep    : parameter names to show (e.g. drop 'A_p', 'A_acc'); default all
    offsets : {name: value} added to that parameter in BOTH sample sets and the truth,
              e.g. to show an amplitude in absolute instead of relative units
    zoom    : {name: factor}; axis range = SBI mean +- factor * SBI std
    limits  : {name: (lo, hi)} explicit axis ranges (in displayed units), overriding zoom
    """
    fonts = {**FONTS, **(fonts or {})}
    zoom = {**DEFAULT_ZOOM, **(zoom or {})}
    names = list(param_names)
    idx = [names.index(n) for n in (keep or names)]
    names = [names[i] for i in idx]
    mcmc = np.array(mcmc_samples, float)[:, idx]
    sbi = np.array(sbi_samples, float)[:, idx]
    tr = None if truths is None else np.array(truths, float)[idx]
    for n, off in (offsets or {}).items():
        if n in names:
            j = names.index(n)
            mcmc[:, j] += off
            sbi[:, j] += off
            if tr is not None:
                tr[j] += off
    if labels is None:
        from ..models import LATEX
        labels = [LATEX.get(n, n) for n in names]
    else:
        labels = [labels[i] for i in idx] if len(labels) == len(param_names) else labels

    rng = np.random.default_rng(0)
    if len(mcmc) > n_plot:
        mcmc = mcmc[rng.choice(len(mcmc), n_plot, replace=False)]
    explicit = dict(limits or {})
    ranges = [tuple(explicit[n]) if n in explicit else
              (sbi[:, j].mean() - zoom.get(n, 5.0) * sbi[:, j].std(),
               sbi[:, j].mean() + zoom.get(n, 5.0) * sbi[:, j].std())
              for j, n in enumerate(names)]

    ndim = len(names)
    fig = plt.figure(figsize=figsize)
    fig = _corner_layer(mcmc, fig, MCMC_COLOR, labels, ranges, tr, False, fonts)
    fig = _corner_layer(sbi, fig, SBI_COLOR, labels, ranges, tr, True, fonts)
    for k, axi in enumerate(fig.get_axes()[: ndim * ndim]):
        if k % ndim > k // ndim:
            axi.set_visible(False)
    fig.subplots_adjust(wspace=0.15, hspace=0.13, left=0.09, bottom=0.08)
    fig.legend(handles=[Line2D([0], [0], color=MCMC_COLOR, lw=3, label="MCMC"),
                        Line2D([0], [0], color=SBI_COLOR, lw=3, label="SBI"),
                        Line2D([0], [0], color="red", lw=3, ls="--", label="Truth")],
               loc="upper right", fontsize=fonts["legend"], bbox_to_anchor=(0.98, 0.98))
    if title:
        fig.suptitle(title, fontsize=fonts["suptitle"], y=1.01)
    if out:
        fig.savefig(out, dpi=200, bbox_inches="tight")
    return fig


def plot_1d_posteriors(mcmc_samples, sbi_samples, param_names, truths=None, labels=None,
                       title=None, out=None):
    """One panel per parameter: MCMC (black) and SBI (blue) step histograms."""
    from ..models import LATEX

    names = list(param_names)
    labels = labels or [LATEX.get(n, n) for n in names]
    fig, axes = plt.subplots(1, len(names), figsize=(3.4 * len(names), 4))
    axes = np.atleast_1d(axes)
    for i, (ax, name) in enumerate(zip(axes, names)):
        mean, std = sbi_samples[:, i].mean(), sbi_samples[:, i].std()
        k = 2 if name in ("logGmu", "A_p") else 6
        rng_ = (mean - k * std, mean + k * std)
        ax.hist(mcmc_samples[:, i], bins=100, range=rng_, density=True, color=MCMC_COLOR,
                histtype="step", linewidth=2, label="MCMC")
        ax.hist(sbi_samples[:, i], bins=100, range=rng_, density=True, color="#01BAFD",
                histtype="step", linewidth=2, label="SBI")
        if truths is not None:
            ax.axvline(truths[i], color="red", lw=2, ls="--", label="Truth")
        ax.set_xlim(*rng_)
        ax.set_xlabel(labels[i], fontsize=15)
    axes[0].set_ylabel("Density", fontsize=15)
    axes[0].legend(fontsize=9)
    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=150, bbox_inches="tight")
    return fig
