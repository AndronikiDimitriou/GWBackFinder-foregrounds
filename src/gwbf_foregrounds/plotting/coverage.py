"""Coverage plots: per-parameter empirical coverage, TARP, SBC ranks."""
import numpy as np
import matplotlib.pyplot as plt

from ..coverage import alpha_from_z, estimate_hat_z


def plot_coverage(gammas, labels, zmax=3.5, ncols=4, out=None, title_fontsize=20,
                  label_fontsize=18, tick_fontsize=16, annot_fontsize=13, legend_fontsize=13):
    """Empirical coverage hat-z vs confidence z, one panel per parameter.

    gammas : list of arrays from ``coverage.gamma_coverage`` (one per parameter)
    labels : panel titles, same order
    """
    n = len(gammas)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5.5 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, g, lab in zip(axes, gammas, labels):
        hz = estimate_hat_z(np.asarray(g), zmax=zmax)
        z = hz["z"]
        ax.plot(z, hz["mean"], "k", linewidth=2)
        ax.fill_between(z, hz["lower"], np.where(np.isinf(hz["upper"]), 100.0, hz["upper"]),
                        color="0.8")
        ax.plot([0, zmax], [0, zmax], "--", color="darkgreen", linewidth=2)
        for t in range(1, int(zmax) + 1):
            level = np.interp(t, z, hz["mean"])
            if np.isfinite(level):
                ax.plot([0, t], [level, level], ":", color="r")
                ax.text(0.1, level + 0.05, f"{(1 - alpha_from_z(level)) * 100:.2f}%",
                        fontsize=annot_fontsize)
                ax.plot([t, t], [0, level], ":", color="r")
            else:
                ax.plot([t, t], [0, 10.0], ":", color="r")
            ax.text(t, 0.3, f"{(1 - alpha_from_z(t)) * 100:.2f}%", rotation=-90,
                    fontsize=annot_fontsize)
        ax.set_title(lab, fontsize=title_fontsize)
        ax.set_xlim(0, zmax)
        ax.set_ylim(0, zmax + 0.5)
        ax.set_ylabel(r"Empirical coverage, $\hat z$", fontsize=label_fontsize)
        ax.set_xlabel("Confidence level, $z$", fontsize=label_fontsize)
        ax.tick_params(axis="both", labelsize=tick_fontsize)
        ax.text(zmax / 2, zmax / 2 + 0.4, "Conservative", ha="center", va="center", rotation=40,
                color="darkgreen", fontsize=legend_fontsize)
        ax.text(zmax / 2 + 0.4, zmax / 2, "Overconfident", ha="center", va="center", rotation=40,
                color="darkgreen", fontsize=legend_fontsize)
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=150, bbox_inches="tight")
    return fig


def plot_tarp(ecp, alpha, n_sims, atc=None, ks_pval=None, interval_alpha=0.05,
              title="TARP coverage diagnostic", out=None):
    """TARP expected coverage with a Jeffreys band for ``n_sims`` test simulations."""
    import scipy.stats

    ecp, alpha = np.asarray(ecp), np.asarray(alpha)
    k = np.clip(np.round(ecp * n_sims), 0, n_sims)
    lo = np.where(k > 0, scipy.stats.beta.ppf(interval_alpha / 2, k + 0.5, n_sims - k + 0.5), 0)
    hi = np.where(k < n_sims,
                  scipy.stats.beta.ppf(1 - interval_alpha / 2, k + 0.5, n_sims - k + 0.5), 1)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], color="darkgreen", ls="--", lw=1.5, label="Ideal", zorder=2)
    ax.fill_between(alpha, lo, hi, color="0.8", alpha=0.7, zorder=1,
                    label=f"{int(100 * (1 - interval_alpha))}% Jeffreys interval")
    ax.plot(alpha, ecp, color="black", lw=2.2, zorder=3, label="TARP")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel(r"Credibility level $\alpha$", fontsize=13)
    ax.set_ylabel("Expected coverage probability", fontsize=13)
    ax.set_title(title, fontsize=13)
    ax.grid(alpha=0.25, linestyle=":")
    txt = []
    if atc is not None:
        txt.append(rf"ATC $= {atc:.3f}$")
    if ks_pval is not None:
        txt.append(rf"KS $p$-value $= {ks_pval:.3f}$")
    if txt:
        ax.text(0.04, 0.93, "\n".join(txt), transform=ax.transAxes, fontsize=11, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.7"))
    ax.legend(fontsize=10.5, loc="lower right")
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=200, bbox_inches="tight")
    return fig


def plot_sbc_ranks(ranks, num_posterior_samples, param_names, out_pattern=None):
    """One SBC rank-CDF figure per parameter (sbi's ``sbc_rank_plot``)."""
    from sbi.analysis import sbc_rank_plot

    figs = []
    for i, name in enumerate(param_names):
        fig, _ = sbc_rank_plot(ranks=np.asarray(ranks)[:, i:i + 1],
                               num_posterior_samples=num_posterior_samples,
                               plot_type="cdf", parameter_labels=[name])
        fig.tight_layout()
        if out_pattern:
            fig.savefig(out_pattern.format(name=name), dpi=200, bbox_inches="tight")
        figs.append(fig)
    return figs
