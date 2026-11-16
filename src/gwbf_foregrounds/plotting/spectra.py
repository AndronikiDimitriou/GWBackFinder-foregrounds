"""Spectrum plots: component reconstruction, and LISA noise with the PLS."""
import numpy as np
import matplotlib.pyplot as plt

from ..models import COMPONENT_STYLE


def plot_reconstruction(recon, out=None, level=0.95, show_noise=True, show_truth=True,
                        data=None, ylim=None, compare=None, compare_label="MCMC"):
    """Median and ``level`` band of each component; dashed = injected.

    recon   : ``reconstruct.Reconstruction``
    data    : optional ``Dataset`` to show D_AA as grey points
    compare : optional second Reconstruction (e.g. MCMC) drawn as dotted outlines
    """
    f = recon.f_Hz
    b = recon.bands(levels=(level,))
    fig, ax = plt.subplots(figsize=(9, 6))
    if data is not None:
        ax.plot(data.f_Hz, data.D_AA, ".", ms=2.5, color="0.75", alpha=0.5,
                label=r"data $D_{AA}$", zorder=1)
    for k, (color, label) in COMPONENT_STYLE.items():
        if k not in b:
            continue
        if k == "MBHB":   # fixed, not inferred: one line
            ax.plot(f, b[k]["median"], "-", color=color, lw=1.2, label=label, zorder=4)
            continue
        lo, hi = b[k][level]
        ax.fill_between(f, lo, hi, color=color, alpha=0.35, lw=0, zorder=3)
        ax.plot(f, b[k]["median"], color=color, lw=1.5, label=label, zorder=4)
        if show_truth and recon.truths is not None:
            ax.plot(f, recon.truths[k], "--", color=color, lw=1.0, zorder=4)
        if compare is not None and k in compare.stacks:
            cb = compare.bands(levels=(level,))[k]
            ax.plot(compare.f_Hz, cb[level][0], ":", color=color, lw=1.2, zorder=5)
            ax.plot(compare.f_Hz, cb[level][1], ":", color=color, lw=1.2, zorder=5)
    if show_noise and "noise_AA" in b:   # after the components: last in the legend, as in the paper
        lo, hi = b["noise_AA"][level]
        ax.fill_between(f, lo, hi, color="black", alpha=0.7, lw=1, zorder=2)
        ax.plot(f, b["noise_AA"]["median"], color="black", lw=1.5, label=r"noise $AA$", zorder=4)
    if compare is not None:
        ax.plot([], [], ":", color="0.3", label=f"{compare_label} {int(level * 100)}%")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(f.min(), f.max())
    if ylim is None:
        keys = [k for k in recon.stacks if not k.startswith("noise")]
        top = max(np.max(b[k]["median"]) for k in keys) * 10
        ylim = (top * 1e-6, top)
    ax.set_ylim(*ylim)
    ax.set_xlabel(r"$f~[\mathrm{Hz}]$", fontsize=16)
    ax.set_ylabel(r"$h^2\,\Omega_{\rm GW}(f)$", fontsize=16)
    ax.tick_params(labelsize=14)
    ax.legend(fontsize=13, loc="upper right", ncol=2)
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=150, bbox_inches="tight")
    return fig


def power_law_sensitivity(f_Hz, h2Omega_noise, snr_thr=10.0, t_obs_yr=3.0, f_ref=1e-3,
                          betas=np.linspace(-8.0, 8.0, 321)):
    """Power-law sensitivity curve of one channel."""
    t_obs = t_obs_yr * 365.25 * 24 * 3600.0
    x = f_Hz / f_ref
    pls = np.zeros_like(f_Hz)
    trapz = getattr(np, "trapezoid", None) or np.trapz
    for beta in betas:
        integral = trapz((x ** beta / h2Omega_noise) ** 2, f_Hz)
        pls = np.maximum(pls, snr_thr / np.sqrt(t_obs * integral) * x ** beta)
    return pls


def plot_noise_and_pls(out=None, A_p=15.0, A_acc=3.0, snr_thr=10.0, t_obs_yr=3.0, n_f=600):
    """AA and TT noise curves and the AA power-law sensitivity."""
    from ..spectra import Hz_to_GeV, noise

    f_Hz = np.logspace(-5, 0, n_f)
    n_AA, n_TT = (np.asarray(a) for a in noise(Hz_to_GeV(f_Hz), A_acc, A_p))
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(f_Hz, n_AA, lw=2, label="AA noise")
    ax.plot(f_Hz, n_TT, lw=2, label="TT noise")
    ax.plot(f_Hz, power_law_sensitivity(f_Hz, n_AA, snr_thr, t_obs_yr), lw=2, ls="--",
            color="black", label="PLS")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-5, 1)
    ax.set_ylim(1e-14, 1e-1)
    ax.set_xlabel(r"$f\,[\mathrm{Hz}]$", fontsize=16)
    ax.set_ylabel(r"$h^2\Omega_{\rm noise}(f)$", fontsize=16)
    ax.tick_params(labelsize=16)
    ax.legend(fontsize=14, loc="upper left")
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=150, bbox_inches="tight")
    return fig
