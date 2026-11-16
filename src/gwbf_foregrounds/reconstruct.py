"""Reconstruct each spectral component (and the noise) from posterior samples.

Works with samples from any source (SBI or MCMC): an (n, ndim) array in the
model's parameter order.
"""
from dataclasses import dataclass

import numpy as np

from . import data
from .models import get_model
from .spectra import component_spectra, noise
from .stats import bands


@dataclass
class Reconstruction:
    f_GeV: np.ndarray
    model: str
    samples: np.ndarray       # (n, ndim) posterior samples used
    stacks: dict              # component -> (n, n_f) spectra, incl. 'noise_AA'/'noise_TT'
    truths: dict = None       # component -> (n_f,) injected spectra, if known

    @property
    def f_Hz(self):
        return self.f_GeV * data.units["GeV_to_Hz"]

    def bands(self, levels=(0.95,)):
        """component -> {'median', level: (lo, hi)}."""
        return {k: bands(v, levels) for k, v in self.stacks.items()}


def reconstruct(samples, model, f_GeV=None, truth=None, include_noise=True, max_samples=None,
                seed=0):
    """Spectra of every component for every sample.

    samples : (n, ndim) array in ``model.param_names`` order
    f_GeV   : frequencies; default = the coarse data grid
    truth   : injected parameter dict, to also return the true spectra
    """
    m = get_model(model)
    samples = np.asarray(samples, np.float64)
    if samples.ndim != 2 or samples.shape[1] != m.ndim:
        raise ValueError(f"samples must have shape (n, {m.ndim}) for model {m.name!r}")
    if max_samples and len(samples) > max_samples:
        idx = np.random.default_rng(seed).choice(len(samples), max_samples, replace=False)
        samples = samples[idx]
    if f_GeV is None:
        f_GeV, _ = data.frequency_grid(m)
    f_GeV = np.asarray(f_GeV)

    keys = list(m.components)
    stacks = {k: np.empty((len(samples), len(f_GeV))) for k in keys}
    if include_noise:
        stacks["noise_AA"] = np.empty((len(samples), len(f_GeV)))
        stacks["noise_TT"] = np.empty((len(samples), len(f_GeV)))
    for i, s in enumerate(samples):
        p = m.to_params(s)
        for k, v in component_spectra(f_GeV, p, m).items():
            stacks[k][i] = v
        if include_noise:
            n_AA, n_TT = noise(f_GeV, p["A_acc"], p["A_p"])
            stacks["noise_AA"][i] = np.asarray(n_AA)
            stacks["noise_TT"][i] = np.asarray(n_TT)

    truths = None
    if truth is not None:
        p = {**m.fixed, **truth}
        truths = component_spectra(f_GeV, p, m)
        if include_noise:
            n_AA, n_TT = noise(f_GeV, p["A_acc"], p["A_p"])
            truths["noise_AA"], truths["noise_TT"] = np.asarray(n_AA), np.asarray(n_TT)
    return Reconstruction(f_GeV, m.name, samples, stacks, truths)
