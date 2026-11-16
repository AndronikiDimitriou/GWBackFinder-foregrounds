"""Fisher-matrix forecast of the precision on G mu, for any of the three models.

F_ij = N_c * sum_k n_k * sum_{c in AA, EE, TT} dln(Omega_c)/dtheta_i * dln(Omega_c)/dtheta_j

with exact derivatives from ``jax.jacrev``. Optionally the Gaussian foreground priors
are added (``with_priors=True``); by default they are not (classic Fisher, as in the scripts).
"""
import os

import jax
import jax.numpy as jnp
import numpy as np

from . import data
from .models import get_model
from .physics import lisa
from .spectra import N_CHUNKS, total_signal
from .stats import sigma_to_precision

DEFAULT_GRID = np.arange(-17.0, -9.0 + 0.125, 0.25)


def _prediction(f_GeV, theta_vec, model):
    p = dict(zip(model.param_names, theta_vec))
    sig = total_signal(f_GeV, p, model)
    AA = lisa.h2Omega_noise_AA(f_GeV, p["A_acc"], p["A_p"]) + sig
    TT = lisa.h2Omega_noise_TT(f_GeV, p["A_acc"], p["A_p"]) + sig
    return jnp.stack([AA, AA, TT])          # EE = AA


def prior_fisher(model):
    """Fisher matrix of the Gaussian priors: diag(1/sigma^2) for the foreground amplitudes,
    0 for parameters with uniform priors (log G mu, A_p, A_acc)."""
    m = get_model(model)
    return np.diag([1.0 / m.prior[n][2] ** 2 if m.prior[n][0] == "gaussian" else 0.0
                    for n in m.param_names])


def fisher_matrix(model, params, f_obs=None, n_k=None, with_priors=False):
    """Fisher matrix at ``params`` (channels independent).

    with_priors=False: classic Fisher of the likelihood only.
    with_priors=True : adds the Gaussian foreground priors (``prior_fisher``), the same
                       prior information SBI and MCMC use.
    """
    m = get_model(model)
    if f_obs is None:
        f_obs, n_k = data.frequency_grid(m)
    theta = jnp.array(m.theta(params))
    f = jnp.asarray(f_obs)
    fid = np.asarray(_prediction(f, theta, m))                           # (3, n_f)
    jac = np.asarray(jax.jacrev(lambda v: _prediction(f, v, m))(theta))  # (3, n_f, ndim)
    dln = jac / fid[:, :, None]
    F = N_CHUNKS * np.einsum("k,cki,ckj->ij", np.asarray(n_k), dln, dln)
    return F + prior_fisher(m) if with_priors else F


def sigma_logGmu(F):
    try:
        var = np.linalg.inv(F)[0, 0]
    except np.linalg.LinAlgError:
        return np.nan
    return float(np.sqrt(var)) if var > 0 else np.nan


def run_fisher(model="full", logGmu_values=DEFAULT_GRID, output_dir=".", output_name=None,
               with_priors=False, verbose=True):
    """Fisher precision for each injected log10 G mu. Saves the same keys as
    ``fisher_95_intervals.npz`` / ``fisher_nofg_95_intervals.npz``."""
    m = get_model(model)
    f_obs, n_k = data.frequency_grid(m)
    rows = []
    for lg in np.asarray(logGmu_values, float):
        F = fisher_matrix(m, m.fiducial(logGmu=lg), f_obs, n_k, with_priors=with_priors)
        sig = sigma_logGmu(F)
        prec, lo, hi = sigma_to_precision(sig, lg)
        rows.append(dict(logGmu_inj=lg, sigma_logGmu=sig, lo_logGmu=lo, hi_logGmu=hi,
                         lo_Gmu=10 ** lo, hi_Gmu=10 ** hi, precision=prec, fisher_matrices=F))
        if verbose:
            print(f"[fisher {m.name}] logGmu={lg:6.2f}  sigma={sig:.4f}  precision={prec:.4g} %")
    out = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    out["Gmu_inj"] = 10 ** out["logGmu_inj"]
    tag = "_priors" if with_priors else ""
    name = output_name or (f"fisher{tag}_95_intervals.npz" if m.name == "full"
                           else f"fisher_{m.name}{tag}_95_intervals.npz")
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, name)
    np.savez(path, **out)
    if verbose:
        print(f"Saved -> {path}")
    return out
