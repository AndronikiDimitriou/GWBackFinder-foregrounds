"""MCMC (emcee) for the full 7-parameter model, as in ``mcmc_paper2.py``.

Likelihood per channel (AA, EE, TT), weighted by n_k and N_c:
    1/3 Gaussian in (D - Omega)/Omega  +  2/3 log-normal in ln(Omega/D).
Prior: ``FULL.prior`` (uniform on logGmu, A_p, A_acc; Gaussian on the foregrounds).
"""
import numpy as np

from . import data
from .models import FULL
from .spectra import N_CHUNKS, model_prediction

MODEL = FULL


def _log_L_channel(D, Omega, n_k):
    log_L_G = -0.5 * N_CHUNKS * np.sum(n_k * ((D - Omega) / Omega) ** 2)
    log_L_LN = -0.5 * N_CHUNKS * np.sum(n_k * np.log(Omega / D) ** 2)
    return log_L_G / 3.0 + 2.0 * log_L_LN / 3.0


def log_likelihood(params, f_obs, D_AA, D_EE, D_TT, n_k):
    O_AA, O_EE, O_TT = model_prediction(f_obs, params, MODEL)
    return (_log_L_channel(D_AA, O_AA, n_k) + _log_L_channel(D_EE, O_EE, n_k)
            + _log_L_channel(D_TT, O_TT, n_k))


def log_prior(theta):
    lp = 0.0
    for name, val in zip(MODEL.param_names, theta):
        kind, a, b = MODEL.prior[name]
        if kind == "uniform":
            if not a <= val <= b:
                return -np.inf
        else:
            lp += -0.5 * ((val - a) / b) ** 2
    return lp


def log_posterior(theta, f_obs, D_AA, D_EE, D_TT, n_k):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    try:
        ll = log_likelihood(MODEL.to_params(theta), f_obs, D_AA, D_EE, D_TT, n_k)
    except Exception:
        return -np.inf
    return lp + ll if np.isfinite(ll) else -np.inf


def make_vectorized_log_posterior(f_obs, D_AA, D_EE, D_TT, n_k):
    """Same posterior as ``log_posterior``, compiled with JAX and evaluated for many walkers
    at once: theta of shape (n_walkers, 7) -> (n_walkers,). ~100x faster."""
    import jax
    import jax.numpy as jnp

    from .physics import lisa
    from .spectra import total_signal

    f = jnp.asarray(f_obs)
    D = jnp.stack([jnp.asarray(D_AA), jnp.asarray(D_EE), jnp.asarray(D_TT)])
    w = jnp.asarray(N_CHUNKS * np.asarray(n_k))

    def loglike(th):
        p = {**MODEL.fixed, **dict(zip(MODEL.param_names, th))}
        S = total_signal(f, p, MODEL)
        nA = lisa.h2Omega_noise_AA(f, p["A_acc"], p["A_p"])
        nT = lisa.h2Omega_noise_TT(f, p["A_acc"], p["A_p"])
        mu = jnp.stack([nA + S, nA + S, nT + S])
        return jnp.sum(w * (-0.5 * ((D - mu) / mu) ** 2 / 3.0 - 0.5 * jnp.log(mu / D) ** 2 * 2.0 / 3.0))

    batched = jax.jit(jax.vmap(loglike))

    def logpost(thetas):
        thetas = np.atleast_2d(thetas)
        lp = np.array([log_prior(t) for t in thetas])
        ll = np.asarray(batched(jnp.asarray(thetas)))
        out = np.where(np.isfinite(lp), lp + ll, -np.inf)
        return np.where(np.isfinite(out), out, -np.inf)

    return logpost


DEFAULT_INIT_SCALE = np.array([0.5, 0.5, 0.2, 0.05, 0.05, 0.05, 0.05])


def run_mcmc(
    dataset,
    output_file=None,
    start_logGmu=-19.0,
    nwalkers=None,
    init_scale=DEFAULT_INIT_SCALE,
    tau_check_interval=500,
    min_tau_multiple=50,
    max_steps=1_000_000,
    discard_tau_mult=5,
    seed=0,
    progress=True,
):
    """Run emcee on a full-model ``Dataset`` until the autocorrelation time has settled.

    Walkers start around the truth except log10 G mu, which starts at ``start_logGmu``
    (deliberately far away, to test recovery). Stops when
    ``steps > min_tau_multiple * tau`` and tau changed < 5% since the last check;
    then discards ``discard_tau_mult * max(tau)`` steps as burn-in.
    Returns (samples, logp, tau, burnin) and saves them to ``output_file`` if given.
    """
    import emcee

    if dataset.model != "full":
        raise ValueError("MCMC is implemented for the full model only")
    ndim = MODEL.ndim
    nwalkers = nwalkers or max(64, 4 * ndim)
    f_obs = dataset.f_obs
    _, n_k = data.frequency_grid("full")
    if len(n_k) != len(f_obs):  # pragma: no cover
        raise ValueError("dataset frequency grid does not match the model grid")
    args = (f_obs, dataset.D_AA, dataset.D_EE, dataset.D_TT, n_k)

    truth = np.array(MODEL.theta(dataset.params))
    center = truth.copy()
    center[0] = start_logGmu
    rng = np.random.default_rng(seed)
    pos = center + np.asarray(init_scale) * rng.standard_normal((nwalkers, ndim))

    logpost = make_vectorized_log_posterior(*args)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, logpost, vectorize=True)
    old_tau = np.full(ndim, np.inf)
    for _ in sampler.sample(pos, iterations=max_steps, progress=progress):
        if sampler.iteration % tau_check_interval:
            continue
        tau = np.asarray(sampler.get_autocorr_time(tol=0))
        n = sampler.iteration
        if (np.all(np.isfinite(old_tau)) and np.all(tau * min_tau_multiple < n)
                and np.all(np.abs(old_tau - tau) / tau < 0.05)):
            break
        old_tau = tau
    else:
        print("WARNING: max_steps reached without convergence.")

    tau = sampler.get_autocorr_time(tol=0)
    burnin = int(discard_tau_mult * np.max(tau))
    samples = sampler.get_chain(discard=burnin, flat=True)
    logp = sampler.get_log_prob(discard=burnin, flat=True)
    if output_file:
        np.savez(output_file, samples=samples, logp=logp, tau=tau, burnin=burnin,
                 param_names=np.array(MODEL.param_names), true_params=truth,
                 mcmc_start_logGmu=start_logGmu)
    return samples, logp, tau, burnin


def load_chain(path):
    """(samples, param_names, true_params) from an MCMC .npz."""
    with np.load(path, allow_pickle=True) as d:
        return d["samples"], [str(n) for n in d["param_names"]], np.asarray(d["true_params"], float)
