import hashlib
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from .cosmo import *

jax.config.update("jax_enable_x64", True)  # optional

# -------------------------------
# 1) Loader for Peera's 3-col files
# -------------------------------
def load_peera_spectra(path):
    """
    Read Peera's data file with columns:
      log10_Gmu   log10_f_Hz   log10_h2_Omega_GW
    Return a dict with a uniform frequency grid shared across all Gmu rows.
    """
    df = pd.read_csv(
        path,
        #delim_whitespace=True,
        sep='\s+',
        header=None,
        names=["log10_Gmu", "log10_f_Hz", "log10_h2_Omega_GW"],
    )
    # group by Gmu, sort each group by frequency
    groups = [g.sort_values("log10_f_Hz") for _, g in df.groupby("log10_Gmu")]

    # sort groups by Gmu value
    logG_list = np.array([g["log10_Gmu"].iloc[0] for g in groups], dtype=float)
    order = np.argsort(logG_list)
    groups = [groups[i] for i in order]
    log10Gmu_grid = np.array(sorted(logG_list), dtype=float)

    # master frequency grid from the first group
    master_logf = groups[0]["log10_f_Hz"].to_numpy(dtype=float)

    def to_master_logOmega(g):
        lf = g["log10_f_Hz"].to_numpy(dtype=float)
        lo = g["log10_h2_Omega_GW"].to_numpy(dtype=float)
        if np.allclose(lf, master_logf, rtol=0, atol=1e-12):
            return lo
        return np.interp(master_logf, lf, lo)

    rows = [to_master_logOmega(g) for g in groups]
    log10_h2Omega_mat = np.stack(rows, axis=0)  # shape: (n_G, n_f)

    return {
        "log10Gmu_grid": log10Gmu_grid,          # (n_G,)
        "log10_f_grid": master_logf,             # (n_f,)
        "log10_h2Omega_mat": log10_h2Omega_mat,  # (n_G, n_f)  <-- consistent key
        "f_min_Hz": float(10**master_logf[0]),
        "f_max_Hz": float(10**master_logf[-1]),
    }

# -------------------------------
# 2) Fast Hz interpolator builder
# -------------------------------
# Cache for interpolation functions to avoid repeated JAX compilation
_interpolator_cache = {}

def build_interpolator_from_spectra_bundle(bundle):
    """
    Build a fast, JIT-compiled evaluator:
      h2Om_Hz(log10Gmu, f_Hz) -> h^2 Omega_GW
    """
    global _interpolator_cache
    
    # Create a cache key that includes the actual spectrum data so that
    # VOS vs BOS (same grid, different values) get different interpolators
    mat = bundle["log10_h2Omega_mat"]
    data_hash = hashlib.sha256(mat.tobytes()).hexdigest()
    cache_key = (
        tuple(bundle["log10Gmu_grid"]),
        tuple(bundle["log10_f_grid"]),
        mat.shape,
        data_hash,
    )
    
    # Return cached function if available
    if cache_key in _interpolator_cache:
        return _interpolator_cache[cache_key]
    
    logG_grid_j = jnp.asarray(bundle["log10Gmu_grid"])
    logf_grid_j = jnp.asarray(bundle["log10_f_grid"])
    logOm_mat_j = jnp.asarray(bundle["log10_h2Omega_mat"])  # <-- fixed key
    
    @jax.jit
    def Om_GeV(f_GeV, log10Gmu):
        #logf = jnp.log10(jnp.asarray(f_Hz))
        logf = jnp.log10(jnp.asarray(f_GeV) * units['GeV_to_Hz'])

        def eval_at_logf(lf):
            vals_over_G = jax.vmap(lambda row: jnp.interp(lf, logf_grid_j, row))(logOm_mat_j)
            return jnp.interp(log10Gmu, logG_grid_j, vals_over_G)

        logOm = jax.vmap(eval_at_logf)(logf) if logf.ndim else eval_at_logf(logf)
        return 10.0**logOm #/ (LCDM['h']**2)

    # Cache the function
    _interpolator_cache[cache_key] = Om_GeV

    return Om_GeV
