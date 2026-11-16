import sys
import os
_CORE_DIR = os.path.dirname(os.path.abspath(__file__))
from .cosmo import *
from .data_processing import load_peera_spectra, build_interpolator_from_spectra_bundle
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax



@jax.jit
def Omega_GW_power_law(f, A=1.0, alpha=1.0, f_0=2.5e-3):
    """
    Compute the gravitational wave spectrum Ω_GW(f) from a power law model.
    
    Parameters:
    -----------
    f : float or array_like
        Frequency in GeV
    A : float
        Amplitude of the power law (dimensionless, normalized to 1 at f_0)
    alpha : float
        Spectral index of the power law
    f_0 : float
        Reference frequency in Hz (default: 3e-3 Hz)
    """
    f_0 = f_0 * units['Hz_to_GeV']
    A=jnp.asarray(A)
    A_0=jnp.asarray(1e-11)
    f = jnp.asarray(f)
    f_0 = jnp.asarray(f_0)
    A_0 = jnp.asarray(A_0)
    alpha = jnp.asarray(alpha)
    
    return A * A_0 * (f / f_0)**alpha




## --- Omega_GW function with "trick" ---
## Global cache for the interpolation function (keyed by VOS_of_BOS)
#_peera_interp_cache = {}
#
#def h2Omega_GW_Peera_interp(VOS_or_BOS=None):
#    """
#    Return the interpolation function for Peera's tabulated spectra data.
#
#    Parameters
#    ----------
#    VOS_or_BOS : str, optional
#        Which tabulated spectra to use: ``"BOS"`` for BOS_saikawa.dat (default),
#        or ``"VOS"`` for vos_saikawa.dat.
#
#    Returns
#    -------
#    Omega_GeV_interp : function
#        Interpolation function for Peera's tabulated spectra data.
#    """
#    global _peera_interp_cache
#    if VOS_or_BOS is None:
#        raise ValueError("VOS_or_BOS must be provided")
#
#    choice = str(VOS_or_BOS).strip().upper()
#    if choice not in ("VOS", "BOS"):
#        raise ValueError(f"VOS_or_BOS must be 'VOS' or 'BOS', got {VOS_or_BOS!r}")
#
#    # Return cached function if available for this choice (cache must be a dict)
#    if isinstance(_peera_interp_cache, dict) and choice in _peera_interp_cache:
#        return _peera_interp_cache[choice]
#
#    # METHOD USING DATA FROM PEERA AND INTERPOLATION BETWEEN SPECTRA
#    # This branch cannot be JIT-compiled because it returns a JAX function.
#    # Tabulated notebook input data now lives under Work/data/notebook_inputs.
#    _physics_dir = os.path.dirname(os.path.abspath(__file__))
#    _repo_root = os.path.dirname(_physics_dir)
#    filename = "vos_saikawa.dat" if choice == "VOS" else "BOS_saikawa.dat"
#    path_data = os.path.join(
#        _repo_root,
#        "Work",
#        "data",
#        "notebook_inputs",
#        "tabulated_spectra",
#        filename,
#    )
#    bundle = load_peera_spectra(path_data)
#    
#    Omega_GeV_interp = build_interpolator_from_spectra_bundle(bundle)
#    
#    # Change from logGmu to Gmu
#    
#    def Omega_GeV_interp_Gmu(f_GeV, string_params=None, j_min=None, j_max=None, j_d=None, N_a=None, N_f=None, N_tail=None, VOSeq=True, use_tabulated_data=None):
#        # Use same keyword arguments to ensure compatibility with original GW spectra fucntion
#        # Convert to JAX array and use JAX log10 (not NumPy) for JIT compatibility
#        Gmu_val = jnp.asarray(string_params['Gmu'])
#        logGmu = jnp.log10(Gmu_val)
#        return Omega_GeV_interp(f_GeV, logGmu)
#    
#    # Cache the function for this choice
#    if not isinstance(_peera_interp_cache, dict):
#        _peera_interp_cache = {}
#    _peera_interp_cache[choice] = Omega_GeV_interp_Gmu
#
#    return Omega_GeV_interp_Gmu


# --- Simpler direct API: h2Omega_GW_Peera(f_GeV, Gmu) ---
# Cache for functions that take Gmu directly instead of string_params.
_peera_gmu_cache = {}

def _h2Omega_GW_Peera_func(VOS_or_BOS="VOS"):
    """
    Return a cached function with the simple signature
        h2Omega_GW_Peera(f_GeV, Gmu)

    Parameters
    ----------
    VOS_or_BOS : str, optional
        Which tabulated spectra to use: ``"VOS"`` or ``"BOS"``.

    Returns
    -------
    function
        Cached function of the form ``func(f_GeV, Gmu)``.
    """
    global _peera_gmu_cache

    choice = str(VOS_or_BOS).strip().upper()
    if choice not in ("VOS", "BOS"):
        raise ValueError(f"VOS_or_BOS must be 'VOS' or 'BOS', got {VOS_or_BOS!r}")

    if isinstance(_peera_gmu_cache, dict) and choice in _peera_gmu_cache:
        return _peera_gmu_cache[choice]

    filename = "vos_saikawa.dat" if choice == "VOS" else "BOS_saikawa.dat"
    _physics_dir = os.path.dirname(os.path.abspath(__file__))
    # remove _repo_root line
    path_data = os.path.join(
        _physics_dir,
        "tabulated_spectra",
        filename,
    )
    bundle = load_peera_spectra(path_data)
    Omega_GeV_interp = build_interpolator_from_spectra_bundle(bundle)

    def Omega_GeV_interp_Gmu_direct(f_GeV, Gmu):
        Gmu_val = jnp.asarray(Gmu)
        logGmu = jnp.log10(Gmu_val)
        return Omega_GeV_interp(f_GeV, logGmu)

    if not isinstance(_peera_gmu_cache, dict):
        _peera_gmu_cache = {}
    _peera_gmu_cache[choice] = Omega_GeV_interp_Gmu_direct

    return Omega_GeV_interp_Gmu_direct


def h2Omega_Cosmic_Strings_Peera_Gmu(f_GeV, Gmu, VOS_or_BOS=None):
    """
    Evaluate Peera's tabulated GW spectrum with the simple signature
        h2Omega_GW_Peera(f_GeV, Gmu, VOS_or_BOS="VOS")

    Parameters
    ----------
    f_GeV : array_like
        Frequency in GeV.
    Gmu : float or array_like
        Cosmic string tension ``Gmu``.
    VOS_or_BOS : str, optional
        Which tabulated spectra to use: ``"VOS"`` or ``"BOS"``.

    Returns
    -------
    array_like
        Interpolated ``h^2 Omega_GW`` values.
    """
    return _h2Omega_GW_Peera_func(VOS_or_BOS=VOS_or_BOS)(f_GeV, Gmu)

