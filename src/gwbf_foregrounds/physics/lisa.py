import numpy as np
from scipy.special import zeta
import sys
import os

_CORE_DIR = os.path.dirname(os.path.abspath(__file__))

from .cosmo import *
from .cosmicstrings import *
from .data_processing import *
from .GW_backgrounds import Omega_GW_power_law, h2Omega_Cosmic_Strings_Peera_Gmu
from .GW_foregrounds import *
import jax.numpy as jnp
from jax.scipy.special import zeta
from jax import jit, vmap, lax
import jax
from functools import partial
jax.config.update("jax_enable_x64", True)
import jax.random as jr


_lisa_response_cache = {}

# I want to creat lisa freqs as follows: 1) np.linspace(3*10**-5, 5*10**-1, 5*10**5) * units['Hz_to_GeV'] 2) After 1000 freqs, I use instead np.linspace(3*10**-5, 5*10**-1, 5*10**4) * units['Hz_to_GeV'],
# Create LISA frequency array with adaptive resolution
def create_lisa_freqs(f_min=3e-5, f_max=5e-1, f_break=1e-3, n_low=1000, n_high=5*10**5):
    """Create LISA frequency array with high resolution at low frequencies."""
    f_low = np.linspace(f_min, f_break, n_low) * units['Hz_to_GeV']
    f_high = np.linspace(f_break, f_max, n_high) * units['Hz_to_GeV']
    return np.concatenate([f_low, f_high])

f_lisa = create_lisa_freqs()
defaultlisaparams = {
    'N_chunks': 94,
    'f_lisa': np.linspace(3*10**-5, 5*10**-1, 5*10**5) * units['Hz_to_GeV'],
    'f_coarse': np.linspace(1e-3, 0.5, 1000) * units['Hz_to_GeV'],
    'N_macro': 1000,
    #'f_coarse': np.linspace(4*10**-5, 5*10**-1, 400) * units['Hz_to_GeV'],
    #'N_macro': 200,
    'L': 2.5 * 10**9 * units['m_to_GeV'],# m
    'T_obs': 3.0, # years,
    'log10AGal': -7.84,
    'log10AExB': -12.38,
    'log10AExWB': -11.0554,
}

@jit
def P_IMS(f, Ap):
    """IMS noise PSD for LISA"""
    L = defaultlisaparams['L']
    mHz = 10**-3 * units['Hz_to_GeV']
    invHz = 1/units['Hz_to_GeV']
    c   =  1 #299792458.0  # m/s
    xf2 = (2 * jnp.pi * f * L / c)**2
    return 1.6e-43 * Ap**2 * (1 + (2*mHz / f)**4) * xf2 * invHz

@jit
def P_acc(f, Aacc):
    """Acceleration noise PSD for LISA"""
    L = defaultlisaparams['L']
    c     = 1 #299792458.0  # m/s
    mHz = 10**-3 * units['Hz_to_GeV']
    invHz = 1/units['Hz_to_GeV']
    xf2   = (2 * jnp.pi * f * L / c)**2
    return 7.7374e-46 * Aacc**2 * (1 + (0.4 * mHz/ f)**2) * (1 + (f / (8.*mHz))**4) / xf2 * invHz


# --- Response function approximations ---
@jit
def R_AA_approx(xf):  # = R_EE
    return 9.0 / (20.0 + 14.0 * xf**2)

@jit
def R_TT_approx(xf):
    return 9.0 * xf**6 / (3.6e4 + 14.0 * xf**8)

@jit
def curly_R_AA_approx(xf):
    return R_AA_approx(xf) * (16*xf**2 * jnp.sin(xf)**2)

@jit
def curly_R_TT_approx(xf):
    return R_TT_approx(xf) * (16*xf**2 * jnp.sin(xf)**2)

curly_R_EE_approx = curly_R_AA_approx


def _load_lisa_response_table(channel):
    choice = str(channel).strip().upper()
    if choice not in ("AA", "EE", "TT"):
        raise ValueError(f"channel must be 'AA', 'EE', or 'TT', got {channel!r}")

    if choice == "EE":
        choice = "AA"

    response_dir = os.path.join(_CORE_DIR, "LISA_curly_R_response_data")
    response_path = os.path.join(response_dir, f"LISA_curly_R_response_{choice}.npz")

    with np.load(response_path) as response_data:
        grid_key = "xf" if "xf" in response_data.files else "freqs"
        xf_grid = np.asarray(response_data[grid_key], dtype=np.float64).copy()
        response = np.asarray(response_data["response"].real, dtype=np.float64).copy()

    return {
        "xf_grid": jnp.asarray(xf_grid),
        "response": jnp.asarray(response),
        "xf_min": xf_grid[0],
        "xf_max": xf_grid[-1],
    }


if not isinstance(_lisa_response_cache, dict):
    _lisa_response_cache = {}

_lisa_response_cache["AA"] = _load_lisa_response_table("AA")
_lisa_response_cache["EE"] = _lisa_response_cache["AA"]
_lisa_response_cache["TT"] = _load_lisa_response_table("TT")


@jit
def curly_R_AA_exact(xf):
    response_table = _lisa_response_cache["AA"]
    xf_arr = jnp.asarray(xf, dtype=response_table["xf_grid"].dtype)
    xf_clip = jnp.clip(xf_arr, response_table["xf_min"], response_table["xf_max"])
    interpolated = jnp.interp(xf_clip, response_table["xf_grid"], response_table["response"])
    in_bounds = (xf_arr >= response_table["xf_min"]) & (xf_arr <= response_table["xf_max"])
    return jnp.where(in_bounds, interpolated, curly_R_AA_approx(xf_arr))


curly_R_EE_exact = curly_R_AA_exact


@jit
def curly_R_TT_exact(xf):
    response_table = _lisa_response_cache["TT"]
    xf_arr = jnp.asarray(xf, dtype=response_table["xf_grid"].dtype)
    xf_clip = jnp.clip(xf_arr, response_table["xf_min"], response_table["xf_max"])
    interpolated = jnp.interp(xf_clip, response_table["xf_grid"], response_table["response"])
    in_bounds = (xf_arr >= response_table["xf_min"]) & (xf_arr <= response_table["xf_max"])
    return jnp.where(in_bounds, interpolated, curly_R_TT_approx(xf_arr))


curly_R_AA = curly_R_AA_exact
curly_R_EE = curly_R_EE_exact
curly_R_TT = curly_R_TT_exact



# --- Noise PSDs ---
@jit
def N_AA(f, Aacc, Ap):
    L = defaultlisaparams['L']
    xf = 2 * jnp.pi * f * L
    sin2xf = jnp.sin(xf)**2
    cosxf = jnp.cos(xf)
    return 8 * sin2xf * (
        4 * (1 + cosxf + cosxf**2) * P_acc(f, Aacc) +
                       (2 + cosxf) * P_IMS(f, Ap)
    )

N_EE = N_AA

@jit
def N_TT(f, Aacc, Ap):
    L = defaultlisaparams['L']
    xf = 2 * jnp.pi * f * L
    sin2xf = jnp.sin(xf)**2
    cosxf = jnp.cos(xf)
    return 16 * sin2xf * (
        2 * (1 - cosxf)**2 * P_acc(f, Aacc) +
        (1 - cosxf) * P_IMS(f, Ap)
    )

@jit
def _median_of_three(left, center, right):
    return (
        left + center + right
        - jnp.minimum(left, jnp.minimum(center, right))
        - jnp.maximum(left, jnp.maximum(center, right))
    )

@jit
def _moving_median_3_full(values):
    values = jnp.asarray(values)
    original_shape = values.shape
    values_1d = jnp.atleast_1d(values)
    padded = jnp.pad(values_1d, (1, 1), mode='edge')
    median_values = _median_of_three(padded[:-2], padded[1:-1], padded[2:])
    return jnp.reshape(median_values, original_shape)

# --- Ω_noise^αβ(f) ---
@partial(jit, static_argnames=("use_3_point_median",))
def h2Omega_noise_AA(f, Aacc, Ap, use_3_point_median=True):
    h2 = LCDM['h']**2
    H0 = LCDM['H0']
    L = defaultlisaparams['L']
    xf = 2 * jnp.pi * f * L
    omega_noise = h2 * (4 * jnp.pi**2 * f**3 / (3 * H0**2)) * N_AA(f, Aacc, Ap) / curly_R_AA(xf) #(16 * xf**2 * jnp.sin(xf)**2 * R_AA(xf))
    if use_3_point_median:
        return _moving_median_3_full(omega_noise)
    return omega_noise 

h2Omega_noise_EE = h2Omega_noise_AA

@partial(jit, static_argnames=("use_3_point_median",))
def h2Omega_noise_TT(f, Aacc, Ap, use_3_point_median=True):
    h2 = LCDM['h']**2
    H0 = LCDM['H0']
    L = defaultlisaparams['L']
    xf = 2 * jnp.pi * f * L
    omega_noise = h2 * (4 * jnp.pi**2 * f**3 / (3 * H0**2)) * N_TT(f, Aacc, Ap) / curly_R_TT(xf) #(16 * xf**2 * jnp.sin(xf)**2 * R_TT(xf))
    if use_3_point_median:
        return _moving_median_3_full(omega_noise)
    return omega_noise 






#@jit
#def generate_lisa_mock_data_old(frequencies, h2Omega_GW, Aacc, Ap, key = 1):
#    """
#    Generate mock LISA data for a given set of parameters.
#    
#    Parameters
#    ----------
#    frequencies : array_like
#        Frequencies at which to generate data
#    h2Omega_GW : array_like
#        Gravitational wave spectrum
#    Aacc : float
#        Acceleration noise parameter
#    Ap : float
#        Position noise parameter
#    key : int or jax.random.PRNGKey
#        Random seed or JAX random key
#        
#    Returns
#    -------
#    tuple
#        Signal and noise components for each channel
#    """
#    # Convert integer key to PRNGKey if needed
#    if np.ndim(key) == 0:
#        key = jr.PRNGKey(key)
#        
#    N_chunks = defaultlisaparams['N_chunks']
#    N_freq = frequencies.shape[0]
#    
#    # Compute signal and noise standard deviations
#    sigma_s   = jnp.sqrt(h2Omega_GW)
#    sigma_n_A = jnp.sqrt(h2Omega_noise_AA(frequencies, Aacc, Ap))
#    sigma_n_T = jnp.sqrt(h2Omega_noise_TT(frequencies, Aacc, Ap))
#
#    # Generate all random numbers in one call
#    noise_shape = (N_chunks, 8, N_freq)
#    all_noise = jr.normal(key, shape=noise_shape)
#
#    def single_chunk(noise):
#        G1, G2, G3, G4, G5, G6, G7, G8 = noise
#        S   = 0.5 * ((G1 * sigma_s  )**2 + (G2 * sigma_s  )**2)
#        N_A = 0.5 * ((G3 * sigma_n_A)**2 + (G4 * sigma_n_A)**2)
#        N_E = 0.5 * ((G5 * sigma_n_A)**2 + (G6 * sigma_n_A)**2)
#        N_T = 0.5 * ((G7 * sigma_n_T)**2 + (G8 * sigma_n_T)**2)
#        return S, N_A, N_E, N_T
#
#    S_all, N_all_A, N_all_E, N_all_T = vmap(single_chunk)(all_noise)
#
#    result = (
#        jnp.mean(S_all, axis=0),
#        jnp.mean(N_all_A, axis=0),
#        jnp.mean(N_all_E, axis=0),
#        jnp.mean(N_all_T, axis=0)
#    )
#
#    return result






@jit
def generate_lisa_mock_data(
    frequencies,
    h2Omega_GW_total,
    A_acc,
    A_p,
    key=1,
    N_chunks=defaultlisaparams['N_chunks'],
):
    """
    Fast, exact-in-law generator for chunk-averaged spectra.

    Returns (same format as slow version):
        (S_bar, N_A_bar, N_E_bar, N_T_bar)
    where each is shape (N_freq,).

    Uses:  S_bar   = Omega_GW * Gamma(Nc, 1)/Nc
           N_A_bar = Omega_AA * Gamma(Nc, 1)/Nc
           N_E_bar = Omega_AA * Gamma(Nc, 1)/Nc
           N_T_bar = Omega_TT * Gamma(Nc, 1)/Nc
    with the *same* Gamma field used for S across channels (to preserve cross-channel covariance).
    """
    # PRNG
    if isinstance(key, int):
        key = jr.PRNGKey(key)

    Nc = N_chunks #defaultlisaparams['N_chunks']

    # Ensure consistent dtype/device
    f = jnp.asarray(frequencies)
    dtype = f.dtype

    # Total GW signal component (all signal pieces should already be included)
    h2Omega_GW_total = jnp.asarray(h2Omega_GW_total, dtype)

    # Variances (σ^2) per frequency
    sigma_s2 = h2Omega_GW_total
    sigma_A2 = jnp.asarray(h2Omega_noise_AA(f, A_acc, A_p), dtype)
    sigma_E2 = sigma_A2
    sigma_T2 = jnp.asarray(h2Omega_noise_TT(f, A_acc, A_p), dtype)

    # Independent Gamma fields (shape=Nc, scale=1), averaged by /Nc
    kS, kA, kE, kT = jr.split(key, 4)
    a = jnp.asarray(Nc, dtype=dtype)  # shape parameter
    invNc = jnp.asarray(1.0 / Nc, dtype=dtype)

    U_S = jr.gamma(kS, a=a, shape=f.shape, dtype=dtype) * invNc          # shared signal factor
    U_A = jr.gamma(kA, a=a, shape=f.shape, dtype=dtype) * invNc          # AA noise factor
    U_E = jr.gamma(kE, a=a, shape=f.shape, dtype=dtype) * invNc          # EE noise factor
    U_T = jr.gamma(kT, a=a, shape=f.shape, dtype=dtype) * invNc          # TT noise factor

    # Chunk-averaged components (match slow function's return ordering)
    S_bar   = sigma_s2 * U_S
    N_A_bar = sigma_A2 * U_A
    N_E_bar = sigma_E2 * U_E
    N_T_bar = sigma_T2 * U_T

    return S_bar, N_A_bar, N_E_bar, N_T_bar


def h2Omega_GW_foreground_and_background(
    *,
    f_GeV,
    logGmu=None,
    logA_pl=None,
    alpha_pl=None,
    logA_Gal=None,
    logA_ExB=None,
    logA_ExWB=None,
    logA_EMRI=None,
    logA_MBHB=None,
    special_arg_dict=None,
):
    """
    Compute the total h^2 Omega from all background and foreground components.
    
    All arguments are keyword-only; components are included only when their parameters
    are provided:
      - Cosmic strings: included if ``logGmu`` is not None. Uses Peera spectra or the
        original spectra based on ``special_arg_dict``:
           * special_arg_dict['lPeerasData'] (bool, default True)
           * special_arg_dict['VOS_or_BOS']  (\"BOS\"/\"VOS\", optional)
           * special_arg_dict['f_0']         (float, optional, Hz pivot for PL background)
    - Power-law GW background: included if both ``logA_pl`` and ``alpha_pl`` are not None.
      - Foregrounds:
          * Galactic foreground: included if ``logA_Gal`` is not None.
          * Extra-galactic Binaries: included if ``logA_ExB`` is not None.
          * Extra-galactic White Dwarfs: included if ``logA_ExWB`` is not None.
          * Perego EMRI: included if ``logA_EMRI`` is not None.
          * Perego MBHB: included if ``logA_MBHB`` is not None.
    """

    # Ensure consistent dtype/device - convert all inputs to JAX arrays
    f_GeV = jnp.asarray(f_GeV)
    dtype = f_GeV.dtype

    HS_or_LS = None
    VOS_or_BOS = None
    f_0_pl = 3e-3
    if special_arg_dict is not None:
        HS_or_LS = special_arg_dict.get("HS_or_LS", None)
        VOS_or_BOS = special_arg_dict.get("VOS_or_BOS", None)
        f_0_pl = special_arg_dict.get("f_0", f_0_pl)
    
    # 1) Cosmic-string contribution (optional, only if logGmu is provided)
    h2Omega_string = 0.0
    if logGmu is not None:
        logGmu = jnp.asarray(logGmu, dtype=dtype)
        h2Omega_string = h2Omega_Cosmic_Strings_Peera_Gmu(f_GeV, Gmu=10.0**logGmu, VOS_or_BOS=VOS_or_BOS)
    
    # 2) Power-law GW contribution (optional). Keep log-parameters in their
    # native space until the physical background amplitude is needed here.
    h2Omega_pl = 0.0
    if (logA_pl is not None) and (alpha_pl is not None):
        logA_pl_arr = jnp.asarray(logA_pl, dtype=dtype)
        A_pl = 10.0**logA_pl_arr
        # Keep A_pl/alpha_pl as Python scalars so they are hashable static args
        h2Omega_pl = Omega_GW_power_law(
            f_GeV,
            A=float(A_pl),
            alpha=float(alpha_pl),
            f_0=float(f_0_pl),
        )
    
    # 3) Foreground contributions 
    h2Omega_fg = 0.0
    # Galactic foreground
    if logA_Gal is not None:
        logA_Gal_arr = jnp.asarray(logA_Gal, dtype=dtype)
        h2Omega_fg = h2Omega_fg + h2Omega_Gal(f_GeV, logAGal=logA_Gal_arr, T_obs_yr=defaultlisaparams['T_obs'])
    # Extra-galactic binaries
    if logA_ExB is not None:
        logA_ExB_arr = jnp.asarray(logA_ExB, dtype=dtype)
        h2Omega_fg = h2Omega_fg + h2Omega_ExB(f_GeV, logAExB=logA_ExB_arr)
    # Extra-galactic white dwarfs
    if logA_ExWB is not None:
        logA_ExWB_arr = jnp.asarray(logA_ExWB, dtype=dtype)
        h2Omega_fg = h2Omega_fg + h2Omega_ExWB(f_GeV, logAExWB=logA_ExWB_arr)
    if logA_EMRI is not None:
        if HS_or_LS is None:
            raise ValueError("special_arg_dict['HS_or_LS'] must be provided when logA_EMRI is used.")
        logA_EMRI_arr = jnp.asarray(logA_EMRI, dtype=dtype)
        A_EMRI_arr = 10.0**logA_EMRI_arr
        h2Omega_fg = h2Omega_fg + h2Omega_Perego(f_GeV, component="EMRI", HS_or_LS=HS_or_LS, A=A_EMRI_arr)
    if logA_MBHB is not None:
        if HS_or_LS is None:
            raise ValueError("special_arg_dict['HS_or_LS'] must be provided when logA_MBHB is used.")
        logA_MBHB_arr = jnp.asarray(logA_MBHB, dtype=dtype)
        A_MBHB_arr = 10.0**logA_MBHB_arr
        h2Omega_fg = h2Omega_fg + h2Omega_Perego(f_GeV, component="MBHB", HS_or_LS=HS_or_LS, A=A_MBHB_arr)
    
    # Total signal from all GW components
    h2Omega_GW_signal_to_return = h2Omega_string + h2Omega_pl + h2Omega_fg
    
    return h2Omega_GW_signal_to_return


def total_log10_h2Omega(
    *,
    f_GeV,
    logGmu=None,
    A_acc=None,
    A_p=None,
    logA_pl=None,
    alpha_pl=None,
    logA_Gal=None,
    logA_ExB=None,
    logA_ExWB=None,
    logA_EMRI=None,
    logA_MBHB=None,
    special_arg_dict=None,
):
    """
    Compute the total h^2 Omega for the AA, EE and TT channels from all included components.
    
    All arguments are keyword-only; components are included only when their parameters
    are provided:
      - Cosmic strings: included if ``logGmu`` is not None. Uses Peera spectra or the
        original spectra based on ``special_arg_dict``:
           * special_arg_dict['lPeerasData'] (bool, default True)
           * special_arg_dict['VOS_or_BOS']  (\"BOS\"/\"VOS\", optional)
      - LISA noise: always included and controlled by ``A_acc`` and ``A_p``.
      - ``special_arg_dict`` may also include ``f_0`` (float, Hz pivot for the PL background).
    - Power-law GW background: included if both ``logA_pl`` and ``alpha_pl`` are not None.
      - Foregrounds:
          * Galactic foreground: included if ``logA_Gal`` is not None.
          * Extra-galactic Binaries: included if ``logA_ExB`` is not None.
          * Extra-galactic White Dwarfs: included if ``logA_ExWB`` is not None.
          * Perego EMRI: included if ``logA_EMRI`` is not None.
          * Perego MBHB: included if ``logA_MBHB`` is not None.
    """

    # Basic validation for noise parameters
    if A_acc is None or A_p is None:
        raise ValueError("total_log10_h2Omega requires A_acc and A_p for LISA noise.")

    # Ensure consistent dtype/device - convert all inputs to JAX arrays
    f_GeV = jnp.asarray(f_GeV)
    dtype = f_GeV.dtype
    A_acc = jnp.asarray(A_acc, dtype=dtype)
    A_p   = jnp.asarray(A_p,   dtype=dtype)

    HS_or_LS = None
    VOS_or_BOS = None
    if special_arg_dict is not None:
        HS_or_LS = special_arg_dict.get("HS_or_LS", None)
        VOS_or_BOS = special_arg_dict.get("VOS_or_BOS", None)
    
    # 1) Noise-only contribution (always present)
    h2Omega_noise_AAandEE = h2Omega_noise_AA(f_GeV, A_acc, A_p)
    h2Omega_noise_TT_val  = h2Omega_noise_TT(f_GeV, A_acc, A_p)
    
    ## 2) Cosmic-string contribution (optional, only if logGmu is provided)
    #h2Omega_string = 0.0
    #if logGmu is not None:
    #    logGmu = jnp.asarray(logGmu, dtype=dtype)
    #    h2Omega_string = h2Omega_Cosmic_Strings_Peera_Gmu(f_GeV, Gmu=10.0**logGmu, VOS_or_BOS=VOS_or_BOS)
    #
    ## 3) Power-law GW contribution (optional, only if both logA_pl and alpha_pl are provided)
    #h2Omega_pl = 0.0
    #if (A_pl is not None) and (alpha_pl is not None):
    #    # Keep A_pl/alpha_pl as Python scalars so they are hashable static args
    #    h2Omega_pl = Omega_GW_power_law(f_GeV, A=float(A_pl), alpha=float(alpha_pl))
    #
    ## 4) Foreground contributions 
    #h2Omega_fg = 0.0
    ## Galactic foreground
    #if logA_Gal is not None:
    #    logA_Gal_arr = jnp.asarray(logA_Gal, dtype=dtype)
    #    h2Omega_fg = h2Omega_fg + h2Omega_Gal(f_GeV, logAGal=logA_Gal_arr, T_obs_yr=defaultlisaparams['T_obs'])
    ## Extra-galactic binaries
    #if logA_ExB is not None:
    #    logA_ExB_arr = jnp.asarray(logA_ExB, dtype=dtype)
    #    h2Omega_fg = h2Omega_fg + h2Omega_ExB(f_GeV, logAExB=logA_ExB_arr)
    ## Extra-galactic white dwarfs
    #if logA_ExWB is not None:
    #    logA_ExWB_arr = jnp.asarray(logA_ExWB, dtype=dtype)
    #    h2Omega_fg = h2Omega_fg + h2Omega_ExWB(f_GeV, logAExWB=logA_ExWB_arr)
    #if logA_EMRI is not None:
    #    if HS_or_LS is None:
    #        raise ValueError("special_arg_dict['HS_or_LS'] must be provided when logA_EMRI is used.")
    #    logA_EMRI_arr = jnp.asarray(logA_EMRI, dtype=dtype)
    #    A_EMRI_arr = 10.0**logA_EMRI_arr
    #    h2Omega_fg = h2Omega_fg + h2Omega_Perego(f_GeV, component="EMRI", HS_or_LS=HS_or_LS, A=A_EMRI_arr)
    #if logA_MBHB is not None:
    #    if HS_or_LS is None:
    #        raise ValueError("special_arg_dict['HS_or_LS'] must be provided when logA_MBHB is used.")
    #    logA_MBHB_arr = jnp.asarray(logA_MBHB, dtype=dtype)
    #    A_MBHB_arr = 10.0**logA_MBHB_arr
    #    h2Omega_fg = h2Omega_fg + h2Omega_Perego(f_GeV, component="MBHB", HS_or_LS=HS_or_LS, A=A_MBHB_arr)

    h2Omega_GW_total_signal_data = h2Omega_GW_foreground_and_background(
        f_GeV = f_GeV,
        logGmu=logGmu,
        logA_pl=logA_pl,
        alpha_pl=alpha_pl,
        logA_Gal=logA_Gal,
        logA_ExB=logA_ExB,
        logA_ExWB=logA_ExWB,
        logA_EMRI=logA_EMRI,
        logA_MBHB=logA_MBHB,
        special_arg_dict=special_arg_dict,
    )
    
    # Total AA/EE and TT backgrounds: noise + all GW components
    h2Omega_total_AAandEE = h2Omega_noise_AAandEE + h2Omega_GW_total_signal_data
    h2Omega_total_TT      = h2Omega_noise_TT_val  + h2Omega_GW_total_signal_data
    
    log10_h2Omega_AAandEE = jnp.log10(h2Omega_total_AAandEE)
    log10_h2Omega_TT      = jnp.log10(h2Omega_total_TT)
    return log10_h2Omega_AAandEE, log10_h2Omega_TT



def coarse_grain_lisa_data_all_channels(frequencies, D_AA, D_EE, D_TT, A_acc, A_p):
    import time
    N_macro = defaultlisaparams['N_macro']
    f_coarse_def = defaultlisaparams['f_coarse']  # Use f_coarse definition from defaultlisaparams
    start_total = time.time()
    
    GeV_to_Hz = units['GeV_to_Hz']
    f_Hz = frequencies * GeV_to_Hz

    # Use NumPy to get static slicing (not traced)
    f_Hz_np = np.asarray(f_Hz)
    
    # Define frequency ranges based on f_coarse definition
    f_coarse_min = f_coarse_def[0] * GeV_to_Hz  # Convert to Hz
    f_coarse_max = f_coarse_def[-1] * GeV_to_Hz  # Convert to Hz
    
    idx_fine = np.where((f_Hz_np >= 3e-5) & (f_Hz_np < f_coarse_min))[0]
    idx_coarse = np.where((f_Hz_np >= f_coarse_min) & (f_Hz_np <= f_coarse_max))[0]

    f_fine = frequencies[idx_fine]
    D_AA_fine = D_AA[idx_fine]
    D_EE_fine = D_EE[idx_fine]
    D_TT_fine = D_TT[idx_fine]

    f_coarse = frequencies[idx_coarse]
    f_Hz_coarse = f_coarse * GeV_to_Hz
    D_AA_coarse = D_AA[idx_coarse]
    D_EE_coarse = D_EE[idx_coarse]
    D_TT_coarse = D_TT[idx_coarse]

    start_noise = time.time()
    h2Omega_AA_vals = h2Omega_noise_AA(f_coarse, A_acc, A_p)
    h2Omega_EE_vals = h2Omega_AA_vals #Omega_noise_EE(f_coarse, A_acc=A_acc, A_p=A_p)
    h2Omega_TT_vals = h2Omega_noise_TT(f_coarse, A_acc, A_p)
    end_noise = time.time()
    #print(f"    Coarse graining: Noise computation: {end_noise - start_noise:.3f}s")

    start_binning = time.time()
    bin_edges_Hz = jnp.logspace(jnp.log10(f_coarse_min), jnp.log10(f_coarse_max), N_macro + 1)

    f_macro, D_AA_macro, D_EE_macro, D_TT_macro = _coarse_grain_binned(
        bin_edges_Hz,
        f_coarse, f_Hz_coarse,
        D_AA_coarse, D_EE_coarse, D_TT_coarse,
        h2Omega_AA_vals, h2Omega_EE_vals, h2Omega_TT_vals
    )

    end_binning = time.time()
    #print(f"    Coarse graining: Binning computation: {end_binning - start_binning:.3f}s")

    # Combine fine and coarse results
    f_out = jnp.concatenate([f_fine, f_macro])
    D_AA_out = jnp.concatenate([D_AA_fine, D_AA_macro])
    D_EE_out = jnp.concatenate([D_EE_fine, D_EE_macro])
    D_TT_out = jnp.concatenate([D_TT_fine, D_TT_macro])
    
    end_total = time.time()
    #print(f"    Coarse graining: Total time: {end_total - start_total:.3f}s")

    return f_out, D_AA_out, D_EE_out, D_TT_out


def _coarse_grain_binned(bin_edges_Hz, f_coarse, f_Hz_coarse,
                         D_AA_coarse, D_EE_coarse, D_TT_coarse,
                         Ω_AA_vals, Ω_EE_vals, Ω_TT_vals):

    N_macro     = bin_edges_Hz.size - 1
    bin_indices = jnp.digitize(f_Hz_coarse, bin_edges_Hz) - 1
    valid_mask  = (bin_indices >= 0) & (bin_indices < N_macro)

    def weighted_average(data, Ω):
        w   = jnp.where(valid_mask, 1.0 / Ω, 0.0)
        xw  = jnp.where(valid_mask, data * w, 0.0)
        wsum   = jnp.bincount(bin_indices, weights=w,  length=N_macro)
        xwsum  = jnp.bincount(bin_indices, weights=xw, length=N_macro)
        return jnp.where(wsum > 0, xwsum / wsum, 0.0), wsum

    f_macro,    counts = weighted_average(f_coarse, Ω_AA_vals)   # use AA for weights
    D_AA_macro, _      = weighted_average(D_AA_coarse, Ω_AA_vals)
    D_EE_macro, _      = weighted_average(D_EE_coarse, Ω_EE_vals)
    D_TT_macro, _      = weighted_average(D_TT_coarse, Ω_TT_vals)

    # --- drop empty bins ----------------------------------------------------
    keep          = counts > 0                           # boolean mask
    return (f_macro[keep], D_AA_macro[keep],
            D_EE_macro[keep], D_TT_macro[keep])




def generate_lisa_mock_data_matrix(parameter_samples=None, key=1, verbose=False, special_arg_dict = None):
    """
    Generate LISA mock data for a list of parameter combinations.
    
    Parameters
    ----------
    parameter_samples : list[dict]
        List of parameter dictionaries, each containing parameter values
        e.g. [{'logGmu': -14, 'A_p': 16, 'A_acc': 2}, ...]
    key : int or jax.random.PRNGKey
        Random seed or JAX random key
    verbose : bool
        If True, print timing information
        
    Returns
    -------
    frequencies_lisa : array
        Array of frequencies
    f_coarse : array
        Array of coarse-grained frequencies
    result_list : list
        List of dictionaries with keys 'prior', 'signal', 'noise', 'data'
    """
    if parameter_samples is None:
        raise ValueError("parameter_samples must be provided")
    
    # Convert integer key to PRNGKey if needed
    if isinstance(key, int):
        key = jr.PRNGKey(key)
        
    import time
    start_total = time.time()
    
    #frequencies_few = np.logspace(np.log10(0.999*3*10**-5), np.log10(1.001*5*10**-1), 1000) * units['Hz_to_GeV']
    frequencies_lisa = defaultlisaparams['f_lisa']
    f_coarse = defaultlisaparams['f_coarse']  # Define f_coarse outside the loop
    cosmicstringparams = canonical_string_params_VOS.copy()
    result_list = []
    
    # Process all parameter combinations
    start_loop = time.time()
    if verbose: print("Processing parameter combinations sequentially...")
    
    # Split the key for each parameter combination (parameter_samples is a list of dictionaries of the form {'logGmu': -15, 'A_p': 16, 'A_acc': 2}, etc.)
    total_combinations = len(parameter_samples)
    keys = jr.split(key, total_combinations)
    for idx, params in enumerate(parameter_samples):
        start_combination = time.time()

        def _get_param(primary_name, legacy_name=None):
            value = params.get(primary_name, None)
            if value is None and legacy_name is not None:
                value = params.get(legacy_name, None)
            return value
        
        # Core noise parameters (always used for noise)
        A_p   = _get_param('A_p', 'Ap')
        A_acc = _get_param('A_acc', 'Aacc')

        # String parameters (optional)
        logGmu = params.get('logGmu', None)

        # Power-law GW background parameters (optional). Keep the amplitude
        # in log-space here and convert only where the physical model needs it.
        logA_pl = _get_param('logA_pl', 'logApl')
        alpha_pl = _get_param('alpha_pl', 'alphapl')

        # Foreground parameters (optional)
        logA_Gal  = _get_param('logA_Gal', 'logAGal')
        logA_ExB  = _get_param('logA_ExB', 'logAExB')
        logA_ExWB = _get_param('logA_ExWB', 'logAExWB')
        logA_EMRI = params.get('logA_EMRI', None)
        logA_MBHB = params.get('logA_MBHB', None)
        
        # Total GW signal from all requested foreground and background components.
        h2Omega_GW_total = h2Omega_GW_foreground_and_background(
            f_GeV=frequencies_lisa,
            logGmu=logGmu,
            logA_pl=logA_pl,
            alpha_pl=alpha_pl,
            logA_Gal=logA_Gal,
            logA_ExB=logA_ExB,
            logA_ExWB=logA_ExWB,
            logA_EMRI=logA_EMRI,
            logA_MBHB=logA_MBHB,
            special_arg_dict=special_arg_dict,
        )
        
        # Use the pre-split key for this combination
        current_key = keys[idx]
        
        # Generate mock data with total GW background and LISA noise (A_p, A_acc)
        start_mock_data = time.time()
        mock_data = generate_lisa_mock_data(
            frequencies_lisa,
            h2Omega_GW_total,
            A_acc,
            A_p,
            key=current_key,
        )
        S, N_AA, N_EE, N_TT = mock_data[0],                mock_data[1],                mock_data[2],                mock_data[3]
        D_AA, D_EE, D_TT    = S            + N_AA,         S            + N_EE,         S            + N_TT
        #D_AA, D_EE, D_TT    = mock_data[0] + mock_data[1], mock_data[0] + mock_data[2], mock_data[0] + mock_data[3]
        #D_AAEE = 0.5*(mock_data[0] + mock_data[2])  
        #D_AAEE = 10**(0.5*(np.log10(mock_data[0]) + np.log10(mock_data[2]))  )
        end_mock_data = time.time()
        
        start_coarse = time.time()
        f_coarse, D_AA_coarse, D_EE_coarse, D_TT_coarse = coarse_grain_lisa_data_all_channels(
            frequencies_lisa, D_AA, D_EE, D_TT, A_acc, A_p
        )
        end_coarse = time.time()
        
        end_combination = time.time()
        mock_time = end_mock_data - start_mock_data
        coarse_time = end_coarse - start_coarse
        total_time = end_combination - start_combination
        if verbose:
            print(
                f"    Combination {idx+1}/{total_combinations} "
                f"(logGmu={logGmu}, logA_pl={logA_pl}, alpha_pl={alpha_pl}, A_p={A_p}, A_acc={A_acc}): "
                f"{total_time:.2f} seconds"
            )
            print(f"      - Mock data generation: {mock_time:.2f}s")
            print(f"      - Coarse graining: {coarse_time:.2f}s")
            print(f"      - Other overhead: {total_time - mock_time - coarse_time:.2f}s")
        
        result_list.append({
            #'prior': {'Gmu': Gmu, 'A_p': A_p, 'A_acc': A_acc},
            'signal': S,
            #'noise': {'N_AA': N_AA/LCDM['h']**2, 'N_EE': N_EE/LCDM['h']**2, 'N_TT': N_TT/LCDM['h']**2},
            #'data': {'D_AA': D_AA_coarse/LCDM['h']**2, 'D_EE': D_EE_coarse/LCDM['h']**2, 'D_TT': D_TT_coarse/LCDM['h']**2}
            'noise': {'N_AA': N_AA, 'N_EE': N_EE, 'N_TT': N_TT},
            'data': {'D_AA': D_AA_coarse, 'D_EE': D_EE_coarse, 'D_TT': D_TT_coarse}
            #'data': {'D_AAEE': D_AAEE_coarse, 'D_TT': D_TT_coarse}

        })
    
    end_total = time.time()
    if verbose: print(f"Total processing time: {end_total - start_total:.2f} seconds")
    
    return frequencies_lisa, f_coarse, result_list
