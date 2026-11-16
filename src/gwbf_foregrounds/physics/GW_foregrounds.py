import os
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit

from .cosmo import units, LCDM

def _get_lisa_t_obs():
    from .lisa import defaultlisaparams
    return defaultlisaparams['T_obs']

@jit
def h2Omega_Gal(f_GeV, logAGal, T_obs_yr=None):
    """
    Galactic foreground contribution to h^2 Omega_GW.
    
    Parameters
    ----------
    f : array_like
        Frequency in GeV
    logAGal : float
        10 Logarithmic amplitude parameter for galactic foreground
    T_obs_yr : float, optional
        Observation time in years (default: `lisa.defaultlisaparams['T_obs']`)
    
    Returns
    -------
    array_like
        h^2 Omega_Gal(f)
    """
    if T_obs_yr is None:
        T_obs_yr = _get_lisa_t_obs()

    AGal = 10**logAGal
    # Convert frequency from GeV to Hz
    f_Hz = f_GeV / units['Hz_to_GeV']
    
    # Constants
    a1 = -0.15
    b1 = -2.72
    ak = -0.37
    bk = -2.49
    nu = 1.56
    f2 = 6.7e-4  # Hz
    
    # Calculate f1 and f_knee in Hz
    log10_f1 = a1 * jnp.log10(T_obs_yr) + b1
    log10_f_knee = ak * jnp.log10(T_obs_yr) + bk
    f1 = 10**log10_f1  # Hz
    f_knee = 10**log10_f_knee  # Hz
    
    # Calculate h^2 Omega_Gal
    term1 = 0.5 * (f_Hz)**(2/3)
    term2 = 1 + jnp.tanh((f_knee - f_Hz) / f2)
    term3 = jnp.exp(-(f_Hz / f1)**nu)
    
    return AGal * term1 * term2 * term3


@jit
def h2Omega_ExB(f_GeV, logAExB):
    """
    Extra-galactic foreground I: BH & NS binaries contribution to h^2 Omega_GW.
    
    Parameters
    ----------
    f : array_like
        Frequency in GeV
    logAExB : float
        10 Logarithmic amplitude parameter for extra-galactic BH & NS binaries
    
    Returns
    -------
    array_like
        h^2 Omega_ExB(f)
    """
    AExB = 10**logAExB
    # Convert frequency from GeV to Hz
    f_Hz = f_GeV / units['Hz_to_GeV']
    
    # Reference frequency: 10^-3 Hz
    f_ref = 1e-3  # Hz
    
    return AExB * (f_Hz / f_ref)**(2/3)


@jit
def h2Omega_ExWB(f_GeV, logAExWB):
    """
    Extra-galactic foreground II: White dwarf binaries contribution to h^2 Omega_GW.
    
    Parameters
    ----------
    f : array_like
        Frequency in GeV
    logAExWB : float
        10 Logarithmic amplitude parameter for extra-galactic white dwarf binaries
    
    Returns
    -------
    array_like
        h^2 Omega_ExWB(f)
    """
    AExWB = 10**logAExWB
    # Convert frequency from GeV to Hz
    f_Hz = f_GeV / units['Hz_to_GeV']
    
    # Constants
    B = 1.54e4  # Hz^-3
    f_hat = 7.2e-3  # Hz
    
    # Calculate h^2 Omega_ExWB
    term1 = (f_Hz / f_hat)**0.741
    term2 = (1 + (f_Hz / f_hat)**4.15)**(-0.255)
    term3 = jnp.exp(-B * f_Hz**3)
    
    return AExWB * term1 * term2 * term3


@jit
def h2Omega_total_foreground(f_GeV, logAGal, logAExB, logAExWB, T_obs_yr):
    """
    Total foreground contribution: Galactic + Extra-galactic BH/NS + Extra-galactic WD.
    
    Parameters
    ----------
    f_GeV : array_like
        Frequency in GeV
    logAGal : float
        10 Logarithmic amplitude parameter for galactic foreground
    logAExB : float
        10 Logarithmic amplitude parameter for extra-galactic BH & NS binaries
    logAExWB : float
        10 Logarithmic amplitude parameter for extra-galactic white dwarf binaries
    T_obs_yr : float, optional
        Observation time in years (default: 3.0)
    
    Returns
    -------
    array_like
        h^2 Omega_foreground(f)
    """
    return h2Omega_Gal(f_GeV, logAGal, T_obs_yr) + h2Omega_ExB(f_GeV, logAExB) + h2Omega_ExWB(f_GeV, logAExWB)


# --- Perego tabulated foregrounds ---
_perego_interp_cache = {}
_perego_filename_map = {
    ("GB", "HS"): "GWB_GB_LISA_HSscenario.dat",
    ("GB", "LS"): "GWB_GB_LISA_LSscenario.dat",
    ("EXTRAGALDWD", "HS"): "GWB_extragalDWD_LISA_HSscenario.dat",
    ("EXTRAGALDWD", "LS"): "GWB_extragalDWD_LISA_LSscenario.dat",
    ("DWD", "HS"): "GWB_extragalDWD_LISA_HSscenario.dat",
    ("DWD", "LS"): "GWB_extragalDWD_LISA_LSscenario.dat",
    ("EMRI", "HS"): "GWB_EMRI_LISA_HSscenario.dat",
    ("EMRI", "LS"): "GWB_EMRI_LISA_LSscenario.dat",
    ("MBHB", "HS"): "GWB_MBHB_HS_LISA_HSscenario.dat",
    ("MBHB", "LS"): "GWB_MBHB_LS_LISA_LSscenario.dat",
    ("SOBBH", "HS"): "GWB_SOBBH_LISA_HSscenario.dat",
    ("SOBBH", "LS"): "GWB_SOBBH_LISA_LSscenario.dat",
}


def _resolve_perego_filename(component, HS_or_LS):
    """
    Resolve a Perego component/scenario pair to the corresponding filename.
    """
    if component is None:
        raise ValueError("component must be provided")
    if HS_or_LS is None:
        raise ValueError("HS_or_LS must be provided")

    component_key = str(component).strip().upper()
    scenario_key = str(HS_or_LS).strip().upper()
    key = (component_key, scenario_key)

    if key not in _perego_filename_map:
        valid = ", ".join(
            f"({comp}, {scenario})" for comp, scenario in sorted(_perego_filename_map)
        )
        raise ValueError(
            f"Unsupported Perego component/scenario pair {key!r}. "
            f"Valid pairs are: {valid}"
        )

    return _perego_filename_map[key]

def _load_h2_perego_curve(path):
    """
    Load one Perego foreground file and return frequency and h^2 Omega arrays.

    Parameters
    ----------
    path : str
        Absolute path to one Perego ``.dat`` file.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(f_Hz, h2Omega)`` arrays.
    """
    arr = np.loadtxt(path)
    f_Hz = np.asarray(arr[:, 0], dtype=float)
    h2Omega = np.asarray(LCDM["h"]**2 * arr[:, 3], dtype=float)

    mask = (f_Hz > 0.0) & (h2Omega > 0.0)
    return f_Hz[mask], h2Omega[mask]


def _build_interpolator_from_perego_curve(f_Hz, h2Omega):
    """
    Build a fast, JIT-compiled log-log interpolator:
        h2Omega_Perego(f_GeV) -> h^2 Omega_GW
    """
    logf_grid_j = jnp.asarray(np.log10(f_Hz))
    logh2_grid_j = jnp.asarray(np.log10(h2Omega))

    @jax.jit
    def h2Omega_interp(f_GeV):
        f_Hz_eval = jnp.asarray(f_GeV) * units["GeV_to_Hz"]
        logf_eval = jnp.log10(f_Hz_eval)
        logh2_eval = jnp.interp(logf_eval, logf_grid_j, logh2_grid_j)
        return 10.0**logh2_eval

    return h2Omega_interp


def _h2Omega_Perego_func(component=None, HS_or_LS=None):
    """
    Return a cached function with the simple signature
        h2Omega_Perego(f_GeV)

    Parameters
    ----------
    component : str, optional
        Perego foreground component.
    HS_or_LS : str, optional
        Scenario selector: ``"HS"`` or ``"LS"``.

    Returns
    -------
    function
        Cached function of the form ``func(f_GeV)``.
    """
    global _perego_interp_cache

    filename = _resolve_perego_filename(component, HS_or_LS)
    key = (str(component).strip().upper(), str(HS_or_LS).strip().upper())

    if isinstance(_perego_interp_cache, dict) and key in _perego_interp_cache:
        return _perego_interp_cache[key]

    _core_dir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(_core_dir, "data", "Perego_GWB", filename)

    f_Hz, h2Omega = _load_h2_perego_curve(path_data)
    h2Omega_interp = _build_interpolator_from_perego_curve(f_Hz, h2Omega)

    if not isinstance(_perego_interp_cache, dict):
        _perego_interp_cache = {}
    _perego_interp_cache[key] = h2Omega_interp

    return h2Omega_interp


def h2Omega_Perego(f_GeV, component=None, HS_or_LS=None, A=1.0):
    """
    Evaluate one Perego tabulated foreground curve with the simple signature
        h2Omega_Perego(f_GeV, component=None, HS_or_LS=None, A=1.0)

    Parameters
    ----------
    f_GeV : array_like
        Frequency in GeV.
    component : str, optional
        Perego foreground component, e.g. ``"GB"``, ``"EMRI"``,
        ``"MBHB"``, ``"SOBBH"``, or ``"extragalDWD"``/``"DWD"``.
    HS_or_LS : str, optional
        Scenario selector: ``"HS"`` or ``"LS"``.
    A : float or array_like, optional
        Overall amplitude rescaling factor. Default is ``1.0``.

    Returns
    -------
    array_like
        Interpolated ``h^2 Omega_GW`` values.
    """
    A = jnp.asarray(A)
    return A * _h2Omega_Perego_func(component, HS_or_LS)(f_GeV)


# --- Pozzo EMRI tabulated foreground ---
_emri_pozzo_interp_cache = None
_emri_pozzo_filename = "emri-mean-foreground.csv"


def _load_h2_emri_pozzo_curve(path):
    """
    Load the Pozzo EMRI mean foreground file.

    Parameters
    ----------
    path : str
        Absolute path to ``emri-mean-foreground.csv``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(f_Hz, h2Omega)`` arrays.
    """
    arr = np.loadtxt(path, delimiter=",", skiprows=2)
    f_Hz = np.asarray(arr[:, 0], dtype=float)
    h2Omega = np.asarray(arr[:, 1], dtype=float)

    mask = (f_Hz > 0.0) & (h2Omega > 0.0)
    return f_Hz[mask], h2Omega[mask]


def _h2Omega_EMRI_Pozzo_func():
    """
    Return the cached Pozzo EMRI foreground interpolator.
    """
    global _emri_pozzo_interp_cache

    if _emri_pozzo_interp_cache is not None:
        return _emri_pozzo_interp_cache

    _core_dir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(_core_dir, "data", _emri_pozzo_filename)

    f_Hz, h2Omega = _load_h2_emri_pozzo_curve(path_data)
    h2Omega_interp = _build_interpolator_from_perego_curve(f_Hz, h2Omega)

    _emri_pozzo_interp_cache = h2Omega_interp
    return h2Omega_interp


def h2Omega_EMRI_Pozzo(f_GeV, A=1.0):
    """
    Evaluate the Pozzo EMRI mean foreground curve.

    Parameters
    ----------
    f_GeV : array_like
        Frequency in GeV.
    A : float or array_like, optional
        Overall amplitude rescaling factor. Default is ``1.0``.

    Returns
    -------
    array_like
        Interpolated ``h^2 Omega_GW`` values.
    """
    A = jnp.asarray(A)
    return A * _h2Omega_EMRI_Pozzo_func()(f_GeV)
