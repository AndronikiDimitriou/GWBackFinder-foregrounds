"""Signal, foreground and noise spectra for a model and parameter set."""
import numpy as np

from .models import get_model
from .physics import lisa
from .physics.GW_foregrounds import (
    h2Omega_ExB,
    h2Omega_ExWB,
    h2Omega_Gal,
    h2Omega_Perego,
)

units = lisa.units
N_CHUNKS = lisa.defaultlisaparams["N_chunks"]
T_OBS_YR = lisa.defaultlisaparams["T_obs"]


def GeV_to_Hz(f_GeV):
    return np.asarray(f_GeV) * units["GeV_to_Hz"]


def Hz_to_GeV(f_Hz):
    return np.asarray(f_Hz) * units["Hz_to_GeV"]


def total_signal(f_GeV, params, model):
    """h^2 Omega_GW of strings + all foregrounds of ``model`` (no noise)."""
    model = get_model(model)
    p = {**model.fixed, **params}
    kwargs = {k: p.get(k) for k in ("logGmu", "logA_Gal", "logA_ExB", "logA_ExWB",
                                    "logA_EMRI", "logA_MBHB")}
    # only pass the components this model contains
    present = set(model.param_names) | set(model.fixed)
    kwargs = {k: (v if k in present else None) for k, v in kwargs.items()}
    return lisa.h2Omega_GW_foreground_and_background(
        f_GeV=f_GeV, special_arg_dict=model.special_arg_dict, **kwargs
    )


def noise(f_GeV, A_acc, A_p):
    """(AA, TT) instrument noise in h^2 Omega. EE = AA."""
    return (lisa.h2Omega_noise_AA(f_GeV, A_acc, A_p),
            lisa.h2Omega_noise_TT(f_GeV, A_acc, A_p))


def model_prediction(f_GeV, params, model):
    """Expected (AA, EE, TT) spectra = noise + signal, as NumPy arrays."""
    sig = np.asarray(total_signal(f_GeV, params, model))
    n_AA, n_TT = noise(f_GeV, params["A_acc"], params["A_p"])
    AA = np.asarray(n_AA) + sig
    return AA, AA, np.asarray(n_TT) + sig


def component_spectra(f_GeV, params, model):
    """Each component separately: {'string', 'Gal', 'ExB', 'ExWB', 'EMRI', 'MBHB'} subset.

    Their sum equals ``total_signal`` (checked in the tests).
    """
    model = get_model(model)
    p = {**model.fixed, **params}
    hs = model.special_arg_dict.get("HS_or_LS")
    f = np.asarray(f_GeV)
    out = {}
    for comp in model.components:
        if comp == "string":
            val = lisa.h2Omega_GW_foreground_and_background(
                f_GeV=f, logGmu=p["logGmu"],
                special_arg_dict={"VOS_or_BOS": model.special_arg_dict.get("VOS_or_BOS")})
        elif comp == "Gal":
            val = h2Omega_Gal(f, p["logA_Gal"], T_obs_yr=T_OBS_YR)
        elif comp == "ExB":
            val = h2Omega_ExB(f, p["logA_ExB"])
        elif comp == "ExWB":
            val = h2Omega_ExWB(f, p["logA_ExWB"])
        elif comp in ("EMRI", "MBHB"):
            val = h2Omega_Perego(f, component=comp, HS_or_LS=hs, A=10.0 ** p[f"logA_{comp}"])
        else:  # pragma: no cover
            raise ValueError(comp)
        out[comp] = np.asarray(val)
    return out
