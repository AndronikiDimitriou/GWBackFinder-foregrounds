"""The three foreground models of paper 2.

Each model fixes:
  * the parameter vector the SBI network was trained on (order matters),
  * the settings passed to the physics (``special_arg_dict``),
  * fixed parameters that are not inferred (MBHB amplitude),
  * which spectral components it contains,
  * the prior it was trained with, and the fiducial (injected) values.
"""
from dataclasses import dataclass, field

# Fiducial foreground amplitudes (= defaultlisaparams in physics/lisa.py) and
# their 1-sigma prior widths. Kept as literals so importing this module does not
# load the physics (JAX, response tables).
LOG_A_GAL = -7.84
LOG_A_EXB = -12.38
LOG_A_EXWB = -11.0554
LOG_A_EMRI = 0.0
LOG_A_MBHB = 0.0

FG_SIGMA = {
    "logA_Gal": 0.21,
    "logA_ExB": 0.344,
    "logA_ExWB": 0.17,
    "logA_EMRI": 0.5,
}

NOISE_FIDUCIAL = {"A_p": 15.0, "A_acc": 3.0}

_NOISE_PRIOR = {
    "logGmu": ("uniform", -18.0, -9.0),
    "A_p": ("uniform", 15.0 * 0.8, 15.0 * 1.2),
    "A_acc": ("uniform", 3.0 * 0.8, 3.0 * 1.2),
}

# Display names for each parameter and component.
LATEX = {
    "logGmu": r"$\log_{10} G\mu$",
    "A_p": r"$A_p$",
    "A_acc": r"$A_{\rm acc}$",
    "logA_Gal": r"$\log_{10} A_{\rm WD}$",
    "logA_ExB": r"$\log_{10} A_{\rm SOBHB}$",
    "logA_ExWB": r"$\log_{10} A_{\rm ExWD}$",
    "logA_EMRI": r"$\log_{10} A_{\rm EMRI}$",
}

COMPONENT_STYLE = {
    # key: (colour from the Wong palette, legend label)
    "string": ("#D55E00", "Cosmic strings"),
    "Gal": ("#0072B2", "WDs"),
    "ExB": ("#009E73", "SOBHBs"),
    "ExWB": ("#CC79A7", "ExWDs"),
    "EMRI": ("#E69F00", "EMRIs"),
    "MBHB": ("0.4", "MBHBs (fixed)"),
}


@dataclass(frozen=True)
class Model:
    name: str
    param_names: tuple
    special_arg_dict: dict
    fixed: dict
    components: tuple
    prior: dict
    posterior_file: str          # asset name on the GitHub release
    description: str = ""
    fg_sigma: dict = field(default_factory=dict)

    @property
    def ndim(self):
        return len(self.param_names)

    @property
    def fg_params(self):
        return tuple(p for p in self.param_names if p.startswith("logA_"))

    def fiducial(self, logGmu=None, sign=0):
        """Injected parameters: fiducial foregrounds shifted by ``sign`` x 1 sigma.

        ``sign=-1/0/+1`` gives the low / normal / high foreground cases of the paper.
        Fixed parameters (MBHB) are included; parameters the model does not have are not.
        """
        values = {
            "logA_Gal": LOG_A_GAL,
            "logA_ExB": LOG_A_EXB,
            "logA_ExWB": LOG_A_EXWB,
            "logA_EMRI": LOG_A_EMRI,
        }
        p = dict(NOISE_FIDUCIAL)
        for name in self.fg_params:
            p[name] = values[name] + sign * self.fg_sigma.get(name, FG_SIGMA[name])
        p.update(self.fixed)
        if logGmu is not None:
            p["logGmu"] = float(logGmu)
        return p

    def theta(self, params):
        """dict -> list in the network's parameter order."""
        return [float(params[k]) for k in self.param_names]

    def to_params(self, theta):
        """Parameter vector (network order) -> full dict including fixed parameters."""
        p = {k: float(v) for k, v in zip(self.param_names, theta)}
        p.update(self.fixed)
        return p

    def labels(self, names=None):
        return [LATEX.get(n, n) for n in (names or self.param_names)]


FULL = Model(
    name="full",
    param_names=("logGmu", "A_p", "A_acc", "logA_Gal", "logA_ExB", "logA_ExWB", "logA_EMRI"),
    special_arg_dict={"HS_or_LS": "HS", "VOS_or_BOS": "BOS"},
    fixed={"logA_MBHB": LOG_A_MBHB},
    components=("string", "Gal", "ExB", "ExWB", "EMRI", "MBHB"),
    prior={
        **_NOISE_PRIOR,
        "logA_Gal": ("gaussian", LOG_A_GAL, FG_SIGMA["logA_Gal"]),
        "logA_ExB": ("gaussian", LOG_A_EXB, FG_SIGMA["logA_ExB"]),
        "logA_ExWB": ("gaussian", LOG_A_EXWB, FG_SIGMA["logA_ExWB"]),
        "logA_EMRI": ("gaussian", LOG_A_EMRI, FG_SIGMA["logA_EMRI"]),
    },
    posterior_file="posterior_full.pkl",
    description="Strings + WDs, SOBHBs, ExWDs, EMRIs (inferred) + MBHBs (fixed); HS, BOS",
    fg_sigma=FG_SIGMA,
)

GBEXB = Model(
    name="gbexb",
    param_names=("logGmu", "A_p", "A_acc", "logA_Gal", "logA_ExB"),
    special_arg_dict={"VOS_or_BOS": "BOS"},
    fixed={},
    components=("string", "Gal", "ExB"),
    prior={
        **_NOISE_PRIOR,
        "logA_Gal": ("gaussian", LOG_A_GAL, FG_SIGMA["logA_Gal"]),
        "logA_ExB": ("gaussian", LOG_A_EXB, FG_SIGMA["logA_ExB"]),
    },
    posterior_file="posterior_gbexb.pkl",
    description="Strings + WDs and SOBHBs only; BOS",
    fg_sigma={k: FG_SIGMA[k] for k in ("logA_Gal", "logA_ExB")},
)

NOFG = Model(
    name="nofg",
    param_names=("logGmu", "A_p", "A_acc"),
    special_arg_dict={"VOS_or_BOS": "BOS"},
    fixed={},
    components=("string",),
    prior=dict(_NOISE_PRIOR),
    posterior_file="posterior_nofg.pkl",
    description="Strings + instrument noise, no foregrounds; BOS",
)

MODELS = {m.name: m for m in (FULL, GBEXB, NOFG)}


def get_model(model):
    """Accept a Model or its name ('full', 'gbexb', 'nofg')."""
    if isinstance(model, Model):
        return model
    try:
        return MODELS[str(model).lower()]
    except KeyError:
        raise ValueError(f"unknown model {model!r}; choose from {list(MODELS)}") from None
