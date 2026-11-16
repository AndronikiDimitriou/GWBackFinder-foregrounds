"""GWBackFinder-foregrounds: cosmic strings and astrophysical foregrounds with LISA.

Companion to GWBackFinder (https://github.com/AndronikiDimitriou/GWBackFinder).
Where GWBackFinder reconstructs the spectral shape model-independently, this package
infers the string tension G mu jointly with the LISA noise and the astrophysical
foregrounds (WDs, SOBHBs, ExWDs, EMRIs), with one joint SBI posterior per model,
MCMC and Fisher cross-checks, and the figures of the paper.

Modules are imported lazily; ``import gwbf_foregrounds`` does not load JAX or torch.
"""
import os as _os

_os.environ.setdefault("JAX_PLATFORMS", "cpu")

from .models import FULL, GBEXB, MODELS, NOFG, get_model  # noqa: E402

__all__ = ["FULL", "GBEXB", "NOFG", "MODELS", "get_model"]
__version__ = "0.1.0"
