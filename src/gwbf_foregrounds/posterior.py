"""Loading and sampling the trained SBI posteriors.

The posteriors (~230 MB each) are too large for git. They are attached to a GitHub
release and downloaded once on first use into a cache folder. A local path can be
given instead everywhere.
"""
import hashlib
import os
import pickle
import sys
import types
import urllib.request
from pathlib import Path

import numpy as np

from .models import get_model

REPO = "AndronikiDimitriou/GWBackFinder-foregrounds"
RELEASE_TAG = "posteriors-v1"

# sha256 of the release assets (= the original posterior.pkl, posterior_isak_new.pkl,
# posterior_no_fore.pkl).
SHA256 = {
    "posterior_full.pkl": "c4db76ddcf7695bebbdc6e74e5e280d4f33d23022449facac6b7d375e6cba249",
    "posterior_gbexb.pkl": "b469ddf66b5236b0af5d44f82c9eb27122b1f539334dd7a5821774942b067d54",
    "posterior_nofg.pkl": "da6c8cb753f679e03c7353c72325a6ce25774fa9dd46b301dba1acbb98b1a3ed",
}


def cache_dir():
    """Folder for downloaded posteriors: $GWBF_FG_CACHE, else ~/.cache/gwbf_foregrounds."""
    d = os.environ.get("GWBF_FG_CACHE") or os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "gwbf_foregrounds")
    Path(d).mkdir(parents=True, exist_ok=True)
    return Path(d)


def _sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def fetch_posterior(model, force=False, verbose=True):
    """Path to the trained posterior of ``model``, downloading it from the release if needed."""
    m = get_model(model)
    name = m.posterior_file
    dest = cache_dir() / name
    if dest.exists() and not force:
        return dest
    url = f"https://github.com/{REPO}/releases/download/{RELEASE_TAG}/{name}"
    tmp = dest.with_suffix(".part")
    if verbose:
        print(f"Downloading {name} (~230 MB) from {url}")
    try:
        urllib.request.urlretrieve(url, tmp)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download {url} ({e}). Pass a local path with posterior=... instead."
        ) from e
    digest = _sha256(tmp)
    if digest != SHA256[name]:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Checksum mismatch for {name}: got {digest}")
    tmp.replace(dest)
    return dest


# --------------------------------------------------------------------------- #
# Unpickling. The networks were pickled together with a custom prior class from
# a module called ``sbi_pipeline`` that is not part of this package. These
# stand-ins let the pickle load; the prior is never used for sampling.
# --------------------------------------------------------------------------- #

class DummyPrior:
    def __init__(self, param_names=None):
        self.param_names = list(param_names or [])
        self.names = self.param_names
        self.ndim = len(self.param_names)

    def log_prob(self, theta):
        import torch

        if theta.dim() == 1:
            theta = theta.unsqueeze(0)
        return torch.zeros(theta.shape[0], dtype=theta.dtype, device=theta.device)

    def sample(self, sample_shape=(1,)):
        import torch

        n = sample_shape[0] if isinstance(sample_shape, tuple) else int(sample_shape)
        return torch.zeros(n, self.ndim)

    def __len__(self):
        return self.ndim


class IndependentJointPrior(DummyPrior):
    def __init__(self, prior_spec=None):
        self.prior_spec = prior_spec or {}
        super().__init__(list(self.prior_spec))

    def __setstate__(self, state):
        # unpickling restores attributes without calling __init__
        self.__dict__.update(state)
        if "param_names" not in state:
            DummyPrior.__init__(self, list(getattr(self, "prior_spec", {}) or []))


_STUBS = {"IndependentJointPrior": IndependentJointPrior, "DummyPrior": DummyPrior}


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module in ("sbi_pipeline", "__main__") and name in _STUBS:
            return _STUBS[name]
        return super().find_class(module, name)


def load_posterior(path_or_model, param_names=None):
    """Load a trained posterior.

    ``path_or_model`` is a file path, or a model name ('full', 'gbexb', 'nofg') to
    use the released posterior (downloaded on first use).
    """
    if str(path_or_model).lower() in ("full", "gbexb", "nofg"):
        model = get_model(path_or_model)
        path = fetch_posterior(model)
        param_names = param_names or model.param_names
    else:
        path = Path(path_or_model)
        if not path.exists():
            raise FileNotFoundError(path)

    # some pickles look the stand-ins up as module attributes
    if "sbi_pipeline" not in sys.modules:
        mod = types.ModuleType("sbi_pipeline")
        mod.IndependentJointPrior = IndependentJointPrior
        mod.DummyPrior = DummyPrior
        sys.modules["sbi_pipeline"] = mod

    try:
        with open(path, "rb") as fh:
            posterior = _Unpickler(fh).load()
    except Exception as e_pickle:
        import torch

        try:  # GPU-saved torch checkpoints on a CPU machine
            posterior = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            raise RuntimeError(f"could not load posterior {path}: {e_pickle}") from e_pickle

    if param_names is not None:
        posterior.prior = DummyPrior(param_names)
    return posterior


def sample(posterior, x, n=10_000, seed=None):
    """``n`` posterior samples at observation ``x`` (a Dataset or the network input tensor),
    as a float64 NumPy array of shape (n, ndim)."""
    import torch

    if hasattr(x, "x"):
        x = x.x()
    if seed is not None:
        torch.manual_seed(seed)
    s = posterior.sample((int(n),), x=x, show_progress_bars=False)
    return s.detach().cpu().numpy().astype(np.float64)
