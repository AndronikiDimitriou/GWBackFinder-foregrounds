"""Mock LISA datasets: simulation, the network input vector, saving and loading.

A dataset saved here can be read by both the SBI and the MCMC code, so the two
methods can be compared on exactly the same noise realisation.
"""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .models import get_model
from .physics import lisa

units = lisa.units


@dataclass
class Dataset:
    f_obs: np.ndarray      # coarse-grained frequencies [GeV]
    D_AA: np.ndarray
    D_EE: np.ndarray
    D_TT: np.ndarray
    model: str
    params: dict           # injected parameters (incl. fixed ones)
    seed: int

    @property
    def f_Hz(self):
        return self.f_obs * units["GeV_to_Hz"]

    @property
    def logGmu_inj(self):
        return self.params.get("logGmu")

    def x(self):
        """Network input: [log10 D_AA, log10 D_EE, log10 D_TT], shape (1, 3 * n_f), float32 tensor."""
        return obs_vector(self.D_AA, self.D_EE, self.D_TT)

    def save(self, path):
        m = get_model(self.model)
        np.savez(
            path,
            f_obs=self.f_obs, D_AA=self.D_AA, D_EE=self.D_EE, D_TT=self.D_TT,
            model=self.model, seed=self.seed,
            param_names=np.array(m.param_names),
            true_params=np.array(m.theta(self.params)),
            logGmu_inj=self.params.get("logGmu", np.nan),
            fixed_names=np.array(list(m.fixed)), fixed_values=np.array(list(m.fixed.values()), float),
        )
        return path

    @classmethod
    def load(cls, path, model=None):
        """Load a dataset saved by ``Dataset.save`` or by the old ``mcmc_paper2.py``
        (``dataset_logGmu_*.npz``; those are full-model datasets with seed 42)."""
        d = np.load(path, allow_pickle=True)
        model = str(d["model"]) if "model" in d.files else (model or "full")
        m = get_model(model)
        names = [str(n) for n in d["param_names"]]
        params = dict(zip(names, map(float, d["true_params"])))
        if "fixed_names" in d.files:
            params.update(zip(map(str, d["fixed_names"]), map(float, d["fixed_values"])))
        else:
            params.update(m.fixed)
        seed = int(d["seed"]) if "seed" in d.files else 42
        return cls(np.asarray(d["f_obs"]), np.asarray(d["D_AA"]), np.asarray(d["D_EE"]),
                   np.asarray(d["D_TT"]), model, params, seed)


def obs_vector(D_AA, D_EE, D_TT):
    import torch

    x = np.concatenate([np.log10(D_AA), np.log10(D_EE), np.log10(D_TT)]).astype(np.float32)
    return torch.tensor(x).unsqueeze(0)


def _run_simulator(param_list, model, seed):
    model = get_model(model)
    _, f_coarse, results = lisa.generate_lisa_mock_data_matrix(
        parameter_samples=param_list, key=int(seed), verbose=False,
        special_arg_dict=model.special_arg_dict,
    )
    return np.asarray(f_coarse), results


def simulate(model, params=None, logGmu=None, seed=0, fg_sign=0):
    """Simulate one mock dataset.

    Either pass the full ``params`` dict, or ``logGmu`` (and optionally ``fg_sign``
    = -1/0/+1 for low/normal/high foregrounds) to use the model's fiducial values.
    """
    model = get_model(model)
    if params is None:
        params = model.fiducial(logGmu=logGmu, sign=fg_sign)
    params = {**model.fixed, **params}
    f_obs, results = _run_simulator([params], model, seed)
    r = results[0]["data"]
    return Dataset(f_obs, np.asarray(r["D_AA"]), np.asarray(r["D_EE"]), np.asarray(r["D_TT"]),
                   model.name, params, int(seed))


def simulate_x(thetas, model, seed=0):
    """Batch simulator for coverage tests: (N, ndim) parameters -> (N, 3 * n_f) float32 tensor."""
    import torch

    model = get_model(model)
    thetas = np.asarray(thetas.detach().cpu() if hasattr(thetas, "detach") else thetas)
    param_list = [model.to_params(row) for row in thetas]
    _, results = _run_simulator(param_list, model, seed)
    xs = [obs_vector(r["data"]["D_AA"], r["data"]["D_EE"], r["data"]["D_TT"])[0] for r in results]
    return torch.stack(xs)


def compute_n_k(f_fine, f_coarse):
    """Number of fine frequencies whose nearest coarse frequency is bin k (at least 1).

    Same result as the ``argmin(|f_fine[:, None] - f_coarse[None, :]|)`` in the scripts,
    but without building the (500 000 x 1970) matrix.
    """
    f_fine = np.asarray(f_fine, float)
    f_coarse = np.asarray(f_coarse, float)
    order = np.argsort(f_coarse, kind="stable")
    fc = f_coarse[order]
    j = np.clip(np.searchsorted(fc, f_fine), 1, len(fc) - 1)
    left, right = fc[j - 1], fc[j]
    # ties go to the lower index, as np.argmin does
    nearest_sorted = np.where(np.abs(f_fine - left) <= np.abs(right - f_fine), j - 1, j)
    # duplicated coarse values: take the first (lowest original index), like np.argmin
    nearest_sorted = np.searchsorted(fc, fc[nearest_sorted], side="left")
    idx = order[nearest_sorted]
    n_k = np.bincount(idx, minlength=len(f_coarse)).astype(float)
    return np.where(n_k > 0, n_k, 1.0)


@lru_cache(maxsize=8)
def _grid(model_name):
    model = get_model(model_name)
    f_fine, f_coarse, _ = lisa.generate_lisa_mock_data_matrix(
        parameter_samples=[model.fiducial(logGmu=-14.0)], key=42, verbose=False,
        special_arg_dict=model.special_arg_dict,
    )
    f_obs = np.asarray(f_coarse)
    return f_obs, compute_n_k(np.asarray(f_fine), f_obs)


def frequency_grid(model="full"):
    """(f_obs [GeV], n_k) of the coarse-grained data, at fiducial noise.

    The coarse bins are noise-weighted averages, so they are computed once with the
    fiducial noise (A_p = 15, A_acc = 3), as in the analysis scripts.
    """
    f_obs, n_k = _grid(get_model(model).name)
    return f_obs.copy(), n_k.copy()


def tag_from_logGmu(logGmu):
    """-14.0 -> 'logGmu_m14p0', -10.25 -> 'logGmu_m10p25' (same names as the old files)."""
    return f"logGmu_{str(round(float(logGmu), 2)).replace('.', 'p').replace('-', 'm')}"
