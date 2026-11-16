"""Train an SBI posterior (NPE, neural spline flow) for one of the models.

Two steps, both resumable:

1. ``simulate_training_set``: draws parameters from the model's prior and simulates the
   network input (log10 D_AA, D_EE, D_TT; 3 x 1970 = 5910 numbers) in chunks, in parallel.
   Each chunk is saved as soon as it is done, so an interrupted run continues where it stopped.
2. ``train``: NPE with an NSF density estimator; saves the posterior (.pkl), the training
   history and a ``model_info.txt``.

The defaults are the settings of the full-foreground network of the paper
(``posterior_full.pkl``, from its model_info.txt): 500 000 simulations, NSF with 10
transforms and 300 hidden features, no embedding net, batch 256, learning rate 1e-5,
at most 1000 epochs, stop after 30 epochs without improvement, 10% validation.
The original training code (``sbi_pipeline``) is not part of this package, so a network
trained here is comparable to, not bit-identical with, the released one.

One simulation takes ~2 s on one CPU core: 500 000 simulations are ~250 core-hours.
"""
import json
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from .models import get_model

DEFAULTS = dict(
    num_simulations=500_000,
    hidden_features=300,
    num_transforms=10,
    batch_size=256,
    learning_rate=1e-5,
    max_epochs=1000,
    stop_after_epochs=30,
    validation_fraction=0.1,
)


# --------------------------------------------------------------------------- #
# Prior
# --------------------------------------------------------------------------- #

def torch_prior(model):
    """The model's training prior as an sbi-compatible torch distribution."""
    import torch
    from torch.distributions import Normal, Uniform
    from sbi.utils import MultipleIndependent

    m = get_model(model)
    dists = []
    for name in m.param_names:
        kind, a, b = m.prior[name]
        a, b = torch.tensor([float(a)]), torch.tensor([float(b)])
        dists.append(Uniform(a, b) if kind == "uniform" else Normal(a, b))
    return MultipleIndependent(dists, validate_args=False)


# --------------------------------------------------------------------------- #
# Simulation of the training set
# --------------------------------------------------------------------------- #

def _simulate_chunk(args):
    """Worker: simulate one chunk and save it. Runs in a fresh process."""
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    model, thetas, seed, path = args
    from .data import simulate_x

    xs = simulate_x(thetas, model, seed=seed).numpy()
    tmp = Path(str(path) + ".tmp.npz")
    np.savez(tmp, theta=thetas.astype(np.float32), x=xs.astype(np.float32), seed=seed)
    tmp.replace(path)
    return path


def simulate_training_set(model="full", num_simulations=DEFAULTS["num_simulations"],
                          out_dir="training_data", chunk_size=1000, n_workers=None, seed=0,
                          verbose=True):
    """Simulate ``num_simulations`` (theta, x) pairs from the prior into ``out_dir``.

    Chunks already on disk are skipped, so calling this again resumes an interrupted run.
    All parameters are drawn up front from ``seed``; chunk k uses simulation seed ``seed + k``.
    Returns the list of chunk files.
    """
    from .coverage import sample_prior

    m = get_model(model)
    out = Path(out_dir) / m.name
    out.mkdir(parents=True, exist_ok=True)
    thetas = sample_prior(m, num_simulations, seed=seed)
    n_chunks = int(np.ceil(num_simulations / chunk_size))
    jobs, files = [], []
    for k in range(n_chunks):
        path = out / f"chunk_{k:05d}.npz"
        files.append(path)
        if not path.exists():
            jobs.append((m.name, thetas[k * chunk_size:(k + 1) * chunk_size], seed + 1 + k, path))
    (out / "simulation_info.json").write_text(json.dumps(dict(
        model=m.name, num_simulations=num_simulations, chunk_size=chunk_size, seed=seed,
        param_names=list(m.param_names), prior={k: list(v) for k, v in m.prior.items()},
        fixed=m.fixed, special_arg_dict=m.special_arg_dict), indent=2))
    if verbose:
        print(f"[{m.name}] {n_chunks - len(jobs)}/{n_chunks} chunks already done, "
              f"{len(jobs)} to simulate in {out}")
    if jobs:
        n_workers = n_workers or max(1, (os.cpu_count() or 2) - 1)
        t0 = time.time()
        with ProcessPoolExecutor(n_workers, mp_context=get_context("spawn")) as pool:
            for i, _ in enumerate(pool.map(_simulate_chunk, jobs), 1):
                if verbose and (i % max(1, len(jobs) // 20) == 0 or i == len(jobs)):
                    el = time.time() - t0
                    print(f"  {i}/{len(jobs)} chunks  {el / 60:.1f} min  "
                          f"(~{el / i * (len(jobs) - i) / 60:.0f} min left)", flush=True)
    return files


def load_training_set(model="full", out_dir="training_data"):
    """(theta, x) float32 tensors from all chunks in ``out_dir/<model>``."""
    import torch

    files = sorted((Path(out_dir) / get_model(model).name).glob("chunk_*.npz"))
    if not files:
        raise FileNotFoundError(f"no chunks in {Path(out_dir) / get_model(model).name}")
    th = np.concatenate([np.load(f)["theta"] for f in files])
    x = np.concatenate([np.load(f)["x"] for f in files])
    ok = np.all(np.isfinite(x), axis=1)
    if not ok.all():
        print(f"dropping {np.sum(~ok)} simulations with non-finite data")
    return torch.as_tensor(th[ok]), torch.as_tensor(x[ok])


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def train(model="full", out_dir="training_data", save_dir="trained", hidden_features=None,
          num_transforms=None, batch_size=None, learning_rate=None, max_epochs=None,
          stop_after_epochs=None, validation_fraction=None, device="cpu", seed=0,
          tensorboard_dir=None, verbose=True):
    """Train NPE (NSF) on the simulated training set and save the posterior.

    Saves in ``save_dir/<model>/``: ``posterior.pkl``, ``history.npz``, ``model_info.txt``.
    Returns the posterior.
    """
    import torch
    from sbi.inference import NPE
    from sbi.neural_nets import posterior_nn

    cfg = {k: v for k, v in DEFAULTS.items() if k != "num_simulations"}
    for k, v in dict(hidden_features=hidden_features, num_transforms=num_transforms,
                     batch_size=batch_size, learning_rate=learning_rate, max_epochs=max_epochs,
                     stop_after_epochs=stop_after_epochs,
                     validation_fraction=validation_fraction).items():
        if v is not None:
            cfg[k] = v
    m = get_model(model)
    torch.manual_seed(seed)
    theta, x = load_training_set(m, out_dir)
    if verbose:
        print(f"[{m.name}] training on {len(theta)} simulations, x dim {x.shape[1]}, config {cfg}")

    summary_writer = None
    if tensorboard_dir:
        from torch.utils.tensorboard import SummaryWriter
        summary_writer = SummaryWriter(tensorboard_dir)
    estimator = posterior_nn(model="nsf", hidden_features=cfg["hidden_features"],
                             num_transforms=cfg["num_transforms"])
    kwargs = dict(prior=torch_prior(m), density_estimator=estimator, device=device)
    if summary_writer is not None:
        kwargs["summary_writer"] = summary_writer
    inference = NPE(**kwargs)
    inference.append_simulations(theta, x)
    t0 = time.time()
    density = inference.train(
        training_batch_size=cfg["batch_size"], learning_rate=cfg["learning_rate"],
        validation_fraction=cfg["validation_fraction"], stop_after_epochs=cfg["stop_after_epochs"],
        max_num_epochs=cfg["max_epochs"], show_train_summary=verbose)
    elapsed = time.time() - t0
    posterior = inference.build_posterior(density)

    s = inference._summary
    history = dict(train_losses=np.asarray(s["training_loss"]),
                   valid_losses=np.asarray(s["validation_loss"]))
    best = float(np.asarray(s["best_validation_loss"]).ravel()[-1])
    epochs = int(np.asarray(s["epochs_trained"]).ravel()[-1])

    out = Path(save_dir) / m.name
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "posterior.pkl", "wb") as fh:
        pickle.dump(posterior, fh)
    np.savez(out / "history.npz", **history, best_validation_loss=best, epochs_trained=epochs)
    info = [
        f"model: {m.name}  ({m.description})",
        f"parameters: {', '.join(m.param_names)}",
        f"fixed: {m.fixed}",
        f"physics settings: {m.special_arg_dict}",
        "prior:",
        *[f"  {k}: {v}" for k, v in m.prior.items()],
        f"simulations: {len(theta)}   feature dimension: {x.shape[1]}",
        "density estimator: NSF (sbi NPE), no embedding net",
        *[f"  {k}: {v}" for k, v in cfg.items()],
        f"epochs trained: {epochs}   best validation loss: {best:.4f}",
        f"training time: {elapsed / 3600:.2f} h   device: {device}",
        f"sbi version: {__import__('sbi').__version__}",
    ]
    (out / "model_info.txt").write_text("\n".join(info) + "\n")
    if verbose:
        print(f"saved {out / 'posterior.pkl'} (best validation loss {best:.3f}, {epochs} epochs)")
    return posterior
