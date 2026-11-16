"""Tests. The reference file was made by running the ORIGINAL analysis modules
(lisa.py etc. in the paper folder), so these check the package reproduces them."""
import os
import pickle
import sys
import types
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import matplotlib  # noqa: E402

matplotlib.use("Agg")

from gwbf_foregrounds import data, fisher, models, spectra, stats  # noqa: E402

REF = np.load(Path(__file__).parent / "data" / "reference_original_code.npz")
FID_SIGN = {"full": 0, "gbexb": 1, "nofg": 0}   # the reference used +1 sigma for gbexb


# --------------------------------------------------------------- models ---- #

def test_models_parameters_and_fiducials():
    assert models.FULL.ndim == 7 and models.GBEXB.ndim == 5 and models.NOFG.ndim == 3
    p = models.FULL.fiducial(logGmu=-14)
    assert p["logA_MBHB"] == 0.0 and p["logA_Gal"] == -7.84 and p["logGmu"] == -14
    hi = models.FULL.fiducial(sign=+1)
    assert hi["logA_EMRI"] == pytest.approx(0.5) and hi["logA_ExWB"] == pytest.approx(-11.0554 + 0.17)
    assert "logA_ExWB" not in models.GBEXB.fiducial()
    assert set(models.NOFG.fiducial()) == {"A_p", "A_acc"}
    assert models.get_model("GBEXB") is models.GBEXB


# ---------------------------------------- physics = original analysis code -- #

@pytest.mark.parametrize("name", ["full", "gbexb", "nofg"])
def test_simulation_matches_original_code(name):
    m = models.get_model(name)
    ds = data.simulate(m, logGmu=-13.0, seed=3, fg_sign=FID_SIGN[name])
    np.testing.assert_allclose(ds.f_obs, REF[f"{name}_f_obs"], rtol=1e-12)
    for c in ("D_AA", "D_EE", "D_TT"):
        np.testing.assert_allclose(getattr(ds, c), REF[f"{name}_{c}"], rtol=1e-10)


@pytest.mark.parametrize("name", ["full", "gbexb", "nofg"])
def test_signal_matches_original_and_components_sum(name):
    m = models.get_model(name)
    f = REF[f"{name}_f_obs"]
    p = m.fiducial(logGmu=-13.0, sign=FID_SIGN[name])
    total = np.asarray(spectra.total_signal(f, p, m))
    np.testing.assert_allclose(total, REF[f"{name}_signal"], rtol=1e-10)
    comps = spectra.component_spectra(f, p, m)
    assert set(comps) == set(m.components)
    np.testing.assert_allclose(sum(comps.values()), total, rtol=1e-10)


def test_noise_matches_original():
    n_AA, n_TT = spectra.noise(REF["full_f_obs"], 3.3, 14.0)
    np.testing.assert_allclose(np.asarray(n_AA), REF["noise_AA"], rtol=1e-12)
    np.testing.assert_allclose(np.asarray(n_TT), REF["noise_TT"], rtol=1e-12)


# --------------------------------------------------------------- data ------ #

def test_compute_n_k_matches_bruteforce():
    rng = np.random.default_rng(1)
    fine = np.sort(rng.uniform(0, 1, 20_000))
    coarse = np.sort(rng.uniform(0, 1, 300))
    coarse[5] = coarse[6]                        # a tie
    brute = np.bincount(np.argmin(np.abs(fine[:, None] - coarse[None, :]), axis=1),
                        minlength=len(coarse)).astype(float)
    brute = np.where(brute > 0, brute, 1.0)
    np.testing.assert_array_equal(data.compute_n_k(fine, coarse), brute)


def test_frequency_grid_counts_every_fine_frequency():
    f_obs, n_k = data.frequency_grid("full")
    assert len(f_obs) == 1970 and n_k.sum() == 500_000


def test_dataset_roundtrip(tmp_path):
    ds = data.simulate("gbexb", logGmu=-12.0, seed=5)
    path = ds.save(tmp_path / "d.npz")
    back = data.Dataset.load(path)
    assert back.model == "gbexb" and back.seed == 5 and back.params == ds.params
    np.testing.assert_array_equal(back.D_TT, ds.D_TT)
    assert tuple(ds.x().shape) == (1, 3 * 1970)


def test_tag():
    assert data.tag_from_logGmu(-14.0) == "logGmu_m14p0"
    assert data.tag_from_logGmu(-10.25) == "logGmu_m10p25"


# --------------------------------------------------------------- stats ----- #

def test_hdi_matches_arviz():
    az = pytest.importorskip("arviz")
    x = np.random.default_rng(0).gamma(3.0, size=20_001)
    assert stats.hdi(x, 0.95) == pytest.approx(tuple(az.hdi(x, hdi_prob=0.95)), rel=1e-12)


def test_precision_formula():
    # interval [inj, inj + log10(2)] is 100 % of G mu wide
    assert stats.precision(-12.0, -12.0 + np.log10(2), -12.0) == pytest.approx(100.0)
    p, lo, hi = stats.sigma_to_precision(0.1, -12.0)
    assert lo == pytest.approx(-12.196) and hi == pytest.approx(-11.804)


def test_crossing():
    x = np.array([-15, -14, -13.0])
    y = np.array([1000, 100, 10.0])
    assert stats.crossing_logGmu(x, y, 100) == pytest.approx(-14.0)
    assert stats.crossing_logGmu(x, y, 31.6227766) == pytest.approx(-13.5)


# --------------------------------------------------------------- fisher ---- #

@pytest.mark.parametrize("name,key", [("full", "fisher_full"), ("nofg", "fisher_nofg")])
def test_fisher_matches_saved_results(name, key):
    m = models.get_model(name)
    for lg, ref in zip(REF[key + "_logGmu"], REF[key + "_precision"]):
        F = fisher.fisher_matrix(m, m.fiducial(logGmu=lg))
        prec = stats.sigma_to_precision(fisher.sigma_logGmu(F), lg)[0]
        assert prec == pytest.approx(ref, rel=1e-8)


# ------------------------------------------------------------ posterior ---- #

def test_unpickle_with_sbi_pipeline_prior(tmp_path, monkeypatch):
    """The trained pickles reference sbi_pipeline.IndependentJointPrior, which does not exist."""
    from gwbf_foregrounds import posterior

    fake = types.ModuleType("sbi_pipeline")

    class IndependentJointPrior:
        def __init__(self, spec):
            self.prior_spec = spec

    IndependentJointPrior.__module__ = "sbi_pipeline"
    IndependentJointPrior.__qualname__ = "IndependentJointPrior"
    fake.IndependentJointPrior = IndependentJointPrior
    monkeypatch.setitem(sys.modules, "sbi_pipeline", fake)
    path = tmp_path / "p.pkl"
    with open(path, "wb") as fh:
        pickle.dump({"prior": IndependentJointPrior({"logGmu": 1})}, fh)
    monkeypatch.delitem(sys.modules, "sbi_pipeline")

    obj = posterior.load_posterior(path)
    assert isinstance(obj["prior"], posterior.IndependentJointPrior)
    assert obj["prior"].param_names == ["logGmu"]


def test_release_checksums_cover_all_models():
    from gwbf_foregrounds.posterior import SHA256

    assert {m.posterior_file for m in models.MODELS.values()} == set(SHA256)


# ------------------------------------------------------- reconstruction ---- #

def test_reconstruction_truth_matches_components():
    from gwbf_foregrounds.reconstruct import reconstruct

    m = models.FULL
    truth = m.fiducial(logGmu=-12.0)
    samples = np.tile(m.theta(truth), (4, 1))
    rec = reconstruct(samples, m, truth=truth)
    b = rec.bands()
    for k in m.components:
        np.testing.assert_allclose(b[k]["median"], rec.truths[k], rtol=1e-12)
    assert set(rec.stacks) == set(m.components) | {"noise_AA", "noise_TT"}


# ------------------------------------------------------------- plotting ---- #

def _fake_summary(scale):
    x = np.arange(-17.0, -9.0 + 0.125, 0.25)
    p = scale * 10 ** (-(x + 9.0) * 0.4) * 5
    return dict(logGmu_inj=x, precision_mean=p, precision_std=0.1 * p)


def test_precision_plots_run(tmp_path):
    from gwbf_foregrounds.plotting import precision as P

    n, g, f = _fake_summary(1), _fake_summary(1.5), _fake_summary(2)
    P.plot_fg_vs_nofg(f, n, out=tmp_path / "a.pdf")
    P.plot_with_validation(f, n, mcmc=(np.array([-14.0, -12.0]), np.array([50.0, 20.0])),
                           fisher=dict(logGmu_inj=n["logGmu_inj"], precision=n["precision_mean"]),
                           out=tmp_path / "b.pdf")
    P.plot_low_normal_high(n, g, f, out=tmp_path / "c.pdf")
    P.plot_nofg_gbexb_full(n, g, f, out=tmp_path / "d.pdf")
    assert len(list(tmp_path.glob("*.pdf"))) == 4


def test_coverage_and_corner_plots_run(tmp_path):
    from gwbf_foregrounds.plotting.corner import plot_1d_posteriors
    from gwbf_foregrounds.plotting.coverage import plot_coverage, plot_tarp

    rng = np.random.default_rng(0)
    plot_coverage([rng.uniform(size=200)] * 3, ["a", "b", "c"], ncols=3, out=tmp_path / "c.pdf")
    a = np.linspace(0, 1, 50)
    plot_tarp(a, a, n_sims=200, atc=0.01, ks_pval=0.5, out=tmp_path / "t.pdf")
    s = rng.normal(size=(2000, 3))
    plot_1d_posteriors(s, s + 0.1, ["logGmu", "A_p", "A_acc"], truths=[0, 0, 0],
                       out=tmp_path / "1d.pdf")
    assert len(list(tmp_path.glob("*.pdf"))) == 3


def test_paper_figures_and_table_run(tmp_path):
    from gwbf_foregrounds import summary
    from gwbf_foregrounds.plotting import precision as P

    n, g, f = _fake_summary(1), _fake_summary(1.5), _fake_summary(2)
    P.plot_paper_fig3(n, g, f, _fake_summary(1.8), _fake_summary(2.5), out=tmp_path / "f3.pdf")
    fis = dict(logGmu_inj=n["logGmu_inj"], precision=n["precision_mean"])
    P.plot_paper_fig4(f, mcmc=(np.array([-14.0, -12.0]), np.array([50.0, 20.0])), fisher=fis,
                      out=tmp_path / "f4.pdf")
    rows = summary.table_ii(f, n, mcmc=(np.array([-14.0]), np.array([50.0])), fisher=fis)
    assert len(rows) == 7 and rows[2]["mcmc"] == 50.0 and np.isnan(rows[0]["mcmc"])
    assert "tabular" in summary.format_table_ii(rows, latex=True)
    assert len(list(tmp_path.glob("*.pdf"))) == 2


def test_gamma_coverage_is_uniform_for_exact_posterior():
    """A posterior that is exactly right gives a uniform coverage statistic."""
    import torch
    from scipy.stats import kstest
    from gwbf_foregrounds.coverage import gamma_coverage_all

    rng = np.random.default_rng(0)

    class GaussPost:   # posterior N(x, 1) for data x = theta + N(0, 1) with a flat prior
        def sample(self, shape, x, show_progress_bars=False):
            return torch.as_tensor(x[0].numpy() + rng.standard_normal((shape[0], 2)))

    thetas = rng.normal(size=(300, 2))
    xs = torch.as_tensor(thetas + rng.standard_normal(thetas.shape))
    g = gamma_coverage_all(GaussPost(), thetas, xs, ["a", "b"], n_post_samples=400)
    assert kstest(g["a"], "uniform").pvalue > 0.01 and kstest(g["b"], "uniform").pvalue > 0.01
