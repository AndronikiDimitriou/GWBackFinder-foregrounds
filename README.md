# GWBackFinder-foregrounds

Measuring the cosmic-string tension Gμ with LISA in the presence of astrophysical foregrounds,
with simulation-based inference. Code, trained networks and figures of
*Cosmic string gravitational wave backgrounds at LISA: II. Reconstruction of conventional signals
over astrophysical foregrounds* ([arXiv:2607.15219](https://arxiv.org/abs/2607.15219)).

Companion of [GWBackFinder](https://github.com/AndronikiDimitriou/GWBackFinder), which reconstructs
the spectral shape model-independently; here the signal is a cosmic-string background and Gμ is inferred
jointly with the LISA noise and the foreground amplitudes (WDs, SOBHBs, ExWDs, EMRIs; MBHBs fixed),
with MCMC and Fisher cross-checks.

## Install

```bash
git clone https://github.com/AndronikiDimitriou/GWBackFinder-foregrounds.git
cd GWBackFinder-foregrounds
pip install -e ".[all]"
```

The trained networks (~230 MB each) are downloaded automatically from the
[release](https://github.com/AndronikiDimitriou/GWBackFinder-foregrounds/releases/tag/posteriors-v1)
on first use. Needs `sbi` 0.25, so use a separate environment from GWBackFinder (`sbi` 0.22).

## Quick start

```python
from gwbf_foregrounds import data, posterior, reconstruct
from gwbf_foregrounds.plotting.spectra import plot_reconstruction

ds   = data.simulate("full", logGmu=-13, seed=0)       # mock LISA data
post = posterior.load_posterior("full")                # trained network
s    = posterior.sample(post, ds, n=1000)              # posterior samples
plot_reconstruction(reconstruct.reconstruct(s, "full", ds.f_obs, truth=ds.params))
```

## Tutorials

The notebooks in [`examples/`](examples/) are saved with their outputs, so they can be read on GitHub.

| Notebook | What it shows |
|---|---|
| [01 Quick start](examples/01_quickstart.ipynb) | models, simulating data, posterior samples, precision on Gμ, reconstructing every component |
| [02 Precision and Fisher](examples/02_precision_and_fisher.ipynb) | precision vs Gμ over noise realisations, Fisher forecasts (with and without priors), plots and tables |
| [03 MCMC](examples/03_mcmc.ipynb) | emcee on the same dataset as SBI, comparison |
| [04 Training](examples/04_training.ipynb) | training record and loss curves of the released full network; simulating a training set and training a network |
| [05 Paper figures](examples/05_paper_figures.ipynb) | Table II and Figs. 2–4 of the paper from [`results/`](results/); PDFs in [`figures/`](figures/) |

## Models

| Name | Parameters | Foregrounds |
|---|---|---|
| `full` | log Gμ, A_p, A_acc, log A_Gal, log A_ExB, log A_ExWB, log A_EMRI | WDs, SOBHBs, ExWDs, EMRIs + MBHBs fixed (HS scenario) |
| `gbexb` | log Gμ, A_p, A_acc, log A_Gal, log A_ExB | WDs, SOBHBs |
| `nofg` | log Gμ, A_p, A_acc | none |

Priors: log Gμ ~ U(−18, −9), A_p ~ U(12, 18), A_acc ~ U(2.4, 3.6), log A_Gal ~ N(−7.84, 0.21),
log A_ExB ~ N(−12.38, 0.344), log A_ExWB ~ N(−11.0554, 0.17), log A_EMRI ~ N(0, 0.5).

Precision on Gμ: 100 · (10^hi − 10^lo) / 10^inj for the 95% HDI [lo, hi] of log10 Gμ.

## Command line

```bash
gwbf-fg reconstruct --model full --logGmu -13 --show-noise    # reconstruction plot
gwbf-fg scan --model full                                      # SBI precision vs Gμ (hours)
gwbf-fg fisher --model full --with-priors                      # Fisher precision vs Gμ
gwbf-fg mcmc --logGmu -12                                      # emcee, full model
gwbf-fg train --model full                                     # simulate + train (see notebook 04)
gwbf-fg fetch                                                  # download the trained networks
```

## Package contents

`physics/` (LISA noise and response, string spectra, foregrounds, mock data; unchanged from the
analysis code except import paths) · `models` · `data` · `spectra` · `posterior` · `reconstruct` ·
`scan` · `fisher` · `mcmc` · `train` · `summary` · `plotting`.

Tests (`pytest`) compare the package with reference values produced by the original analysis code.

## Citation

A. Dimitriou, D. G. Figueroa, P. Simakachorn, I. Stomberg, B. Zaldívar, *Cosmic string gravitational wave
backgrounds at LISA: II. Reconstruction of conventional signals over astrophysical foregrounds*,
[arXiv:2607.15219](https://arxiv.org/abs/2607.15219), and
A. Dimitriou, D. G. Figueroa, B. Zaldívar, *Fast Likelihood-free Reconstruction of Gravitational Wave
Backgrounds*, [arXiv:2309.08430](https://arxiv.org/abs/2309.08430).
