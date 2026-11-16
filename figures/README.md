# Figures

Made with `examples/paper_figures.py` from the precision summaries of the analysis
(`sbi_*_summary_with_errors.npz`, 20 noise realisations × 10 000 posterior samples per
injection), the Fisher files (`fisher_95_intervals.npz`, `fisher_nofg_95_intervals.npz`,
classic Fisher, no prior terms) and the MCMC chains `mcmc_logGmu_*.npz`.

| File | Content |
|---|---|
| `sbi_precision_fg_vs_nofg` | 95% HDI precision on Gμ: no foregrounds (blue) and the extra width from all foregrounds (orange) |
| `sbi_precision_normal_with_mcmc_fisher` | the same over the full grid, with MCMC (red, from the chains, −16 ≤ log Gμ ≤ −10) and Fisher (stars: with foregrounds, triangles: without) |
| `sbi_precision_nofg_gbexb_fullfg` | no foregrounds < WD + SOBHB only (`gbexb` model) < all foregrounds |
| `sbi_precision_low_normal_high` | foreground amplitudes at fiducial −1σ / fiducial / +1σ |
| `reconstruction_all_logGmu_m14p0` | every spectral component and the AA noise reconstructed from the SBI posterior (median, 95% band); dotted: MCMC 95% band; dashed: injected |
| `reconstruction_gbexb_logGmu_m12p0` | the same for the WD + SOBHB model at Gμ = 10⁻¹² |
| `corner_hdi_logGmu_m14p0`, `..._filtered` | MCMC (black) vs SBI (green) on the same dataset (seed 42); `_filtered` without A_p, A_acc. Diagonal titles: SBI 95% HDI |
| `1d_posteriors_logGmu_m14p0` | 1D marginals, MCMC vs SBI |
| `lisa_noise_pls_AA_TT` | LISA AA and TT noise and the AA power-law sensitivity (SNR 10, 3 years) |

Notes

* The corner plots show the samples exactly as produced by each method. For
  log A_SOBHB the two differ by ~0.2 dex (within 1σ), along a direction that is
  degenerate with ExWD, Gμ, WD and EMRI amplitudes.
* MCMC at log Gμ = −9 is left out: −9 is the upper edge of the prior.
