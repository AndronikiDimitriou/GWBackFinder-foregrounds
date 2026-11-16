"""Command line: ``gwbf-fg <command> ...``. Run ``gwbf-fg -h`` for the list."""
import argparse
import glob
import os


def _out(args, name):
    os.makedirs(args.output_dir, exist_ok=True)
    return os.path.join(args.output_dir, name)


def cmd_reconstruct(args):
    from . import data, posterior, reconstruct
    from .models import get_model
    from .plotting.spectra import plot_reconstruction

    m = get_model(args.model)
    ds = (data.Dataset.load(args.dataset) if args.dataset
          else data.simulate(m, logGmu=args.logGmu, seed=args.seed, fg_sign=args.fg_sign))
    post = posterior.load_posterior(args.posterior or m.name, m.param_names)
    s = posterior.sample(post, ds, n=args.n_post, seed=args.seed)
    rec = reconstruct.reconstruct(s, m, ds.f_obs, truth=ds.params)
    out = _out(args, args.out or f"reconstruction_{m.name}_{data.tag_from_logGmu(ds.logGmu_inj)}.pdf")
    plot_reconstruction(rec, out=out, show_noise=args.show_noise,
                        data=ds if args.show_data else None)
    print(f"saved {out}")


def cmd_scan(args):
    from . import scan

    grid = scan.np.arange(args.logGmu_min, args.logGmu_max + 1e-9, args.step)
    scan.run_sbi_scan(args.model, posterior=args.posterior, logGmu_values=grid,
                      fg_sign=args.fg_sign, n_datasets=args.n_datasets, n_samples=args.n_post,
                      output_dir=args.output_dir)


def cmd_fisher(args):
    from . import fisher

    grid = fisher.np.arange(args.logGmu_min, args.logGmu_max + 1e-9, args.step)
    fisher.run_fisher(args.model, grid, output_dir=args.output_dir)


def cmd_mcmc(args):
    from . import data, mcmc

    ds = (data.Dataset.load(args.dataset) if args.dataset
          else data.simulate("full", logGmu=args.logGmu, seed=args.seed))
    tag = data.tag_from_logGmu(ds.logGmu_inj)
    if not args.dataset:
        ds.save(_out(args, f"dataset_{tag}.npz"))
    mcmc.run_mcmc(ds, output_file=_out(args, f"mcmc_{tag}.npz"), start_logGmu=args.start_logGmu,
                  max_steps=args.max_steps, seed=args.seed)


def cmd_simulate(args):
    from . import data

    ds = data.simulate(args.model, logGmu=args.logGmu, seed=args.seed, fg_sign=args.fg_sign)
    out = _out(args, args.out or f"dataset_{args.model}_{data.tag_from_logGmu(args.logGmu)}_seed{args.seed}.npz")
    ds.save(out)
    print(f"saved {out}")


def cmd_plot_precision(args):
    from . import scan
    from .plotting import precision as P

    d = args.summary_dir
    f = lambda name: os.path.join(d, name)  # noqa: E731
    P.plot_fg_vs_nofg(f("sbi_fg_summary_with_errors.npz"), f("sbi_nofg_summary_with_errors.npz"),
                      out=_out(args, "sbi_precision_fg_vs_nofg.pdf"))
    mcmc = None
    if args.mcmc_dir:
        files = sorted(glob.glob(os.path.join(args.mcmc_dir, "mcmc_logGmu_*.npz")))
        x, y = scan.mcmc_precision(files)
        keep = (x >= args.mcmc_min) & (x <= args.mcmc_max)
        mcmc = (x[keep], y[keep])
    fis = f("fisher_95_intervals.npz")
    fis_n = f("fisher_nofg_95_intervals.npz")
    P.plot_with_validation(f("sbi_fg_summary_with_errors.npz"), f("sbi_nofg_summary_with_errors.npz"),
                           mcmc=mcmc, fisher=fis if os.path.exists(fis) else None,
                           fisher_nofg=fis_n if os.path.exists(fis_n) else None,
                           out=_out(args, "sbi_precision_normal_with_mcmc_fisher.pdf"))
    if os.path.exists(f("sbi_lowfg_summary_with_errors.npz")):
        P.plot_low_normal_high(f("sbi_lowfg_summary_with_errors.npz"),
                               f("sbi_fg_summary_with_errors.npz"),
                               f("sbi_highfg_summary_with_errors.npz"),
                               out=_out(args, "sbi_precision_low_normal_high.pdf"))
    if os.path.exists(f("sbi_gbexb_summary_with_errors.npz")):
        P.plot_nofg_gbexb_full(f("sbi_nofg_summary_with_errors.npz"),
                               f("sbi_gbexb_summary_with_errors.npz"),
                               f("sbi_fg_summary_with_errors.npz"),
                               out=_out(args, "sbi_precision_nofg_gbexb_fullfg.pdf"))
    print(f"plots saved in {args.output_dir}")


def cmd_fetch(args):
    from .posterior import fetch_posterior

    for m in args.models:
        print(fetch_posterior(m))


def main(argv=None):
    p = argparse.ArgumentParser(prog="gwbf-fg", description=__doc__)
    p.add_argument("--output-dir", default=os.environ.get("GWBF_FG_OUTPUT", "."),
                   help="where results go (default: current folder or $GWBF_FG_OUTPUT)")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, model=True):
        if model:
            sp.add_argument("--model", default="full", choices=["full", "gbexb", "nofg"])
        sp.add_argument("--posterior", help="local posterior .pkl (default: download)")
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--fg-sign", type=int, default=0, choices=[-1, 0, 1],
                        help="foregrounds at fiducial -1 sigma / fiducial / +1 sigma")

    sp = sub.add_parser("reconstruct", help="posterior reconstruction of every component")
    common(sp)
    sp.add_argument("--logGmu", type=float, default=-14.0)
    sp.add_argument("--dataset", help="use a saved dataset .npz instead of simulating")
    sp.add_argument("--n-post", type=int, default=1000)
    sp.add_argument("--show-noise", action="store_true")
    sp.add_argument("--show-data", action="store_true")
    sp.add_argument("--out")
    sp.set_defaults(func=cmd_reconstruct)

    for name, func, helptext in (("scan", cmd_scan, "SBI precision vs injected G mu"),
                                 ("fisher", cmd_fisher, "Fisher precision vs injected G mu")):
        sp = sub.add_parser(name, help=helptext)
        common(sp)
        sp.add_argument("--logGmu-min", type=float, default=-17.0)
        sp.add_argument("--logGmu-max", type=float, default=-9.0)
        sp.add_argument("--step", type=float, default=0.25)
        sp.add_argument("--n-datasets", type=int, default=20)
        sp.add_argument("--n-post", type=int, default=10_000)
        sp.set_defaults(func=func)

    sp = sub.add_parser("mcmc", help="emcee run, full model only")
    sp.add_argument("--logGmu", type=float, default=-14.0)
    sp.add_argument("--dataset")
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--start-logGmu", type=float, default=-19.0)
    sp.add_argument("--max-steps", type=int, default=1_000_000)
    sp.set_defaults(func=cmd_mcmc)

    sp = sub.add_parser("simulate", help="simulate and save one mock dataset")
    common(sp)
    sp.add_argument("--logGmu", type=float, required=True)
    sp.add_argument("--out")
    sp.set_defaults(func=cmd_simulate)

    sp = sub.add_parser("plot-precision", help="the precision bar plots from summary files")
    sp.add_argument("--summary-dir", default=".")
    sp.add_argument("--mcmc-dir", help="folder with mcmc_logGmu_*.npz")
    sp.add_argument("--mcmc-min", type=float, default=-16.0)
    sp.add_argument("--mcmc-max", type=float, default=-10.0)
    sp.set_defaults(func=cmd_plot_precision)

    sp = sub.add_parser("fetch", help="download the trained posteriors")
    sp.add_argument("models", nargs="*", default=["full", "gbexb", "nofg"])
    sp.set_defaults(func=cmd_fetch)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
