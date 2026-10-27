"""
How strongly does the ACT map alone prefer one set of cluster parameters over another?

Built to debug the joint M2+ACT MOOJ1142 fit, where adding ACT pushed gNFW r_s from
about 24 (M2 only) to the prior edge at 50. Only the ACT dataset of a joint config is
loaded (no TODs), so this runs on a single process.

For each `par_results_*.dill` given it reports the ACT chi-squared:
    as_saved   : the parameters exactly as saved.
    ps_solved  : the ACT point source amplitude re-solved. The model is linear in it,
                 so this is exact, and parameter sets from an M2-only fit (which have
                 no ACT point source) can be compared fairly.
    zero_mean  : as ps_solved, but with the ivar weighted mean removed from the model,
                 matching what `load_maps` does to the data.
It also prints how much of the ACT map the model grid covers, writes model and residual
FITS for each parameter set, and plots radial profiles of the data and models.

Parameters are matched by name and by the order in which repeated names appear
(e.g. the first `beta` is the isobeta one and the second the gNFW one). This works for
results from any config that lists its models in the same order as the joint config.
Parameters missing from a result keep their config values.

Usage
-----
    python scripts/act_chisq_diagnostic.py \\
        configs/MOOJ1142_Joint/MOOJ1142_joint_M2_ACT_gNFW_rs_peak_subcluster.yaml \\
        /path/to/m2_only/par_results_final_fit.dill \\
        /path/to/joint/par_results_fit18.dill \\
        --labels m2_only joint
"""

import argparse
import os
from copy import copy

import dill
import jax

jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

import witch.utils as wu
from witch.cfg_loader import load_cfg
from witch.fitter import load_config
from witch.objective import chisq_objective

# Categorical slots in fixed order, the data is drawn in neutral ink
SERIES_COLORS = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
]
INK = "#333333"
MUTED = "#8a8a8a"


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config", help="Joint config with the ACT dataset")
    parser.add_argument("results", nargs="+", help="par_results_*.dill files")
    parser.add_argument("--labels", nargs="+", help="One label per results file")
    parser.add_argument("--dataset", default="act", help="Name of the ACT dataset")
    parser.add_argument(
        "--ps-model",
        default="ps_act",
        help="Model holding the ACT point source, its amp_g is re-solved",
    )
    parser.add_argument("--ps-par", default="amp_g")
    parser.add_argument(
        "--bin", type=float, default=1.0, help="Radial bin width in arcmin"
    )
    return parser.parse_args()


def _keyed(names):
    """(name, occurrence) for each name, so repeated names can be told apart."""
    seen = {}
    keys = []
    for name in names:
        n = seen.get(name, 0)
        seen[name] = n + 1
        keys.append((name, n))
    return keys


def _load_pars(path, metamodel):
    with open(path, "rb") as f:
        res = dill.load(f)
    idx = {key: i for i, key in enumerate(_keyed(metamodel.par_names))}
    pars = np.array(metamodel.parameters)
    matched = np.zeros(len(pars), dtype=bool)
    unmatched = []
    for key, val in zip(_keyed([str(n) for n in res["par_names"]]), res["parameters"]):
        if key in idx:
            pars[idx[key]] = val
            matched[idx[key]] = True
        else:
            unmatched.append(key[0])
    return pars, matched, unmatched


def _with_pars(metamodel, pars):
    # `update` modifies in place, copy so the config values stay the defaults
    return copy(metamodel).update(
        jnp.array(pars), metamodel.errs, metamodel.cov, metamodel.chisq
    )


def _chisq(imap, pred):
    resid = imap.data - pred
    return float(jnp.sum(resid * imap.noise.apply_noise(resid)))


def _wmean(imap, arr):
    return jnp.sum(arr * imap.ivar) / jnp.sum(imap.ivar)


def _solve_amp(imap, base, template):
    """Best fit amplitude of `template` on top of `base`."""
    resid = imap.data - base
    tn = imap.noise.apply_noise(template)
    return float(jnp.sum(tn * resid) / jnp.sum(tn * template))


def main():
    args = _parse_args()
    labels = args.labels or [
        os.path.basename(os.path.dirname(os.path.abspath(r))) for r in args.results
    ]
    if len(labels) != len(args.results):
        raise ValueError("Need one label per results file")

    # ACT-only view of the joint config
    cfg = load_config({}, args.config)
    act = args.dataset
    cfg["datasets"] = {act: cfg["datasets"][act]}
    cfg["metamodel"]["model_map"] = {act: cfg["metamodel"]["model_map"][act]}
    cfg["paths"]["subdir"] = os.path.join(
        cfg["paths"].get("subdir", ""), "act_diagnostic"
    )
    cfg["fit"] = False
    cfg["sim"] = False
    _, outdir, _, metamodel = load_cfg(cfg)

    dset_ind = metamodel.get_dataset_ind(act)
    maps = list(metamodel.datasets[dset_ind].datavec)
    model_names = [model.name for model in metamodel.models]
    ps_model = model_names.index(args.ps_model)
    ps_ind = metamodel.parameter_map[ps_model][
        list(metamodel.models[ps_model].par_names).index(args.ps_par)
    ]

    # The map itself
    print(f"\nOutputs in {outdir}")
    cluster = metamodel.models[metamodel.model_map[dset_ind][0]]
    xg, yg = np.asarray(cluster.xyz[0]).ravel(), np.asarray(cluster.xyz[1]).ravel()
    for imap in maps:
        x, y = (np.asarray(a) * wu.rad_to_arcsec for a in imap.xy)
        inside = (x >= xg.min()) & (x <= xg.max()) & (y >= yg.min()) & (y <= yg.max())
        sigma_white = float(jnp.mean(1 / jnp.sqrt(imap.ivar)))
        print(f"\nMap {imap.name}: {imap.data.shape}")
        print(f"  map std            {float(jnp.std(imap.data)):10.2f} uK")
        print(f"  white noise sigma  {sigma_white:10.2f} uK")
        print(f"  chi2 with no model {_chisq(imap, 0.0):10.1f} for {imap.data.size} px")
        print(
            f"  pixels inside the model grid {inside.mean():.1%} (zero model outside)"
        )

    # Chi2 for each parameter set
    rows = []
    profiles = {}
    for label, path in zip(labels, args.results):
        pars, matched, unmatched = _load_pars(path, metamodel)
        if unmatched:
            print(f"\n[{label}] not in this metamodel, ignored: {unmatched}")
        if not np.all(matched):
            missing = [n for n, m in zip(metamodel.par_names, matched) if not m]
            print(f"[{label}] not in results, using config values: {missing}")

        mm = _with_pars(metamodel, pars)
        check = float(chisq_objective(mm, dset_ind, True, False, False)[0])
        amp = pars[ps_ind]
        pars_1 = pars.copy()
        pars_1[ps_ind] = amp + 1.0
        mm_1 = _with_pars(metamodel, pars_1)

        chi_saved = chi_solved = chi_zm = 0.0
        amps, amps_zm = [], []
        for i, imap in enumerate(maps):
            pred = mm.model_proj(dset_ind, i)
            template = mm_1.model_proj(dset_ind, i) - pred
            base = pred - amp * template
            chi_saved += _chisq(imap, pred)

            a = _solve_amp(imap, base, template)
            pred_solved = base + a * template
            chi_solved += _chisq(imap, pred_solved)
            amps.append(a)

            base_zm = base - _wmean(imap, base)
            template_zm = template - _wmean(imap, template)
            a_zm = _solve_amp(imap, base_zm, template_zm)
            pred_zm = base_zm + a_zm * template_zm
            chi_zm += _chisq(imap, pred_zm)
            amps_zm.append(a_zm)

            _write(
                imap,
                pred_solved,
                os.path.join(outdir, f"{imap.name}_{label}_model.fits"),
            )
            _write(
                imap,
                imap.data - pred_solved,
                os.path.join(outdir, f"{imap.name}_{label}_residual.fits"),
            )
            profiles.setdefault(i, {})[label] = np.asarray(pred_zm)

        if not np.isclose(check, chi_saved, rtol=1e-6):
            print(f"[{label}] WARNING: chisq_objective gives {check}, here {chi_saved}")
        shown = {
            n: float(pars[i])
            for i, n in enumerate(metamodel.par_names)
            if bool(metamodel.to_fit_ever[i])
        }
        rows.append((label, chi_saved, chi_solved, chi_zm, amp, amps, amps_zm, shown))

    print("\nACT chi2 (lower is better)")
    print(f"{'label':>20} {'as_saved':>12} {'ps_solved':>12} {'zero_mean':>12}")
    best = min(r[2] for r in rows)
    for label, c0, c1, c2, *_ in rows:
        print(
            f"{label:>20} {c0:12.1f} {c1:12.1f} {c2:12.1f}   d(ps_solved) = {c1 - best:+.1f}"
        )
    print(f"\nACT {args.ps_model}.{args.ps_par}: saved / solved / solved zero-mean")
    for label, _, _, _, amp, amps, amps_zm, _ in rows:
        print(f"{label:>20} {amp:10.2f} / {np.round(amps, 2)} / {np.round(amps_zm, 2)}")
    print("\nFitted parameters used")
    for label, *_, shown in rows:
        print(f"{label:>20} " + ", ".join(f"{k}={v:.4g}" for k, v in shown.items()))

    # Radial profiles around the grid centre
    x0 = float(eval(str(cfg["coords"]["x0"])))
    y0 = float(eval(str(cfg["coords"]["y0"])))
    r_grid = float(eval(str(cfg["coords"]["r_map"]))) / 60.0
    for i, imap in enumerate(maps):
        ra, dec = (np.asarray(a) for a in imap.xy)
        r = np.rad2deg(np.hypot((ra - x0) * np.cos(y0), dec - y0)) * 60.0
        edges = np.arange(0, r.max() + args.bin, args.bin)
        mids = 0.5 * (edges[1:] + edges[:-1])
        which = np.digitize(r.ravel(), edges) - 1
        nbin = len(mids)

        def prof(arr):
            return np.array(
                [
                    arr.ravel()[which == b].mean() if np.any(which == b) else np.nan
                    for b in range(nbin)
                ]
            )

        data = np.asarray(imap.data)
        counts = np.bincount(which, minlength=nbin)[:nbin]
        err = float(jnp.mean(1 / jnp.sqrt(imap.ivar))) / np.sqrt(np.maximum(counts, 1))

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.axvline(r_grid, color=MUTED, lw=1, ls=":")
        ax.annotate(
            "model grid edge",
            (r_grid, 1),
            xycoords=("data", "axes fraction"),
            xytext=(4, -12),
            textcoords="offset points",
            color=MUTED,
            fontsize=9,
        )
        ax.errorbar(
            mids,
            prof(data),
            yerr=err,
            fmt="o",
            ms=5,
            color=INK,
            lw=1,
            label="ACT data (white-noise errors only)",
        )
        for j, (label, pred) in enumerate(profiles[i].items()):
            ax.plot(mids, prof(pred), color=SERIES_COLORS[j % 8], lw=2, label=label)
        ax.set_xlabel("Radius from grid centre [arcmin]")
        ax.set_ylabel("Azimuthal mean [uK_CMB]")
        ax.set_title(
            f"{imap.name}: data vs models (zero-mean, ACT PS amp re-solved)",
            fontsize=10,
        )
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()
        fpath = os.path.join(outdir, f"{imap.name}_radial_profile.png")
        fig.savefig(fpath, dpi=150)
        plt.close(fig)
        print(f"\nRadial profile plot: {fpath}")


def _write(imap, arr, path):
    # load_maps transposes the FITS data, so transpose back
    hdu = fits.PrimaryHDU(data=np.asarray(arr).T, header=imap.wcs.to_header())
    hdu.writeto(path, overwrite=True)


if __name__ == "__main__":
    main()
