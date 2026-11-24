"""
Are the two datasets of a joint fit consistent, and does the second one (ACT) accept the first one's (M2) solution?
Written for MOOJ1142 (joint M2 + ACT fit vs the M2 only fit), read by notebooks/joint_consistency.ipynb.

Everything is evaluated on one fixed set of data and noise: the joint run's data (M2 TODs and the ACT map), with the M2
noise estimated in turn from the data alone (no model subtracted, as aic_refit) and from the residual of each run's MCMC
medians, since a residual based noise favours its reference model (the ACT noise is ivar + CMB and doesn't depend on
the model).

1. Posterior predictive: chi-squared of draws from each run's posterior on each dataset separately. The ACT point
   source amplitude isn't in the M2 only model and enters the ACT model linearly, so it is set to its best value
   (within its prior) for every draw of both posteriors.
2. Q_DMAP (Raveri & Hu 2019): Q = chi2_joint,min - chi2_M2,min - chi2_ACT,min, each minimum an LM fit of the
   parameters that dataset sees, from both runs' medians, keeping the lower. Q splits into what the joint fit costs
   each dataset, chi2_M2(joint min) - chi2_M2,min and chi2_ACT(joint min) - chi2_ACT,min. It is compared to a chi2
   with N_eff(M2) + N_eff(ACT) - N_eff(joint) degrees of freedom, N_eff = tr(F (F + C_prior^-1)^-1) with each flat
   prior taken as a Gaussian of the same variance and F the dataset's Fisher matrix at its own minimum.

Run it under MPI like the joint fit, one task per TOD plus one for the ACT map:

    srun -n 107 python scripts/joint_consistency.py <joint outdir> <M2 only outdir> --out consistency.yaml
    srun -n 5 python scripts/joint_consistency.py <joint outdir> <M2 only outdir> --out consistency.yaml --debug

or with scripts/joint_consistency.sbatch on Trillium.
"""

import argparse
import os
from copy import copy
from typing import TYPE_CHECKING, Callable

import numpy as np
import yaml

from witch import model_comparison as wmc

if TYPE_CHECKING:
    from witch.containers import MetaModel


def dataset_mask(cfg: dict, mm: "MetaModel", name: str) -> np.ndarray:
    """
    Parameters that enter a dataset's models, tied parameters included.

    Parameters
    ----------
    cfg : dict
        The run's config.
    mm : MetaModel
        The metamodel.
    name : str
        Dataset name.

    Returns
    -------
    np.ndarray
        Boolean mask over `mm.parameters`.
    """
    mlist = cfg["metamodel"]["models"]
    mask = np.zeros(len(mm.parameters), dtype=bool)
    for model in cfg["metamodel"]["model_map"][name]:
        mask[np.asarray(mm.parameter_map[mlist.index(model)])] = True
    return mask


def per_dataset(
    mm: "MetaModel",
    pars: np.ndarray,
    objectives: dict[str, Callable],
    names: list[str],
    grad_names: tuple[str, ...] = (),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Chi-squared, gradient and curvature of each dataset separately, summed over ranks. Collective.

    Parameters
    ----------
    mm : MetaModel
        The metamodel, with the data and noise to use.
    pars : np.ndarray
        Parameters to evaluate at.
    objectives : dict[str, Callable]
        Each dataset's own objective (the metamodel's may be swapped for a fit).
    names : list[str]
        All dataset names, the order of the outputs.
    grad_names : tuple[str, ...], default: ()
        Datasets to compute the gradient and curvature of, zeros for the others.

    Returns
    -------
    chisq : np.ndarray
        `(ndataset,)`.
    grad : np.ndarray
        `(ndataset, npar)`, `J^T N^-1 r` (minus half the chi-squared gradient).
    curve : np.ndarray
        `(ndataset, npar, npar)`, `J^T N^-1 J` (the Fisher matrix).
    """
    import jax.numpy as jnp

    npar = len(pars)
    trial = copy(mm).update(jnp.asarray(pars), mm.errs, mm.cov, mm.chisq)
    chisq = np.zeros(len(names))
    grad = np.zeros((len(names), npar))
    curve = np.zeros((len(names), npar, npar))
    for i, dset in enumerate(trial.datasets):
        k, do = names.index(dset.name), dset.name in grad_names
        c, g, cu = objectives[dset.name](trial, i, True, do, do)
        chisq[k] += float(c)
        if do:
            grad[k] += np.asarray(g)
            curve[k] += np.asarray(cu)
    comm = mm.global_comm
    return comm.allreduce(chisq), comm.allreduce(grad), comm.allreduce(curve)


def minimize(
    mm: "MetaModel",
    mask: np.ndarray,
    null_names: tuple[str, ...],
    starts: dict[str, np.ndarray],
    maxiter: int,
    chitol: float,
) -> tuple[np.ndarray, str]:
    """
    LM fit of the masked parameters to the datasets not in `null_names`, from each start, keeping the lowest.
    The metamodel's objectives and round are restored afterwards. Collective.

    Parameters
    ----------
    mm : MetaModel
        The metamodel.
    mask : np.ndarray
        Parameters to fit.
    null_names : tuple[str, ...]
        Datasets left out of the fit.
    starts : dict[str, np.ndarray]
        Named starting points.
    maxiter : int
        LM iterations.
    chitol : float
        LM convergence tolerance.

    Returns
    -------
    pars : np.ndarray
        The best fit.
    start : str
        The start it came from.
    """
    import jax.numpy as jnp

    from witch.fitting import run_lmfit
    from witch.objective import null_objective

    saved = [d.objective for d in mm.datasets]
    for d in mm.datasets:
        if d.name in null_names:
            d.objective = null_objective
    full_round = int(
        mm.cur_round
    )  # MetaModel round properties are jax arrays, the models need ints
    mm.set_round(int(mm.n_rounds) - 1).add_round(jnp.asarray(mask))
    best = (np.inf, None, None)
    try:
        for name, start in starts.items():
            trial = copy(mm).update(jnp.asarray(start), mm.errs, mm.cov, mm.chisq)
            fit, *_ = run_lmfit(trial, maxiter, chitol)
            pars = mm.global_comm.bcast(np.array(fit.parameters, dtype=float), root=0)
            chisq = mm.global_comm.bcast(float(np.asarray(fit.chisq)), root=0)
            if chisq < best[0]:
                best = (chisq, pars, name)
    finally:
        for d, obj in zip(mm.datasets, saved):
            d.objective = obj
        mm.set_round(full_round)
    return best[1], best[2]


def summary(x: np.ndarray) -> dict:
    """Median and 16-84 percentiles."""
    return {
        "median": float(np.median(x)),
        "p16_p84": [float(v) for v in np.percentile(x, [16, 84])],
    }


def main() -> None:
    """Posterior predictive chi-squared per dataset and Q_DMAP, for each noise reference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "joint", help="Output directory of the joint ACT + M2 run (data, joint model)"
    )
    parser.add_argument(
        "m2", help="Output directory of the M2 only run (its posterior)"
    )
    parser.add_argument(
        "--out",
        required=True,
        help="YAML file for the results (draws go next to it as .npz)",
    )
    parser.add_argument("--act", default="act", help="Name of the ACT dataset")
    parser.add_argument(
        "--ndraws", type=int, default=300, help="Draws from each posterior"
    )
    parser.add_argument(
        "--maxiter", type=int, default=40, help="LM iterations per minimisation (as fitting.maxiter)"
    )
    parser.add_argument(
        "--noise",
        nargs="+",
        choices=["initial", "joint_median", "m2_median"],
        default=["initial", "joint_median", "m2_median"],
        help="M2 noise models to evaluate under: from the data with no model subtracted (as aic_refit), or from "
        "the residual of either run's MCMC medians",
    )
    parser.add_argument(
        "--debug", action="store_true", help="4 TODs, 3 draws, 2 LM steps"
    )
    args = parser.parse_args()

    import jax
    import jax.numpy as jnp

    from witch.containers import MetaModel
    from witch.fitter import _initial_noise, _reestimate_noise

    cfg, mm = wmc.reload_run(os.path.abspath(args.joint), args.debug)
    m2, _ = MetaModel.load(os.path.join(args.m2, "results_mcmc.dill"))
    n = len(m2.parameters)
    if list(mm.par_names[:n]) != list(m2.par_names):
        raise ValueError(
            "The joint run's leading parameters aren't the M2 only model's"
        )
    comm = mm.global_comm
    rank0 = comm.Get_rank() == 0
    names = list(cfg["datasets"])
    ka = names.index(args.act)
    km = 1 - ka
    objectives = {d.name: d.objective for d in mm.datasets}
    mm.set_round(int(mm.n_rounds) - 1)  # the full fit round
    free = np.asarray(mm.to_fit)
    lower, upper = (np.asarray(p, dtype=float) for p in mm.priors)
    masks = {name: free & dataset_mask(cfg, mm, name) for name in names}
    only_act = np.flatnonzero(masks[args.act] & ~masks[names[km]])
    if len(only_act) != 1:
        raise ValueError(f"Expected one parameter only ACT sees, got {only_act}")
    ia = int(only_act[0])

    pj = np.array(mm.parameters, dtype=float)
    pm = pj.copy()
    pm[:n] = np.array(m2.parameters, dtype=float)
    medians = {"joint_median": pj, "m2_median": pm}

    ndraws = 3 if args.debug else args.ndraws
    maxiter = 2 if args.debug else args.maxiter
    chitol = eval(str(cfg["fitting"].get("chitol", "1e-5")))
    rng = np.random.default_rng(0)  # same draws on every rank
    draws = {}
    for label, outdir, base in (("joint", args.joint, pj), ("m2", args.m2, pm)):
        s = np.load(os.path.join(outdir, "samples_mcmc.npz"))
        pars = np.repeat(base[None], ndraws, axis=0)
        pars[:, np.flatnonzero(s["to_fit"])] = s["samples"][
            rng.choice(len(s["samples"]), ndraws)
        ]
        draws[label] = pars

    def evaluate(pars: np.ndarray) -> tuple[float, float, float]:
        """M2 chi2, ACT chi2 with the ACT amplitude at its best, and that amplitude."""
        chisq, grad, curve = per_dataset(mm, pars, objectives, names, (args.act,))
        act, amp = wmc.profile_linear(
            chisq[ka], grad[ka], curve[ka], pars, ia, lower[ia], upper[ia]
        )
        return float(chisq[km]), act, amp

    results, arrays = {}, {}
    for ref in args.noise:
        if ref == "initial":
            _initial_noise(mm)
        else:
            _reestimate_noise(
                copy(mm).update(jnp.asarray(medians[ref]), mm.errs, mm.cov, mm.chisq)
            )
        jax.clear_caches()
        if rank0:
            print(f"M2 noise: {ref}", flush=True)

        # Minima: joint, M2 alone (ACT amplitude fixed, it doesn't enter), ACT alone (M2 only parameters fixed)
        mins, start_of = {}, {}
        for label, null in (
            ("joint", ()),
            (names[km], (args.act,)),
            (args.act, (names[km],)),
        ):
            mask = free if label == "joint" else masks[label]
            mins[label], start_of[label] = minimize(
                mm, mask, null, medians, maxiter, chitol
            )
        at = {}
        for label, pars in mins.items():
            chisq, _, curve = per_dataset(mm, pars, objectives, names, tuple(names))
            at[label] = (chisq, curve)
        chi_j = at["joint"][0]
        chi_m2_min = float(at[names[km]][0][km])
        chi_act_min = float(at[args.act][0][ka])
        cost_m2 = float(chi_j[km] - chi_m2_min)
        cost_act = float(chi_j[ka] - chi_act_min)
        neff = {
            "joint": wmc.n_eff(at["joint"][1].sum(axis=0), free, lower, upper),
            names[km]: wmc.n_eff(at[names[km]][1][km], masks[names[km]], lower, upper),
            args.act: wmc.n_eff(at[args.act][1][ka], masks[args.act], lower, upper),
        }
        tension = wmc.q_dmap(
            cost_m2, cost_act, neff[names[km]], neff[args.act], neff["joint"]
        )

        # Posterior predictive, relative to each dataset's own minimum
        point = {k: evaluate(v) for k, v in medians.items()}
        dd = {}
        for label, pars in draws.items():
            ev = np.array([evaluate(row) for row in pars])
            dd[label] = ev
            arrays[f"{ref}_{label}"] = ev
        if rank0:
            results[ref] = {
                "Q_DMAP": {
                    "Q": tension["Q"],
                    "cost_to_m2": cost_m2,
                    "cost_to_act": cost_act,
                    "n_eff": neff,
                    "dof": tension["dof"],
                    "p": tension["p"],
                    "sigma": tension["sigma"],
                    "minima_start": start_of,
                    "chisq_min": {
                        "joint_total": float(chi_j.sum()),
                        names[km]: float(chi_m2_min),
                        args.act: float(chi_act_min),
                    },
                },
                "medians": {
                    k: {
                        f"dchisq_{names[km]}": v[0] - chi_m2_min,
                        f"dchisq_{args.act}": v[1] - chi_act_min,
                        "act_amp_best": v[2],
                    }
                    for k, v in point.items()
                },
                "posterior_draws": {
                    label: {
                        "n": len(ev),
                        f"dchisq_{names[km]}": summary(ev[:, 0] - chi_m2_min),
                        f"dchisq_{args.act}": summary(ev[:, 1] - chi_act_min),
                        "act_amp_best": summary(ev[:, 2]),
                    }
                    for label, ev in dd.items()
                },
                "minima": {
                    label: {
                        str(mm.par_names[i]) + f"[{i}]": float(pars[i])
                        for i in np.flatnonzero(free)
                    }
                    for label, pars in mins.items()
                },
            }
            print(
                yaml.safe_dump({ref: results[ref]["Q_DMAP"]}, sort_keys=False),
                flush=True,
            )

    if rank0:
        out = {
            "data": f"{os.path.abspath(args.joint)} (M2 TODs + ACT map, current loaders)",
            "posteriors": {
                "joint": os.path.abspath(args.joint),
                "m2": os.path.abspath(args.m2),
            },
            "dchisq": "relative to each dataset's own minimum under the same noise",
            **results,
        }
        with open(args.out, "w") as f:
            yaml.safe_dump(out, f, sort_keys=False)
        np.savez(
            os.path.splitext(args.out)[0] + "_draws.npz",
            **arrays,
            **{f"pars_{k}": v for k, v in draws.items()},
        )
        print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
