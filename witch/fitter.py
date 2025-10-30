"""
Master fitting and map making script.
You typically want to run the `witcher` command instead of this.
"""

import argparse as argp
import os
import sys
import time
from copy import copy, deepcopy
from importlib import import_module

import corner
import iteround
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mpi4jax
import numpy as np
import yaml
from mpi4py import MPI
from typing_extensions import Any, Unpack

from . import utils as wu
from .containers import MetaModel, Model, Model_xfer
from .dataset import DataSet
from .fitting import run_lmfit, run_mcmc
from .nonparametric import para_to_non_para
from .objective import joint_objective

comm = MPI.COMM_WORLD.Clone()


def print_once(*args: Unpack[tuple[Any, ...]]):
    """
    Helper function to print only once when running with MPI.
    Only the rank 0 process will print.

    Parameters
    ----------
    *args : Unpack[tuple[Any, ...]]
        Arguments to pass to print.
    """
    if comm.Get_rank() == 0:
        print(*args)
        sys.stdout.flush()


def _make_parser() -> argp.ArgumentParser:
    # Parse arguments
    parser = argp.ArgumentParser(
        description="Fit cluster profiles using mikasi and minkasi_jax"
    )
    parser.add_argument("config", help="Path to config file")
    parser.add_argument(
        "--nofit",
        "-nf",
        action="store_true",
        help="Don't actually fit, just use values from config",
    )
    parser.add_argument(
        "--nosub",
        "-ns",
        action="store_true",
        help="Don't subtract the sim'd or fit model. Useful for mapmaking a sim'd cluster",
    )
    parser.add_argument(
        "--wnoise",
        "-wn",
        action="store_true",
        help="Use whitenoise instead of map noise. Only for use with sim",
    )
    parser.add_argument(
        "--cpu",
        "-cpu",
        default=False,
        type=bool,
        help="If True, then run on CPU. Otherwise and by default run GPU.",
    )
    return parser


def _mpi_fsplit(fnames, comm):
    rank = comm.Get_rank()
    nproc = comm.Get_size()

    # Check that we have the minimum number of procs
    if nproc < len(fnames):
        raise ValueError("More datasets than procs!")

    # Kill extra procs
    flat_fnames = [
        f for fname in [fnames[dset] for dset in fnames.keys()] for f in fname
    ]
    if nproc > len(flat_fnames):
        nproc = len(flat_fnames)
        group = comm.Get_group()
        new_group = group.Incl(list(range(nproc)))
        new_comm = comm.Create(new_group)
        if rank >= len(flat_fnames):
            print(f"More procs than files!, exiting process {rank}")
            MPI.Finalize()
            sys.exit(0)
        comm = new_comm

    # Decide the number of procs for each dataset
    nprocs = {
        dset: 1
        + (nproc - len(fnames))
        * (len(fnames[dset]) - 1)
        / len(flat_fnames)  # (len(flat_fnames) - len(fnames))
        for dset in fnames.keys()
    }
    nprocs = iteround.saferound(nprocs, 0)
    dsets = np.array(
        [
            ds
            for dset in [
                [dset_name] * int(nprocs[dset_name]) for dset_name in fnames.keys()
            ]
            for ds in dset
        ]
    )

    # Die if we messed up
    for dset, f in fnames.items():
        if len(f) < nprocs[dset]:
            raise ValueError(
                f"Too many procs allocated to {dset}! This shouldn't be possible please report it!"
            )

    # Handle my local case
    dset_local = dsets[rank]
    group = comm.Get_group()
    comms_local = {
        dset: comm.Create(group.Incl(list(np.where(dsets == dset)[0])))
        for dset in fnames.keys()
    }
    local_rank = comms_local[dset_local].Get_rank()
    fnames_local = {
        dset: (
            fnames[dset][local_rank :: int(nprocs[dset])] if dset == dset_local else []
        )
        for dset in fnames.keys()
    }

    return fnames_local, comm, comms_local


def process_tods(cfg, dataset, metamodel):
    todvec = dataset.datavec
    info = dataset.info
    rank = todvec.comm.Get_rank()
    nproc = todvec.comm.Get_size()
    sim = cfg.get("sim", False)
    if metamodel is None and sim:
        raise ValueError("metamodel cannot be None when simming!")
    metamodel_cur = metamodel
    xfer = info.get("info", "")
    if xfer:
        metamodel_cur = deepcopy(metamodel)
        metamodel_cur.models = tuple(
            Model_xfer.from_parent(model, xfer) for model in metamodel_cur.models
        )

    dset_ind = metamodel_cur.get_dataset_ind(dataset.name)
    for i, tod in enumerate(todvec.tods):
        if sim:
            if cfg["wnoise"]:
                temp = jnp.percentile(
                    jnp.diff(tod.data, axis=-1), jnp.array([33.0, 68.0]), axis=-1
                )
                scale = (temp[1] - temp[0]) / jnp.sqrt(8)
                tod.data = (
                    jax.random.normal(
                        jax.random.key(0), shape=tod.data.shape, dtype=tod.data.dtype
                    )
                    * scale[..., None]
                )
            else:
                tod.data *= tod.data * (-1) ** ((rank + nproc * i) % 2)

            pred = metamodel_cur.model_proj(dset_ind, i)
            tod.data = tod.data + pred
        tod.data = tod.data - jnp.mean(tod.data, axis=-1)[..., None]
        tod.compute_noise(
            dataset.noise_class, None, *dataset.noise_args, **dataset.noise_kwargs
        )
    return todvec


def process_maps(cfg, dataset, metamodel):
    mapset = dataset.datavec
    info = dataset.info
    sim = cfg.get("sim", False)
    if metamodel is None and sim:
        raise ValueError("metamodel cannot be None when simming!")
    metamodel_cur = metamodel
    xfer = info.get("info", "")
    if xfer:
        metamodel_cur = deepcopy(metamodel)
        metamodel_cur.models = tuple(
            Model_xfer.from_parent(model, xfer) for model in metamodel_cur.models
        )

    dset_ind = metamodel_cur.get_dataset_ind(dataset.name)
    for i, imap in enumerate(mapset):
        if sim:
            if jnp.all(imap.ivar == 0):
                print_once(
                    f"ivar for map {imap.name} is all 0! Filling with a dummy value but you should check your maps!"
                )
                imap.ivar = imap.ivar.at[:].set(cfg.get("default_ivar", 1e8))
            scale = 1.0 / jnp.sqrt(imap.ivar)
            avg_scale = np.nanmean(scale)
            print("Map noise: ", avg_scale)
            scale = jnp.nan_to_num(
                scale, nan=avg_scale, posinf=avg_scale, neginf=avg_scale
            )
            imap.data = (
                jax.random.normal(
                    jax.random.key(0), shape=imap.data.shape, dtype=imap.data.dtype
                )
                * scale
            )
            if not cfg["wnoise"]:
                print_once(
                    "Only white noise currently supported for map sims... simming map with white noise"
                )

            pred = metamodel_cur.model_proj(dset_ind, i)
            imap.data = imap.data + pred
        print("Map scale: ", jnp.mean(jnp.abs(imap.data)))
        imap.data = imap.data - jnp.mean(imap.data)
        imap.compute_noise(
            dataset.noise_class, None, *dataset.noise_args, **dataset.noise_kwargs
        )
    return mapset


def get_outdir(cfg, metamodel, nonpara=False):
    outroot = cfg["paths"]["outroot"]
    if not os.path.isabs(outroot):
        outroot = os.path.join(
            os.environ.get("WITCH_OUTROOT", os.environ["HOME"]), outroot
        )

    name = ""
    if metamodel is not None:
        name = "_".join([model.name for model in metamodel.models]) + (
            "_ns" * (not cfg["sub"])
        )
    outdir = os.path.join(outroot, cfg["name"], name)
    if "subdir" in cfg["paths"]:
        outdir = os.path.join(outdir, cfg["paths"]["subdir"])
    if metamodel is not None:
        if cfg["fit"]:
            outdir = os.path.join(
                outdir,
                "-".join(
                    [
                        name
                        for name, to_fit in zip(
                            metamodel.par_names, metamodel.to_fit_ever
                        )
                        if to_fit
                    ]
                ),
            )
        else:
            outdir = os.path.join(outdir, "not_fit")
    if cfg["sim"] and metamodel is not None:
        outdir += "-sim"
    if nonpara:
        outdir += "-nonpara"

    print_once("Outputs can be found in", outdir)
    if comm.Get_rank() == 0:
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, "config.yaml"), "w") as file:
            yaml.dump(cfg, file)

    return outdir


def _save_model(cfg, metamodel, desc_str):
    outdir = cfg["outdir"]
    if comm.Get_rank() != 0:
        return
    res_path = os.path.join(outdir, f"results_{desc_str}.dill")
    print_once("Saving results to", res_path)
    metamodel.save(res_path)

    final = {"metamodel": cfg["metamodel"]}
    for model in metamodel.models:
        final[model.name] = cfg[model.name]
        for i, (struct_name, structure) in zip(
            model.original_order, cfg[model.name]["structures"].items()
        ):
            model_struct = model.structures[i]
            for par, par_name in zip(
                model_struct.parameters, structure["parameters"].keys()
            ):
                final[model.name]["structures"][struct_name]["parameters"][par_name][
                    "value"
                ] = [float(cur_par) for cur_par in par.val]
    with open(os.path.join(outdir, f"results_{desc_str}.yaml"), "w") as file:
        yaml.dump(final, file)


def _reestimate_noise(metamodel):
    for i, dataset in enumerate(metamodel.datasets):
        for j, data in enumerate(dataset.datavec):
            pred = metamodel.model_proj(i, j)
            data.compute_noise(
                dataset.noise_class,
                data.data - pred,
                *dataset.noise_args,
                **dataset.noise_kwargs,
            )
    return metamodel


def _run_fit(
    cfg,
    metamodel,
    r,
):
    metamodel.set_round(r)
    to_fit = np.array(metamodel.to_fit)
    print_once(f"Starting round {r+1} of fitting with {np.sum(to_fit)} pars free")
    t1 = time.time()
    models, i, delta_chisq = run_lmfit(
        metamodel,
        eval(str(cfg["fitting"].get("maxiter", "10"))),
        eval(str(cfg["fitting"].get("chitol", "1e-5"))),
    )
    _ = mpi4jax.barrier(comm=comm)
    t2 = time.time()
    print_once(
        f"Took {t2 - t1} s to fit with {i} iterations and final delta chisq of {delta_chisq}"
    )

    print_once(models)
    _save_model(cfg, metamodel, f"fit{r}")

    metamodel = _reestimate_noise(metamodel)
    return metamodel


def _run_mcmc(cfg, metamodel):
    print_once("Running MCMC")
    init_pars = np.array(metamodel.parameters.copy())
    t1 = time.time()
    metamodel, samples = run_mcmc(
        metamodel,
        num_steps=int(cfg["mcmc"].get("num_steps", 5000)),
        num_leaps=int(cfg["mcmc"].get("num_leaps", 10)),
        step_size=float(cfg["mcmc"].get("step_size", 0.02)),
        sample_which=int(cfg["mcmc"].get("sample_which", -1)),
        burn_in=float(cfg["mcmc"].get("burn_in", 0.1)),
        max_tries=int(cfg["mcmc"].get("max_tries", 20)),
    )
    _ = mpi4jax.barrier(comm=comm)
    t2 = time.time()
    print_once(f"Took {t2 - t1} s to run mcmc")

    message = str(metamodel).split("\n")
    message[1] = "MCMC estimated pars:"
    print_once("\n".join(message))

    _save_model(cfg, metamodel, "mcmc")
    if comm.Get_rank() == 0:
        samples = np.array(samples)
        samps_path = os.path.join(cfg["outdir"], f"samples_mcmc.npz")
        print_once("Saving samples to", samps_path)
        np.savez_compressed(samps_path, samples=samples)
        try:
            to_fit = np.array(metamodel.to_fit)
            # ranges = [prior if prior is not None else [0.5 * model.params[i], 2 * model.params[i]] for i, prior in enumerate(model.priors)]
            corner.corner(
                samples,
                labels=np.array(metamodel.par_names)[to_fit],
                truths=init_pars[to_fit],
            )
            plt.savefig(os.path.join(cfg["outdir"], "corner.png"))
        except Exception as e:
            print_once(f"Failed to make corner plot with error: {str(e)}")
    else:
        # samples are incomplete on non root procs
        del samples

    metamodel = _reestimate_noise(metamodel)
    return metamodel


def fit_loop(metamodel, cfg, comm):
    if metamodel is None:
        raise ValueError("Can't fit without a model defined!")
    if cfg["sim"]:
        # Remove structs we deliberately want to leave out of model
        metamodel = metamodel.remove_structs(cfg)

        params = jnp.array(metamodel.parameters)
        par_offset = cfg.get("par_offset", 1.1)
        params = params.at[metamodel.to_fit_ever].multiply(
            par_offset
        )  # Don't start at exactly the right value
        metamodel = metamodel.update(params, metamodel.errors, metamodel.chisq)

    print_once("Compiling objective function")
    t0 = time.time()
    chisq, *_ = joint_objective(metamodel)
    metamodel = metamodel.update(metamodel.parameters, metamodel.errors, chisq)
    print_once(f"Took {time.time() - t0} s to compile")

    message = str(metamodel).split("\n")
    message[1] = "Starting pars:"
    print_once("\n".join(message))
    for r in range(metamodel.n_rounds):
        metamodel = _run_fit(
            cfg,
            metamodel,
            r,
        )
        _ = mpi4jax.barrier(comm=comm)

    if "mcmc" in cfg and cfg["mcmc"].get("run", True):
        metamodel = _run_mcmc(
            cfg,
            metamodel,
        )
        _ = mpi4jax.barrier(comm=comm)

    # Save final pars
    _save_model(cfg, metamodel, "final_fit")

    return metamodel


def deep_merge(a: dict, b: dict) -> dict:
    """
    Based on https://gist.github.com/angstwad/bf22d1822c38a92ec0a9?permalink_comment_id=3517209
    """
    result = deepcopy(a)
    for bk, bv in b.items():
        av = result.get(bk)
        if isinstance(av, dict) and isinstance(bv, dict):
            result[bk] = deep_merge(av, bv)
        else:
            result[bk] = deepcopy(bv)
    return result


def load_config(start_cfg, cfg_path):
    """
    We want to load a config and if it has the key "base",
    load that as well and merge them.
    We only want to take things from base that are not in the original config
    so we merge the original into the newly loaded one.
    """
    with open(cfg_path) as file:
        new_cfg = yaml.safe_load(file)
    cfg = deep_merge(new_cfg, start_cfg)
    if "base" in new_cfg:
        base_path = new_cfg["base"]
        if not os.path.isabs(base_path):
            base_path = os.path.join(os.path.dirname(cfg_path), base_path)
        return load_config(cfg, base_path)
    return cfg


def main():
    if "MJ_TODROOT" in os.environ:
        if "WITCH_DATROOT" not in os.environ:
            print_once(
                "You are using MJ_TODROOT, this is depreciated in favor of WITCH_DATROOT. Setting WITCH_DATROOT to the value of MJ_TODROOT..."
            )
            os.environ["WITCH_DATROOT"] = os.environ["MJ_TODROOT"]
        else:
            print_once(
                "Both WITCH_DATROOT and MJ_TODROOT provided! MJ_TODROOT is depreciated, using WITCH_DATROOT..."
            )

    if "MJ_OUTROOT" in os.environ:
        if "WITCH_OUTROOT" not in os.environ:
            print_once(
                "You are using MJ_OUTROOT, this is depreciated in favor of WITCH_OUTROOT. Setting WITCH_OUTROOT to the value of MJ_OUTROOT..."
            )
            os.environ["WITCH_OUTROOT"] = os.environ["MJ_OUTROOT"]
        else:
            print_once(
                "Both WITCH_OUTROOT and MJ_OUTROOT provided! MJ_OUTROOT is depreciated, using WITCH_OUTROOT..."
            )

    parser = _make_parser()
    args = parser.parse_args()

    if args.cpu == True:
        jax.config.update("jax_platform_name", "cpu")

    # TODO: Serialize cfg to a data class (pydantic?)
    cfg = load_config({}, args.config)
    cfg["fit"] = cfg.get("fit", "model" in cfg)
    cfg["sim"] = cfg.get("sim", False)
    cfg["wnoise"] = cfg.get("wnoise", False)
    cfg["map"] = cfg.get("map", True)
    cfg["sub"] = cfg.get("sub", True)
    cfg["mem_starved"] = cfg.get("mem_starved", False)
    if args.wnoise:
        cfg["wnoise"] = True
    if args.nosub:
        cfg["sub"] = False
    if args.nofit:
        cfg["fit"] = False

    # Do imports
    for module, name in cfg.get("imports", {}).items():
        mod = import_module(module)
        if isinstance(name, str):
            locals()[name] = mod
        elif isinstance(name, list):
            for n in name:
                locals()[n] = getattr(mod, n)
        else:
            raise TypeError("Expect import name to be a string or a list")

    # Get the functions needed to work with out dataset
    dset_names = list(cfg["datasets"].keys())
    models = []
    datasets = []
    outdir = None
    # First lets plan how to split things up among mpi tasks
    fnames = {}
    for dset_name in dset_names:
        get_files = eval(cfg["datasets"][dset_name]["funcs"]["get_files"])
        fnames[dset_name] = get_files(dset_name, cfg)
    global comm
    fnames, comm, comms_local = _mpi_fsplit(fnames, comm)

    for dset_name in dset_names:
        # If this dataset doesn't live in this proc then skip
        if len(fnames[dset_name]) == 0:
            continue

        if "load_tods" in cfg["datasets"][dset_name]["funcs"]:
            cfg["datasets"][dset_name]["funcs"]["load"] = cfg["datasets"][dset_name][
                "funcs"
            ]["load_tods"]
        elif "load_maps" in cfg["datasets"][dset_name]["funcs"]:
            cfg["datasets"][dset_name]["funcs"]["load"] = cfg["datasets"][dset_name][
                "funcs"
            ]["load_maps"]
        get_files = eval(cfg["datasets"][dset_name]["funcs"]["get_files"])
        load = eval(cfg["datasets"][dset_name]["funcs"]["load"])
        get_info = eval(cfg["datasets"][dset_name]["funcs"]["get_info"])
        make_metadata = eval(cfg["datasets"][dset_name]["funcs"]["make_metadata"])
        preproc = eval(cfg["datasets"][dset_name]["funcs"]["preproc"])
        postproc = eval(cfg["datasets"][dset_name]["funcs"]["postproc"])
        postfit = eval(cfg["datasets"][dset_name]["funcs"]["postfit"])
        if "xray" in dset_name:
            make_exp_maps = eval(cfg["datasets"][dset_name]["funcs"]["make_exp_maps"])
            make_back_map = eval(cfg["datasets"][dset_name]["funcs"]["make_back_map"])
            dataset = DataSet(
                dset_name,
                get_files,
                load,
                get_info,
                make_beam,
                make_exp_maps,
                make_back_map,
                preproc,
                postproc,
                postfit,
                comm,
            )
        else:
            # it gives this error: File "/home/elebar/joint/WITCH/witch/dataset.py", line 363, in __post_init__
            # assert isinstance(self.make_exp_maps, MakeExpMaps)
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            # AssertionError
            dataset = DataSet(
                dset_name,
                get_files,
                load,
                get_info,
                make_beam,
                jnp.array([[1]]),
                jnp.array([[0]]),
                preproc,
                postproc,
                postfit,
                comm,
            )

        # Get data
        dataset.datavec = load(
            dset_name, cfg, fnames[dset_name], comms_local[dset_name]
        )

        # Get any info we need specific to an expiriment
        dataset.info = get_info(dset_name, cfg, dataset.datavec)

        # Get the beam
        beam = make_beam(dset_name, cfg, comms_local[dset_name])
        dataset.beam = beam

        # Get the exp maps xray
        exp_maps = make_exp_maps(dset_name, cfg, comms_local[dset_name])
        dataset.exp_maps = exp_maps

        # Get the back map xray
        back_map = make_back_map(dset_name, cfg, comms_local[dset_name])
        dataset.back_map = back_map

        # Prefactor
        prefactor = dataset.info.get("prefactor", None)
        if prefactor is not None:
            dataset.prefactor = prefactor

        # Setup noise
        noise_class = eval(str(cfg["datasets"][dset_name]["noise"]["class"]))
        noise_args = tuple(eval(str(cfg["datasets"][dset_name]["noise"]["args"])))
        noise_kwargs = eval(str(cfg["datasets"][dset_name]["noise"]["kwargs"]))
        dataset.info["noise_class"] = noise_class
        dataset.info["noise_args"] = noise_args
        dataset.info["noise_kwargs"] = noise_kwargs

        if "base" in cfg.keys():
            del cfg[
                "base"
            ]  # We've collated to the cfg files so no need to keep the base
        datasets += [dataset]

    datasets = tuple(datasets)
    # Define the model and get stuff setup fitting
    if "metamodel" in cfg:
        metamodel = MetaModel.from_config(
            comm,
            cfg,
            datasets,
        )
    else:
        metamodel = MetaModel(
            comm,
            tuple(),
            datasets,
            tuple(),
            tuple(),
            jnp.zeros(0),
            jnp.zeros(0),
            jnp.zeros(0),
        )
        print_once("No model defined, setting fit, sim, and sub to False")
        cfg["fit"] = False
        cfg["sim"] = False
        cfg["sub"] = False
    outdir = get_outdir(cfg, metamodel)
    cfg["outdir"] = outdir

    # Now process
    for dataset in metamodel.datasets:
        os.makedirs(os.path.join(outdir, dataset.name), exist_ok=True)
        dataset.info["outdir"] = outdir
        dataset.preproc(dataset, cfg, metamodel)
        if dataset.mode == "tod":
            dataset.datavec = process_tods(cfg, dataset, metamodel)
        elif dataset.mode == "map":
            dataset.datavec = process_maps(cfg, dataset, metamodel)
        dataset = jax.block_until_ready(dataset)
        dataset.postproc(dataset, cfg, metamodel)

    # Now we fit
    to_fit = cfg.get("fit", True)
    if to_fit and outdir is not None:
        metamodel = fit_loop(metamodel, cfg, comm)
        for dataset in metamodel.datasets:
            dataset.postfit(dataset, cfg, metamodel)
        if "nonpara" in cfg and cfg["nonpara"]["convert"]:
            outdir = get_outdir(cfg, metamodel)
            cfg["outdir"] = outdir
            for dataset in metamodel.datasets:
                os.makedirs(os.path.join(outdir, dataset.name), exist_ok=True)
                dataset.info["outdir"] = outdir
            n_rounds = cfg["nonpara"].get("n_rounds", None)
            nonpara_models = []
            for model in metamodel.models:
                to_copy = cfg[model.name].get("to_copy", [])
                if len(to_copy) == 0:
                    nonpara_models += [copy(model)]
                    continue
                nonpara_model = para_to_non_para(
                    model, n_rounds=n_rounds, to_copy=to_copy
                )
                nonpara_models += [nonpara_model]
            nonpara_models = tuple(nonpara_models)
            par_map, pars, errs = _compute_par_map_and_pars(
                cfg.get("metamodel", {}), nonpara_models
            )
            nonparametamodel = copy(metamodel)
            nonparametamodel.parameter_map = par_map
            nonparametamodel.parameters = pars
            nonparametamodel.errors = errs
            nonparametamodel = nonparametamodel.update(pars, errs, metamodel.chisq)

            nonparametamodel = fit_loop(nonparametamodel, cfg, comm)
            for dataset in nonparametamodel.datasets:
                dataset.postfit(dataset, cfg, nonparametamodel)

    print_once("Outputs can be found in", outdir)
