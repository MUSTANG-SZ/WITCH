import argparse as argp
import os
import sys
import time
from copy import deepcopy
from importlib import import_module

import corner
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mpi4jax
import numpy as np
import yaml
from mpi4py import MPI
from typing_extensions import Any, Unpack

from witch.fitter import *
from witch.fitter import _run_fit
from witch.fitter import _mpi_fsplit

from witch import utils as wu
from witch.containers import Model, Model_xfer
from witch.dataset import DataSet
from witch.fitting import run_mcmc


def load_cfg(cfg):
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

    # TODO: Serialize cfg to a data class (pydantic?)
    cfg["fit"] = cfg.get("fit", "model" in cfg)
    cfg["sim"] = cfg.get("sim", False)
    cfg["wnoise"] = cfg.get("wnoise", False)
    cfg["map"] = cfg.get("map", True)
    cfg["sub"] = cfg.get("sub", True)
    cfg["mem_starved"] = cfg.get("mem_starved", False)

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
        dataset = DataSet(
            dset_name,
            get_files,
            load,
            get_info,
            make_metadata,
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

        # Get the metadata
        metadata = make_metadata(dset_name, cfg, dataset.info)
        dataset.metadata = metadata

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
        if "model" in cfg:
            raise ValueError(
                "No MetaModel defined but Model definition found. This appears to be a legacy configuration! Please update!"
            )
        metamodel = MetaModel(
            comm,
            tuple(),
            datasets,
            tuple(),
            tuple(),
            tuple(),
            jnp.zeros(0),
            jnp.zeros(0),
            jnp.zeros(0),
        )
        print_once("No metamodel defined, setting fit, sim, and sub to False")
        cfg["fit"] = False
        cfg["sim"] = False
        cfg["sub"] = False
    outdir = get_outdir(cfg, metamodel)
    cfg["outdir"] = outdir

    # Now process
    for dataset in metamodel.datasets:
        os.makedirs(os.path.join(outdir, dataset.name), exist_ok=True)
        dataset.info["outdir"] = outdir
        comm.barrier()
        dataset.preproc(dataset, cfg, metamodel)
        comm.barrier()
        if dataset.mode == "tod":
            dataset.datavec = process_tods(cfg, dataset, metamodel)
        elif dataset.mode == "map":
            dataset.datavec = process_maps(cfg, dataset, metamodel)
        comm.barrier()
        dataset = jax.block_until_ready(dataset)
        dataset.postproc(dataset, cfg, metamodel)

    return datasets, outdir, dset_names, metamodel
