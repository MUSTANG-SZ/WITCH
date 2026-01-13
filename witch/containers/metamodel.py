"""
Dataclass for storing multi model and dataset systems.
"""

from copy import copy, deepcopy
from dataclasses import dataclass
from functools import cached_property
from typing import Self

import dill
import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import register_pytree_node_class
from mpi4py import MPI

from .. import utils as wu
from ..dataset import DataSet
from .model import Model


def _project(ip, x, y, xyz):
    return wu.bilinear_interp(
        x.ravel(), y.ravel(), xyz[0].ravel(), xyz[1].ravel(), ip
    ).reshape(x.shape)


_project_vectorized = jax.vmap(_project, in_axes=(0, None, None, None))


@register_pytree_node_class
@dataclass
class MetaModel:
    """
    Class that manages multiple models and datasets at once.
    This enables computing a single model for multiple datasets
    as well as multiple models with shared parameters.

    Attributes
    ----------
    global_comm : MPI.Comm | MPI.Intracomm
        The MPI communicator used for all datasets.
    models : tuple[Model, ...]
        Tuple of models to fit together.
    datasets : tuple[DataSet]
        Tuple of datasets to fit together.
    parameter_map : tuple[tuple[int, ...], ...]
        Structure to map the parameters held in `MetaModel`
        to each individual `Model` instance.
        Thie `i`th entry is a tuple that indexes `MetaModel.parameters`
        to get the parameters for the `i`th `Model` in `MetaModel.models`
    model_map : tuple[tuple[int, ...], ...]
        Structure to map models to datasets.
        The `j`th entry is an `n_model` length tuple in which the
        `i`th entry lists the indices of `MetaModel.models` used by
        `MetaModel.datasets[j]`.
    metadata_map : tuple[tuple[tuple[int, ...], ...], ...]
        Structure to map what metadata to apply to which model.
        The `j`th entry is an `n_model` length tuple in which the
        `i`th entry lists the indices of `MetaModel.datasets[j].metadata`
        to apply to `MetaModel.models[i]`.
    parameters : jax.Array
        The parameters of this `MetaModel`.
    errors : jax.Array
        The error currently associated with each element of `MetaModel.parameters`.
    chisq : jax.Array
        The log-likelihood of the current state of the `MetaModel`.
    """

    global_comm: MPI.Comm | MPI.Intracomm | wu.NullComm
    models: tuple[Model, ...]
    datasets: tuple[DataSet, ...]
    parameter_map: tuple[tuple[int, ...], ...]
    model_map: tuple[tuple[int, ...], ...]
    metadata_map: tuple[tuple[tuple[int, ...], ...], ...]
    parameters: jax.Array
    errors: jax.Array
    chisq: jax.Array

    def __repr__(self) -> str:
        reprs = [str(model) for model in self.models]
        rep = []
        idx = 0
        for r in reprs:
            message = r.split("\n")
            rep += message[idx:-1]
        rep = "\n".join(rep)
        rep += f"\nchisq is {self.chisq}"
        return rep

    @cached_property
    def par_names(self) -> tuple[str]:
        """
        Get the parameter names in the same order as `self.parameters`.

        Returns
        -------
        par_name : jax.Array
            String array of parameter names.
        """
        par_names = np.zeros_like(self.parameters, dtype="U128")
        for model, par_map in zip(self.models, self.parameter_map):
            par_names[list(par_map)] = np.array(model.par_names, dtype="U128")

        return tuple(par_names.tolist())

    @property
    def to_fit(self) -> jax.Array:
        """
        Get which parameters will  be fit this round in the same order `self.parameters`.

        Returns
        -------
        to_fit : jax.Array
            Boolean array that is True for parameters that will be fit this round.
        """
        to_fit = jnp.zeros_like(self.parameters, dtype=bool)
        for model, par_map in zip(self.models, self.parameter_map):
            to_fit = to_fit.at[jnp.array(par_map)].set(jnp.array(model.to_fit))

        return to_fit

    @cached_property
    def to_fit_ever(self) -> jax.Array:
        """
        Get which parameters will ever be fit in the same order `self.parameters`.

        Returns
        -------
        to_fit_ever : jax.Array
            Boolean array that is True for parameters that will be fit.
        """
        to_fit_ever = jnp.zeros_like(self.parameters, dtype=bool)
        for model, par_map in zip(self.models, self.parameter_map):
            to_fit_ever = to_fit_ever.at[jnp.array(par_map)].set(
                jnp.array(model.to_fit_ever)
            )

        return to_fit_ever

    @cached_property
    def priors(self) -> tuple[jax.Array, jax.Array]:
        """
        Get priors in the same order as `self.parameters`.

        Returns
        -------
        priors_low : jax.Array
            Lower bound of prior ranges.
        priors_high : jax.Array
            Higher bound of prior ranges.
        """
        priors_low = jnp.zeros_like(self.parameters)
        priors_high = jnp.zeros_like(self.parameters)
        for model, par_map in zip(self.models, self.parameter_map):
            priors_low = priors_low.at[jnp.array(par_map)].set(model.priors[0])
            priors_high = priors_high.at[jnp.array(par_map)].set(model.priors[1])

        return priors_low, priors_high

    @cached_property
    def cur_round(self) -> jax.Array:
        """
        Get the current round of fitting.

        Returns
        -------
        cur_round : int
            The current fitting round.
        """
        cur_rounds = jnp.array([model.cur_round for model in self.models]).ravel()
        cur_round = cur_rounds[0]

        return cur_round

    @cached_property
    def n_rounds(self) -> jax.Array:
        """
        Get the total rounds of fitting.

        Returns
        -------
        n_rounds : int
            The total fitting rounds.
        """
        n_rounds_all = jnp.array([model.n_rounds for model in self.models]).ravel()
        n_rounds = n_rounds_all[0]

        return n_rounds

    def model_grid(self, dataset_ind: int) -> jax.Array:
        """
        Get the model for a dataset on the computed grid.
        This currently assumes that all models have the same grid.
        This will not apply any metadata (ie. beam convolution).

        Parameters
        ----------
        dataset_ind : int
            The index of the dataset in `self.datasets` to use.

        Returns
        -------
        model_grid : jax.Array
            The model on the computed grid.
        """
        m_map = self.model_map[dataset_ind]
        proj = jnp.zeros_like(self.models[m_map[0]].model)
        for i in m_map:
            model = self.models[i]
            ip = model.model
            proj = proj.at[:].add(ip)
        return proj

    def model_proj(self, dataset_ind: int, datavec_ind: int) -> jax.Array:
        """
        Project the models held in the metamodel to some data in a dataset.

        Parameters
        ----------
        dataset_ind : int
            The index of the dataset in `self.datasets` to use.
        datavec_ind : int
            The index of the data in `self.datasets[dataset_ind].datavec` to use.

        Returns
        -------
        model_proj : jax.Array
            The metamodel projected into an array that matches the shape of
            `self.datasets[dataset_ind].datavec[datavec_ind]`.
        """
        dset = self.datasets[dataset_ind]
        m_map = self.model_map[dataset_ind]
        md_map = self.metadata_map[dataset_ind]
        data = dset.datavec[datavec_ind]
        if dset.mode == "tod":
            x = data.x
            y = data.y
        else:
            x, y = data.xy
        x = x * wu.rad_to_arcsec
        y = y * wu.rad_to_arcsec
        proj = jnp.zeros_like(x)
        for i in m_map:
            model = self.models[i]
            md = md_map[i]
            ip = model.model
            for md_idx in md:
                ip = dset.metadata[md_idx].apply(ip)
            _proj = _project(ip, x, y, model.xyz)
            for md_idx in md:
                _proj = dset.metadata[md_idx].apply_proj(_proj)
            proj = proj.at[:].add(_proj)
        return proj

    def model_grad_proj(self, dataset_ind: int, datavec_ind: int) -> jax.Array:
        """
        Project the models held in the metamodel to some data in a dataset.

        Parameters
        ----------
        dataset_ind : int
            The index of the dataset in `self.datasets` to use.
        datavec_ind : int
            The index of the data in `self.datasets[dataset_ind].datavec` to use.

        Returns
        -------
        model_grad_proj : jax.Array
            The metamodel gradients projected into an array with `len(self.parameters)` elements
            where each element matches the shape of `self.datasets[dataset_ind].datavec[datavec_ind]`.
        """
        dset = self.datasets[dataset_ind]
        m_map = self.model_map[dataset_ind]
        md_map = self.metadata_map[dataset_ind]
        data = dset.datavec[datavec_ind]
        if dset.mode == "tod":
            x = data.x
            y = data.y
        else:
            x, y = data.xy
        x = x * wu.rad_to_arcsec
        y = y * wu.rad_to_arcsec
        proj_grad = jnp.zeros(self.parameters.shape + x.shape)
        for i in m_map:
            model = self.models[i]
            md = md_map[i]
            par_map = self.parameter_map[i]
            ip_grad = model.model_grad[1]
            for md_idx in md:
                ip_grad = dset.metadata[md_idx].apply_grad(ip_grad)
            _proj_grad = _project_vectorized(ip_grad, x, y, model.xyz)
            for md_idx in md:
                _proj_grad = dset.metadata[md_idx].apply_grad_proj(_proj_grad)
            proj_grad = proj_grad.at[jnp.array(par_map)].add(_proj_grad)
        return proj_grad

    def get_dataset_ind(self, dset_name: str) -> int:
        """
        Get the index of a dataset

        Parameters
        ----------
        dset_name : str
            The name of the dataset to find

        Returns
        -------
        dataset_ind : int
            The index of the dataset.
        """
        dataset_ind = -1
        for i, dset in enumerate(self.datasets):
            if dset_name == dset.name:
                dataset_ind = i
                break
        return dataset_ind

    def update(self, vals: jax.Array, errs: jax.Array, chisq: jax.Array) -> Self:
        """
        Update the parameter values and errors as well as the model chi-squared
        for all models in the metamodel.

        Parameters
        ----------
        vals : jax.Array
            The new parameter values.
            Should be in the same order as `pars`.
        errs : jax.Array
            The new parameter errors.
            Should be in the same order as `pars`.
        chisq : jax.Array
            The new chi-squared.
            Should be a scalar float array.

        Returns
        -------
        updated : MetaModel
            The updated metamodel.
            While nominally the metamodel will update in place, returning it
            alows us to use this function in JITed functions.
        """
        self.parameters = vals
        self.errors = errs
        self.chisq = chisq
        self.models = tuple(
            deepcopy(model).update(
                vals[jnp.array(par_map)], errs[jnp.array(par_map)], chisq
            )
            for model, par_map in zip(self.models, self.parameter_map)
        )

        return copy(self)

    def add_round(self, to_fit: jax.Array) -> Self:
        """
        Add an additional round to the metamodel.

        Parameters
        ----------
        to_fit : jax.Array
            Boolean array denoting which parameters to fit this round.
            Should be in the same order as `self.parameters`.

        Returns
        -------
        updated : MetaModel
            The updated metamodel with the new round.
            While nominally the model will update in place, returning it
            alows us to use this function in JITed functions.
        """
        self.__dict__.pop("n_rounds", None)
        self.__dict__.pop("to_fit", None)
        self.__dict__.pop("to_fit_ever", None)
        self.models = tuple(
            model.add_round(to_fit[jnp.array(par_map)])
            for model, par_map in zip(self.models, self.parameter_map)
        )

        return self

    def set_round(self, new_round: int) -> Self:
        """
        Set the round of the metamodel.

        Parameters
        ----------
        new_round : int
            The number of the round to go to.

        Returns
        -------
        updated : MetaModel
            The updated metamodel with the round updated.
            While nominally the model will update in place, returning it
            alows us to use this function in JITed functions.
        """
        if new_round > self.n_rounds or new_round < 0:
            raise ValueError("Trying to set a round that doesn't exist!")
        self.__dict__.pop("cur_round", None)
        self.__dict__.pop("to_fit", None)
        for model in self.models:
            model.cur_round = new_round

        return self

    def save(self, path: str):
        """
        Serialize the model to a file with dill.

        Parameters
        ----------
        path : str
            The file to save to.
            Does not check to see if the path is valid.
        """
        datavecs = []
        comms = []
        gcomm = self.global_comm
        for dataset in self.datasets:
            datavecs += [dataset.datavec]
            comms += [dataset.global_comm]
            dataset.datavec = None
            dataset.global_comm = wu.NullComm()
        self.global_comm = wu.NullComm()
        with open(path, "wb") as f:
            dill.dump(self, f)
        for dataset, datavec, comm in zip(self.datasets, datavecs, comms):
            dataset.datavec = datavec
            dataset.global_comm = comm
        self.global_comm = gcomm

    @classmethod
    def load(cls, path: str) -> Self:
        """
        Load the model from a file with dill.

        Parameters
        ----------
        path : str
            The path to the saved model.
            Does not check to see if the path is valid.

        Returns
        -------
        model : MetaModel
            The loaded model.
        """
        with open(path, "rb") as f:
            return dill.load(f)

    def remove_structs(self, cfg):
        """
        Create a new metamodel with marked structures removed.

        Parameters
        ----------
        cfg : dict
            The config loaded into a dict.

        Returns
        -------
        metamodel : MetaModel
            The metamodel described with structures removed.
        """
        new_metamodel = self.__class__.from_config(
            self.global_comm, cfg, self.datasets, True
        )
        new_metamodel = new_metamodel.update(self.parameters, self.errors, self.chisq)
        return new_metamodel

    @classmethod
    def from_config(
        cls,
        global_comm: MPI.Comm | MPI.Intracomm | wu.NullComm,
        cfg: dict,
        datasets: tuple[DataSet, ...],
        remove_structs: bool = False,
    ) -> Self:
        """
        Create an instance of metamodel from a witcher config.

        Parameters
        ----------
        global_comm: MPI.Comm | MPI.Intracomm | wu.NullComm,
            The communicator for this metamodel.
        cfg : dict
            The config loaded into a dict.
        datasets : tuple[DataSet]
            The datasets to associate with this model
        remove_structs : bool, default: False
            If True don't include structures marked for removal.

        Returns
        -------
        metamodel : MetaModel
            The metamodel described by the config.
        """
        metacfg = cfg.get("metamodel", {})

        # First lets load all the models
        mlist = metacfg.get("models", ["model"])
        models = tuple(
            Model.from_cfg(cfg, model_field, False, remove_structs)
            for model_field in mlist
        )
        model_ind = {model_field: i for i, model_field in enumerate(mlist)}

        # Now lets make the model map
        mmap = metacfg.get("model_map", {dset.name: mlist for dset in datasets})
        model_map = tuple(
            tuple(
                sorted(tuple(model_ind[model_field] for model_field in mmap[dset.name]))
            )
            for dset in datasets
        )

        par_map, pars, errs = _compute_par_map_and_pars(metacfg, models)

        metadata_map = _compute_metadata_map(models, datasets)

        return cls(
            global_comm,
            models,
            datasets,
            par_map,
            model_map,
            metadata_map,
            pars,
            errs,
            models[0].chisq,
        )

    # Functons for making this a pytree
    # Don't call this on your own
    def tree_flatten(self) -> tuple[tuple, tuple]:
        children = (
            self.models,
            self.datasets,
            self.parameters,
            self.errors,
            self.chisq,
        )
        aux_data = (
            self.global_comm,
            self.model_map,
            self.parameter_map,
            self.metadata_map,
        )

        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> Self:
        global_comm, model_map, parameter_map, metadata_map = aux_data
        models, datasets, parameters, errors, chisq = children

        return cls(
            global_comm,
            models,
            datasets,
            parameter_map,
            model_map,
            metadata_map,
            parameters,
            errors,
            chisq,
        )


def _compute_par_map_and_pars(
    metacfg,
    models,
):
    # Figure out parameter map and use to get initial parameters
    # Non-parametric stuff is treated as seperate and flat here (ie. name_0, name_1, etc.)
    pmap = metacfg.get("parameter_map", [])
    # Get the names of all things
    full_par_names = []
    model_par_names = {}
    for model in models:
        par_names = model.par_names
        par_struct_names = model.par_struct_names
        par_model_names = [model.name] * len(par_names)
        model_par_names[model.name] = [
            f"{m}.{s}.{p}"
            for m, s, p in zip(par_model_names, par_struct_names, par_names)
        ]
        full_par_names += model_par_names[model.name]
    # Mark repeats and figure out indexing
    repeats = np.zeros(len(full_par_names))
    par_idx = jnp.arange(len(full_par_names))
    for p in pmap:
        for n in p[1:]:
            repeats[full_par_names.index(n)] += 1
            par_idx = par_idx.at[full_par_names.index(n) :].add(-1)
            par_idx = par_idx.at[full_par_names.index(n)].set(
                full_par_names.index(p[0])
            )
    if np.any(repeats > 1):
        raise ValueError("Some parameters mapped multiple times!")
    repeats = repeats.astype(bool)
    # Now build
    n_pars = int(np.sum(~repeats))
    pars = jnp.zeros(n_pars)
    errs = jnp.zeros(n_pars)
    n = 0
    par_map = []
    for model in models:
        npar = len(model.par_names)
        _par_map = jnp.array(par_idx.at[n : n + npar].get())
        pars = pars.at[_par_map].set(model.pars)
        errs = errs.at[_par_map].set(model.errs)
        par_map += [tuple(_par_map)]
        n += npar
    par_map = tuple(par_map)

    return par_map, pars, errs


def _compute_metadata_map(models, datasets):
    metadata_map = tuple(
        tuple(
            tuple(
                i
                for i, metadata in enumerate(dataset.metadata)
                if metadata.check_apply(model.name)
            )
            for model in models
        )
        for dataset in datasets
    )
    return tuple(metadata_map)
