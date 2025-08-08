"""
Dataclass for storing multi model and dataset systems.
"""

from dataclasses import dataclass
from functools import cached_property
from typing import Self

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
from mpi4py import MPI

from .. import core
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
    global_comm: MPI.Intracomm
    models: tuple[tuple[Model, Model], ...]
    datasets: tuple[DataSet, ...]
    parameter_map: tuple[jax.Array, ...]
    model_map: tuple[jax.Array, ...]
    parameters: jax.Array
    errors: jax.Array
    chisq: jax.Array

    @cached_property
    def to_fit(self) -> jax.Array:
        """
        Get which parameters will  be fit this round in the same order `self.parameters`.
        Also checks that all models agree on this.

        Returns
        -------
        to_fit : jax.Array
            Boolean array that is True for parameters that will be fit this round.
        """
        to_fit = jnp.zeros_like(self.parameters, dtype=bool)
        for (model0, _), par_map in zip(self.models, self.parameter_map):
            to_fit = to_fit.at[par_map].set(jnp.array(model0.to_fit))

        # Now lets check
        good = True
        for (model0, model1), par_map in zip(self.models, self.parameter_map):
            good = jnp.array_equal(jnp.array(model0.to_fit), to_fit[par_map])
            good *= jnp.array_equal(jnp.array(model1.to_fit), to_fit[par_map])

        if not good:
            raise ValueError("To fit not consistant across models!")

        return to_fit

    @cached_property
    def to_fit_ever(self) -> jax.Array:
        """
        Get which parameters will ever be fit in the same order `self.parameters`.
        Also checks that all models agree on this.

        Returns
        -------
        to_fit_ever : jax.Array
            Boolean array that is True for parameters that will be fit.
        """
        to_fit_ever = jnp.zeros_like(self.parameters, dtype=bool)
        for (model0, _), par_map in zip(self.models, self.parameter_map):
            to_fit_ever = to_fit_ever.at[par_map].set(jnp.array(model0.to_fit_ever))

        # Now lets check
        good = True
        for (model0, model1), par_map in zip(self.models, self.parameter_map):
            good = jnp.array_equal(jnp.array(model0.to_fit_ever), to_fit_ever[par_map])
            good *= jnp.array_equal(jnp.array(model1.to_fit_ever), to_fit_ever[par_map])

        if not good:
            raise ValueError("To fit not consistant across models!")

        return to_fit_ever

    @cached_property
    def priors(self) -> tuple[jax.Array, jax.Array]:
        """
        Get priors in the same order as `self.parameters`.
        Also checks that all models agree on this.

        Returns
        -------
        priors_low : jax.Array
            Lower bound of prior ranges.
        priors_high : jax.Array
            Higher bound of prior ranges.
        """
        priors_low = jnp.zeros_like(self.parameters)
        priors_high = jnp.zeros_like(self.parameters)
        for (model0, _), par_map in zip(self.models, self.parameter_map):
            priors_low = priors_low.at[par_map].set(model0.priors[0])
            priors_high = priors_low.at[par_map].set(model0.priors[1])

        # Now lets check
        good = True
        for (model0, model1), par_map in zip(self.models, self.parameter_map):
            good = jnp.array_equal(model0.priors[0], priors_low[par_map])
            good *= jnp.array_equal(model0.priors[1], priors_high[par_map])
            good *= jnp.array_equal(model1.priors[0], priors_low[par_map])
            good *= jnp.array_equal(model1.priors[1], priors_high[par_map])

        if not good:
            raise ValueError("Priors not consistant across models!")

        return priors_low, priors_high

    @cached_property
    def cur_round(self) -> int:
        """
        Get the current round of fitting.
        Also checks that all models agree on this.

        Returns
        -------
        cur_round : int
            The current fitting round.
        """
        cur_rounds = jnp.array(
            [(model0.cur_round, model1.cur_round) for model0, model1 in self.models]
        ).ravel()
        cur_round = cur_rounds[0].item()
        if jnp.any(cur_rounds != cur_round):
            raise ValueError("Models don't agree on current round!")
        return cur_round

    @cached_property
    def n_rounds(self) -> int:
        """
        Get the total rounds of fitting.
        Also checks that all models agree on this.

        Returns
        -------
        n_rounds : int
            The total fitting rounds.
        """
        n_rounds_all = jnp.array(
            [(model0.n_rounds, model1.n_rounds) for model0, model1 in self.models]
        ).ravel()
        n_rounds = n_rounds_all[0].item()
        if jnp.any(n_rounds_all != n_rounds):
            raise ValueError("Models don't agree on current round!")
        return n_rounds

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
        m_map = dset.datavec[dataset_ind]
        data = dset.datavec[datavec_ind]
        if dset.mode == "tod":
            x = data.x
            y = data.y
        else:
            x, y = data.xy
        x = x * wu.rad_to_arcsec
        y = y * wu.rad_to_arcsec
        proj = jnp.zeros_like(x)
        models = [model for i, model in enumerate(self.models) if i in m_map]
        for model0, model1 in models:
            ip = model0.model
            ip = core._beam_conv(ip, dset.beam)
            ip = ip.at[:].add(model1.model)
            ip = ip.at[:].multiply(dset.prefactor)
            proj = proj.at[:].add(_project_vectorized(ip, x, y, model0.xyz))
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
        m_map = dset.datavec[dataset_ind]
        data = dset.datavec[datavec_ind]
        if dset.mode == "tod":
            x = data.x
            y = data.y
        else:
            x, y = data.xy
        x = x * wu.rad_to_arcsec
        y = y * wu.rad_to_arcsec
        proj_grad = jnp.zeros(self.parameters.shape + x.shape)
        models = [
            (model, self.parameter_map[i])
            for i, model in enumerate(self.models)
            if i in m_map
        ]
        for (model0, model1), par_map in models:
            ip_grad = model0.model_grad
            ip_grad = core._beam_conv_vec(ip_grad, dset.beam)
            ip_grad = ip_grad.at[:].add(model1.model_grad)
            ip_grad = ip_grad.at[:].multiply(dset.prefactor)
            proj_grad = proj_grad.at[par_map].add(
                _project_vectorized(ip_grad, x, y, model0.xyz)
            )
        return proj_grad

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
            (
                (
                    model0.update(vals[par_map], errs[par_map], chisq),
                    model1.update(vals[par_map], errs[par_map], chisq),
                )
                for (model0, model1), par_map in zip(self.models, self.parameter_map)
            )
        )

        return self

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
            (
                (
                    model0.add_round(to_fit[par_map]),
                    model1.add_round(to_fit[par_map]),
                )
                for (model0, model1), par_map in zip(self.models, self.parameter_map)
            )
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
        for model0, model1 in self.models:
            model0.cur_round = new_round
            model1.cur_round = new_round

        return self

    # Functions for making this a pytree
    # Don't call this on your own
    def tree_flatten(self) -> tuple[tuple, tuple]:
        children = (
            self.models,
            self.datasets,
            self.parameter_map,
            self.model_map,
            self.parameters,
            self.errors,
            self.chisq,
        )
        aux_data = (self.global_comm,)

        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> Self:
        global_comm = aux_data[0]
        models, datasets, parameter_map, model_map, parameters, errors, chisq = children

        return cls(
            global_comm,
            models,
            datasets,
            parameter_map,
            model_map,
            parameters,
            errors,
            chisq,
        )
