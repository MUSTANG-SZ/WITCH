"""
Data classes for describing models in a structured way.
"""

from copy import deepcopy
from dataclasses import dataclass, field
from functools import cached_property
from importlib import import_module
from typing import Self

import dill
import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import register_pytree_node_class
from scipy import interpolate
from typing_extensions import Self

from .. import core
from .. import grid as wg
from .base import Parameter, Structure


def load_xfer(xfer_str) -> jax.Array:
    # Code from Charles
    tab = np.loadtxt(xfer_str).T
    tdim = tab.shape
    # import pdb;pdb.set_trace()
    pfit = np.polyfit(tab[0, tdim[1] // 2 :], tab[1, tdim[1] // 2 :], 1)
    addt = np.max(tab[0, :]) * np.array([2.0, 4.0, 8.0, 16.0, 32.0])
    extt = np.polyval(pfit, addt)
    ### For better backwards compatability I've editted to np.vstack instead of np.stack
    if tdim[0] == 2:
        foo = np.vstack((addt, extt))  # Mar 5, 2018
    else:
        pfit2 = np.polyfit(tab[0, tdim[1] // 2 :], tab[2, tdim[1] // 2 :], 1)
        extt2 = np.polyval(pfit2, addt)
        foo = np.vstack((addt, extt, extt2))  # Mar 5, 2018

    # print(tab.shape, foo.shape)
    tab = np.concatenate((tab, foo), axis=1)
    return jnp.array(tab)


@register_pytree_node_class
@dataclass
class Model:
    """
    Dataclass to describe a model.
    This includes some caching features that improve performance when fitting.
    Note that because of the caching dynamically modifying what structures compose
    the model may not work as intended so beware.

    Attributes
    ----------
    name : str
        The name of the model.
        This is used for display purposes only.
    structures : list[Structure]
        The structures that compose the model.
        Will be sorted to match `core.ORDER` once initialized.
    xyz : tuple[jax.Array, jax.Array, jax.Array, float, float]
        Defines the grid used by model computation.
        The first three elements are a sparse 3D grid in arcseconds,
        with the first two elements being RA and Dec respectively and the third
        element being the LOS. The last two elements are the model center in Ra and Dec
        (also in arcseconds). The structure functions use this as the coordinate to reference
        `dx` and `dy` to.
    dz : float
        The LOS integration factor.
        Should minimally be the pixel size in arcseconds along the LOS,
        but can also include additional factors for performing unit conversions.
    n_rounds : int
        How many rounds of fitting to perform.
    cur_round : int, default: 0
        Which round of fitting we are currently in,
        rounds are 0 indexed.
    to_run : tuple[bool, bool, bool, bool], default: (True, True, True, True)
        The model stages to run.
        See `core.make_to_run` for details.
    chisq : float, default: np.inf
        The chi-squared of this model relative to some data.
        Used when fitting.
    """

    name: str
    structures: tuple[Structure, ...]
    xyz: tuple[jax.Array, jax.Array, jax.Array, float, float]  # arcseconds
    dz: float  # arcseconds * unknown
    n_rounds: int
    cur_round: int = 0
    to_run: tuple[bool, bool, bool, bool] = field(default_factory=core.make_to_run)
    chisq: jax.Array = field(
        default_factory=jnp.array(jnp.inf).copy
    )  # scalar float array

    def __setattr__(self, name, value):
        if name == "cur_round" or name == "xyz":
            self.__dict__.pop("model_grad", None)
            self.__dict__.pop("model", None)
        if name == "n_rounds":
            self.__dict__.pop("to_fit_ever", None)
        return super().__setattr__(name, value)

    def __repr__(self) -> str:
        rep = self.name + ":\n"
        rep += f"Round {self.cur_round + 1} out of {self.n_rounds}\n"
        for struct in self.structures:
            rep += "\t" + struct.name + ":\n"
            for par in struct.parameters:
                rep += (
                    "\t\t"
                    + par.name
                    + "*" * par.fit[self.cur_round]
                    + " ["
                    + str(par.prior[0])
                    + ", "
                    + str(par.prior[1])
                    + "]"
                    + " = "
                    + str(par.val)
                    + " ± "
                    + str(par.err)
                    + " ("
                    + str(jnp.abs(par.val / par.err))
                    + " σ)"
                    + "\n"
                )
        rep += f"chisq is {self.chisq}"
        return rep

    @cached_property
    def n_struct(self) -> list[int]:
        """
        Number of each type of structures in the model.
        Note that this is cached.

        Returns
        -------
        n_struct : list[int]
            `n_struct[i]` is the number of `core.ORDER[i]`
            structures in this model.
        """
        n_struct = [0] * len(core.ORDER)
        for structure in self.structures:
            idx = core.ORDER.index(structure.structure)
            n_struct[idx] += 1
        return n_struct

    @cached_property
    def n_rbins(self) -> list[int]:
        """
        Number of r bins for nonparametric structures.
        Note that this is cached.

        Returns
        -------
        n_rbins : list[int]
            `n_rbins[i]` is the number of rbins in this structure.
        """
        n_rbins = [structure.n_rbins for structure in self.structures]

        return n_rbins

    @property
    def pars(self) -> jax.Array:
        """
        Get the current parameter values.

        Returns
        -------
        pars :  jax.Array
            The parameter values in the order expected by `core.model`.
        """
        pars = jnp.array([])
        for structure in self.structures:
            for parameter in structure.parameters:
                pars = jnp.append(pars, parameter.val.ravel())
        return jnp.array(pars)

    @cached_property
    def par_names(self) -> list[str]:
        """
        Get the names of all parameters.
        Note that this is cached.

        Returns
        -------
        par_names : list[str]
            Parameter names in the same order as `pars`.
        """
        par_names = []
        for structure in self.structures:
            for parameter in structure.parameters:
                if len(parameter.val) > 1:
                    for i in range(len(parameter.val)):
                        par_names += [parameter.name + "_{}".format(i)]
                else:
                    par_names += [parameter.name]

        return par_names

    @cached_property
    def par_struct_names(self) -> list[str]:
        """
        Get the struct names of all parameters.
        Note that this is cached.

        Returns
        -------
        par_struct_names : list[str]
            Parameter struct names in the same order as `pars`.
        """
        par_struct_names = []
        for structure in self.structures:
            for parameter in structure.parameters:
                par_struct_names += [structure.name] * len(parameter.val)

        return par_struct_names

    @cached_property
    def struct_names(self) -> list[str]:
        """
        Get the names of all structures in this model
        """
        return [struct.name for struct in self.structures]

    @property
    def errs(self) -> jax.Array:
        """
        Get the current parameter errors.

        Returns
        -------
        errs : jax.Array
            The errors in the same order as vals.
        """
        errs = jnp.array([])
        for structure in self.structures:
            for parameter in structure.parameters:
                errs = jnp.append(errs, parameter.err.ravel())
        return jnp.array(errs)

    @cached_property
    def priors(self) -> tuple[jax.Array, jax.Array]:
        """
        Get the priors for all parameters.
        Note that this is cached.

        Returns
        -------
        priors : tuple[jax.Array, jax.Array]
            Parameter priors in the same order as `pars`.
            This is a tuple with the first element being an array
            of lower bounds and the second being upper.
        """
        lower = []
        upper = []
        for structure in self.structures:
            for parameter in structure.parameters:
                lower += [parameter.prior[0]] * len(parameter.val)
                upper += [parameter.prior[1]] * len(parameter.val)
        priors = (jnp.array(lower), jnp.array(upper))
        return priors

    @property
    def to_fit(self) -> tuple[bool]:  # jax.Array:
        """
        Get which parameters we want to fit for the current round.

        Returns
        -------
        to_fit : jax.Array
            `to_fit[i]` is True if we want to fit the `i`'th parameter
            in the current round.
            This is in the same order as `pars`.
        """

        to_fit = []
        for structure in self.structures:
            for parameter in structure.parameters:
                to_fit += [parameter.fit[self.cur_round]] * len(parameter.val)
                # to_fit = jnp.append(to_fit, jnp.array([parameter.fit[self.cur_round]] * len(parameter.val)).ravel())

        return tuple(to_fit)  # jnp.ravel(jnp.array(to_fit))

    @cached_property
    def to_fit_ever(self) -> jax.Array:
        """
        Check which parameters we ever fit.
        Note that this is cached.

        Returns
        -------
        to_fit_ever : jax.Array
            `to_fit[i]` is True if we ever want to fit the `i`'th parameter.
            This is in the same order as `pars`.
        """
        to_fit = jnp.array([], dtype=bool)
        for structure in self.structures:
            for parameter in structure.parameters:
                to_fit = jnp.append(
                    to_fit,
                    jnp.array(
                        [parameter.fit_ever] * len(parameter.val), dtype=bool
                    ).ravel(),
                )

        return jnp.ravel(jnp.array(to_fit))

    @cached_property
    def model(self) -> jax.Array:
        """
        The evaluated model, see `core.model` for details.
        Note that this is cached, but is automatically reset whenever
        `update` is called or `cur_round` or `xyz` changes.

        Returns
        -------
        model : jax.Array
            The model evaluted on `xyz` with the current values of `pars`.
        """
        return core.model(
            self.xyz,
            tuple(self.n_struct),
            tuple(self.n_rbins),
            self.dz,
            self.to_run,
            *self.pars,
        )

    @cached_property
    def model_grad(self) -> tuple[jax.Array, jax.Array]:
        """
        The evaluated model and its gradient, see `core.model_grad` for details.
        Note that this is cached, but is automatically reset whenever
        `update` is called or `cur_round` changes.

        Returns
        -------
        model : jax.Array
            The model evaluted on `xyz` with the current values of `pars`.
        grad : jax.Array
            The gradient evaluted on `xyz` with the current values of `pars`.
            Has shape `(len(pars),) + model.shape`.
        """
        argnums = tuple(np.where(self.to_fit)[0] + core.ARGNUM_SHIFT)
        return core.model_grad(
            self.xyz,
            tuple(self.n_struct),
            tuple(self.n_rbins),
            self.dz,
            self.to_run,
            argnums,
            *self.pars,
        )

    def update(self, vals: jax.Array, errs: jax.Array, chisq: jax.Array) -> Self:
        """
        Update the parameter values and errors as well as the model chi-squared.
        This also resets the cache on `model` and `model_grad`
        if `vals` is different than `self.pars`.

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
        updated : Model
            The updated model.
            While nominally the model will update in place, returning it
            alows us to use this function in JITed functions.
        """
        if not np.array_equal(self.pars, vals):
            self.__dict__.pop("model", None)
            self.__dict__.pop("model_grad", None)
        n = 0
        for struct in self.structures:
            for par in struct.parameters:
                for i in range(len(par.val)):
                    par.val = par.val.at[i].set(vals[n])
                    par.err = par.err.at[i].set(errs[n])
                    n += 1
        self.chisq = chisq

        return self

    def add_round(self, to_fit) -> Self:
        """
        Add an additional round to the model.

        Parameters
        ----------
        to_fit : jax.Array
            Boolean array denoting which parameters to fit this round.
            Should be in the same order as `pars`.

        Returns
        -------
        updated : Model
            The updated model with the new round.
            While nominally the model will update in place, returning it
            alows us to use this function in JITed functions.
        """
        self.n_rounds = self.n_rounds + 1
        self.cur_round = self.cur_round + 1
        n = 0
        for struct in self.structures:
            for par in struct.parameters:
                par.fit = par.fit + tuple((to_fit[n].item(),))
                n += 1
        return self

    def remove_struct(self, struct_name: str):
        """
        Remove structure by name.

        Parameters
        ----------
        struct_name : str
            Name of struct to be removed.
        """
        self.structures = tuple(
            structure for structure in self.structures if structure.name != struct_name
        )

        to_pop = ["to_fit_ever", "n_struct", "priors", "par_names"]
        for key in self.__dict__.keys():
            if key in to_pop:  # Pop keys if they are in dict
                self.__dict__.pop(key)

        self.__dict__.pop("model", None)
        self.__dict__.pop("model_grad", None)

    def save(self, path: str):
        """
        Serialize the model to a file with dill.

        Parameters
        ----------
        path : str
            The file to save to.
            Does not check to see if the path is valid.
        """
        with open(path, "wb") as f:
            dill.dump(self, f)

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
        model : Model
            The loaded model.
        """
        with open(path, "rb") as f:
            return dill.load(f)

    @classmethod
    def from_cfg(
        cls,
        cfg: dict,
        model_field: str = "model",
        generate_name: bool = True,
        remove_structs: bool = False,
    ) -> Self:
        """
        Create an instance of model from a witcher config.

        Parameters
        ----------
        cfg : dict
            The config loaded into a dict.
        model_field : str, default: "model"
            The name of the model in the config.
        generate_name : bool, default: True
            If True generate a name for the model.
            If False use `model_field`.
        remove_structs : bool, default: False
            If True then don't include structures marked for removal.

        Returns
        -------
        model : Model
            The model described by the config.
        """
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

        # Load constants
        constants = {
            name: eval(str(const)) for name, const in cfg.get("constants", {}).items()
        }
        constants = constants

        # Get jax device
        dev_id = cfg.get("jax_device", 0)
        device = jax.devices()[dev_id]

        # Setup coordindate stuff
        r_map = eval(str(cfg["coords"]["r_map"]))
        dr = eval(str(cfg["coords"]["dr"]))
        dz = eval(str(cfg["coords"].get("dz", dr)))
        x0 = eval(str(cfg["coords"]["x0"]))
        y0 = eval(str(cfg["coords"]["y0"]))

        xyz_host = wg.make_grid(
            r_map, dr, dr, dz, x0 * wg.rad_to_arcsec, y0 * wg.rad_to_arcsec
        )
        xyz = jax.device_put(xyz_host, device)
        xyz[0].block_until_ready()
        xyz[1].block_until_ready()
        xyz[2].block_until_ready()

        n_rounds = cfg.get("n_rounds", 1)

        structures = []
        for name, structure in cfg[model_field]["structures"].items():
            if structure.get("to_remove", False) and remove_structs:
                continue
            n_rbins = structure.get("n_rbins", 0)
            parameters = []
            for par_name, param in structure["parameters"].items():
                val = eval(str(param["value"]))
                fit = param.get("to_fit", [False] * max(1, n_rounds))
                if isinstance(fit, bool):
                    fit = [fit] * max(1, n_rounds)
                if len(fit) != max(1, n_rounds):
                    raise ValueError(
                        f"to_fit has {len(fit)} entries but we only have {n_rounds} rounds"
                    )
                priors = param.get("priors", None)
                if priors is not None:
                    priors = eval(str(priors))
                else:
                    priors = (-1 * np.inf, np.inf)
                parameters.append(
                    Parameter(
                        par_name,
                        tuple(fit),
                        jnp.atleast_1d(jnp.array(val, dtype=float)),
                        jnp.zeros_like(jnp.atleast_1d(jnp.array(val)), dtype=float),
                        jnp.array(priors, dtype=float),
                    )
                )
            structures.append(
                Structure(
                    name,
                    structure["structure"],
                    parameters,
                    n_rbins=n_rbins,
                )
            )

        name = model_field
        if generate_name:
            name = "-".join([structure.name for structure in structures])

        # Make sure the structure is in the order that core expects
        structure_idx = jnp.argsort(
            jnp.array(
                [core.ORDER_DICT[structure.structure] for structure in structures]
            )
        ).ravel()
        structures = tuple([structures[i] for i in structure_idx])

        return cls(name, structures, xyz, dz, n_rounds)

    # Functions for making this a pytree
    # Don't call this on your own
    def tree_flatten(self) -> tuple[tuple, tuple]:
        children = (self.structures, self.xyz, self.dz, self.chisq)
        aux_data = (
            self.name,
            self.n_rounds,
            self.cur_round,
            self.to_run,
        )

        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> Self:
        name, n_rounds, cur_round, to_run = aux_data
        structures, xyz, dz, chisq = children

        return cls(
            name,
            structures,
            xyz,
            dz,
            n_rounds,
            cur_round,
            to_run,
            chisq,
        )


@dataclass
class Model_xfer(Model):

    ks: jax.Array = field(default_factory=jnp.array([0]).copy)  # scalar float array
    xfer_vals: jax.Array = field(
        default_factory=jnp.array([1]).copy
    )  # scalar float array

    def __post_init__(self):
        pass

    @classmethod
    def from_parent(cls, parent, xfer_str) -> Self:
        xfer = load_xfer(xfer_str)

        pixsize = np.abs(parent.xyz[1][0][1] - parent.xyz[1][0][0])
        ks = xfer[0, 0:] * pixsize  # This picks up an extra dim?
        xfer_vals = xfer[1, 0:]

        my_dict = {}
        for key in parent.__dataclass_fields__.keys():
            if parent.__dataclass_fields__[key].init:
                my_dict[key] = deepcopy(parent.__dict__[key])

        return cls(**my_dict, ks=ks.ravel(), xfer_vals=xfer_vals)

    @cached_property
    def model(self) -> jax.Array:
        cur_map = core.model(
            self.xyz,
            tuple(self.n_struct),
            tuple(self.n_rbins),
            self.dz,
            *self.pars,
        )
        # Code from JMP, whoever that is, by way of Charles
        farr = np.fft.fft2(cur_map)
        nx, ny = cur_map.shape
        kx = np.outer(np.fft.fftfreq(nx), np.zeros(ny).T + 1.0)
        ky = np.outer(np.zeros(nx).T + 1.0, np.fft.fftfreq(ny))
        k = np.sqrt(kx * kx + ky * ky)

        filt = self.table_filter_2d(k)
        farr *= filt

        return np.real(np.fft.ifft2(farr))

    def table_filter_2d(self, k) -> jax.Array:
        f = interpolate.interp1d(self.ks, self.xfer_vals)
        kbin_min = self.ks.min()
        kbin_max = self.ks.max()

        filt = k * 0.0
        filt[(k >= kbin_min) & (k <= kbin_max)] = f(
            k[(k >= kbin_min) & (k <= kbin_max)]
        )
        filt[(k < kbin_min)] = self.xfer_vals[self.ks == kbin_min]
        filt[(k > kbin_max)] = self.xfer_vals[self.ks == kbin_max]

        return filt

    @cached_property
    def model_grad(self) -> None:
        """
        The evaluated model and its gradient, see `core.model_grad` for details.
        Note that this is cached, but is automatically reset whenever
        `update` is called or `cur_round` changes. Currently computing
        grad for models with transfer function is not supported.

        Returns
        -------
        None
        """

        raise TypeError(
            "Error; Grad cannot currently be computed on Models with transfer function"
        )
        return None  # Shouldnt get here
