"""
Data classes for describing models in a structured way.
"""

from copy import deepcopy
from dataclasses import dataclass, field
from functools import cached_property
from importlib import import_module
from typing import Optional, Self

import dill
import jax
import jax.numpy as jnp
import numpy as np
from astropy.wcs import WCS
from jax.tree_util import register_pytree_node_class
from jax.typing import ArrayLike
from scipy import interpolate
from typing_extensions import Self

from . import core, nonparametric
from . import grid as wg
from . import utils as wu
from .structure import STRUCT_N_PAR


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
class Parameter:
    """
    Dataclass to represent a single parameter of a model.

    Attributes
    ----------
    name : str
        The name of the parameter.
        This is used only for display purposes.
    fit : jax.Array
        Should be array with length `Model.n_rounds`.
        `fit[i]` is True if we want to fit this parameter in the `i`'th round.
    val : float
        The value of the parameter.
    err : float
        The error on the parameter value.
    prior : tuple[float, float]
        The prior on this parameter.
        Should be the tuple `(lower_bound, upper_bound)`.
    """

    name: str
    fit: tuple[bool]  # 1d bool array
    val: jax.Array  # Scalar float array
    err: jax.Array  # Scalar float array
    prior: jax.Array  # 2 element float array

    @property
    def fit_ever(self) -> bool:  # jax.Array:
        """
        Check if this parameter is set to ever be fit.

        Returns
        -------
        fit_ever : jax.Array
            Single element jax boolean array.
            True if this parameter is ever fit.
        """
        return np.any(self.fit).item()

    # Functions for making this a pytree
    # Don't call this on your own
    def tree_flatten(self) -> tuple[tuple, tuple]:
        children = (self.val, self.err, self.prior)
        aux_data = (self.name, self.fit)

        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> Self:
        name, fit = aux_data
        return cls(name, fit, *children)


@register_pytree_node_class
@dataclass
class Structure:
    """
    Dataclass to represent a structure within the model.

    Attributes
    ----------
    name : str
        The name of the structure.
        This is used only for display purposes.
    structure : str
        The type of structure that this is an instance of.
        Should be a string that appears in `core.ORDER`
    parameters : list[Parameter]
        The model parameters for this structure.

    Raises
    ------
    ValueError
        If `structure` is not a valid structure.
        If we have the wrong number of parameters.
    """

    name: str
    structure: str
    parameters: list[Parameter]
    n_rbins: int = 0

    def __post_init__(self):
        self.structure = self.structure.lower()
        # Check that this is a valid structure
        if self.structure not in STRUCT_N_PAR.keys():
            raise ValueError(f"{self.name} has invalid structure: {self.structure}")
        # Check that we have the correct number of params
        if len(self.parameters) != STRUCT_N_PAR[self.structure]:
            raise ValueError(
                f"{self.name} has incorrect number of parameters, expected {STRUCT_N_PAR[self.structure]} for {self.structure} but was given {len(self.parameters)}"
            )

    # Functions for making this a pytree
    # Don't call this on your own
    def tree_flatten(self) -> tuple[tuple, tuple]:
        children = tuple(self.parameters)
        aux_data = (self.name, self.structure, self.n_rbins)

        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> Self:
        name, structure, n_rbins = aux_data
        parameters = children

        return cls(name, structure, list(parameters), n_rbins)


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
    beam : jax.Array
        The beam to convolve the model with.
    n_rounds : int
        How many rounds of fitting to perform.
    cur_round : int, default: 0
        Which round of fitting we are currently in,
        rounds are 0 indexed.
    chisq : float, default: np.inf
        The chi-squared of this model relative to some data.
        Used when fitting.
    original_order : list[int]
        The original order than the structures in `structures` were inputted.
    """

    name: str
    structures: list[Structure]
    xyz: tuple[jax.Array, jax.Array, jax.Array, float, float]  # arcseconds
    dz: float  # arcseconds * unknown
    beam: jax.Array
    n_rounds: int
    cur_round: int = 0
    chisq: jax.Array = field(
        default_factory=jnp.array(jnp.inf).copy
    )  # scalar float array
    original_order: list[int] = field(init=False)

    def __post_init__(self):
        # Make sure the structure is in the order that core expects
        structure_idx = np.argsort(
            np.array(
                [core.ORDER.index(structure.structure) for structure in self.structures]
            )
        )
        self.structures = [self.structures[i] for i in structure_idx]
        self.original_order = list(jnp.sort(structure_idx))

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
        for i in self.original_order:
            struct = self.structures[i]
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
            self.beam,
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
            self.beam,
            argnums,
            *self.pars,
        )

    def to_tod(self, dx: ArrayLike, dy: ArrayLike) -> jax.Array:
        """
        Project the model into a TOD.

        Parameters
        ----------
        dx : ArrayLike
            The RA TOD in arcseconds.
        dy : ArrayLike
            The Dec TOD in arcseconds.

        Returns
        -------
        tod : jax.Array
            The model as a TOD.
            Same shape as dx.
        """
        tod = wu.bilinear_interp(
            dx, dy, self.xyz[0].ravel(), self.xyz[1].ravel(), self.model
        )
        tod = tod - jnp.mean(tod, axis=-1)[..., None]
        return tod

    def to_tod_grad(self, dx: ArrayLike, dy: ArrayLike) -> tuple[jax.Array, jax.Array]:
        """
        Project the model and gradient into a TOD.

        Parameters
        ----------
        dx : ArrayLike
            The RA TOD in arcseconds.
        dy : ArrayLike
            The Dec TOD in arcseconds.

        Returns
        -------
        tod : jax.Array
            The model as a TOD.
            Same shape as dx.
        grad_tod : jax.Array
            The gradient as a TOD.
            Has shape `(len(pars),) + dx.shape`.
        """
        model, grad = self.model_grad
        tod = wu.bilinear_interp(
            dx, dy, self.xyz[0].ravel(), self.xyz[1].ravel(), model
        )
        tod = tod - jnp.mean(tod, axis=-1)[..., None]
        grad_tod = jnp.array(
            [
                (
                    wu.bilinear_interp(
                        dx, dy, self.xyz[0].ravel(), self.xyz[1].ravel(), _grad
                    )
                    if _fit
                    else jnp.zeros_like(tod)
                )
                for _grad, _fit in zip(grad, self.to_fit)
            ]
        )

        return tod, grad_tod

    def to_map(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """
        Project the model into a map.

        Parameters
        ----------
        x : jax.Array
            The RA of the map at each pixel.
        y : jax.Array
            Th Dec of the map at each pixel.

        Returns
        -------
        omap : jax.Array
            The model projected onto a map.
            Same shape as x.
        """
        omap = wu.bilinear_interp(
            x.ravel(), y.ravel(), self.xyz[0].ravel(), self.xyz[1].ravel(), self.model
        ).reshape(x.shape)
        omap = omap - jnp.mean(omap)
        return omap

    def to_map_grad(self, x: jax.Array, y: jax.Array) -> tuple[jax.Array, jax.Array]:
        """
        Project the model into a map.

        Parameters
        ----------
        x : jax.Array
            The RA of the map at each pixel.
        y : jax.Array
            Th Dec of the map at each pixel.

        Returns
        -------
        omap : jax.Array
            The model projected onto a map.
            Same shape as x.
        grad_map : jax.Array
            The gradient of each parameter projected onto a map.
            Has shape `(npar,) + x.shape`.
        """
        model, grad = self.model_grad
        omap = wu.bilinear_interp(
            x.ravel(), y.ravel(), self.xyz[0].ravel(), self.xyz[1].ravel(), model
        ).reshape(x.shape)
        omap = omap - jnp.mean(omap)
        grad_map = jnp.array(
            [
                (
                    wu.bilinear_interp(
                        x.ravel(),
                        y.ravel(),
                        self.xyz[0].ravel(),
                        self.xyz[1].ravel(),
                        _grad,
                    ).reshape(x.shape)
                    if _fit
                    else jnp.zeros_like(omap)
                )
                for _grad, _fit in zip(grad, self.to_fit)
            ]
        )
        return omap, grad_map

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
        n = None
        for i, structure in enumerate(self.structures):
            if str(structure.name) == str(struct_name):
                n = i
        if type(n) == int:
            self.structures.pop(n)
        else:
            raise ValueError("Error: {} not in structure names".format(struct_name))

        to_pop = ["to_fit_ever", "n_struct", "priors", "par_names"]
        for key in self.__dict__.keys():
            if key in to_pop:  # Pop keys if they are in dict
                self.__dict__.pop(key)

        self.__dict__.pop("model", None)
        self.__dict__.pop("model_grad", None)
        self.__post_init__()

    def para_to_non_para(
        self,
        n_rounds: Optional[int] = None,
        to_copy: list[str] = ["gnfw", "gnfw_rs", "a10", "isobeta", "uniform"],
    ) -> Self:
        """
        Function which approximately converts cluster profiles into a non-parametric form. Note this is
        only approximate and should be fit afterwords.

        Parameters
        ----------
        n_rounds: Optional int | None, default: None
            Number of rounds to fit for output model. If none, copy from self
        to_copy : list[str], default: gnfw, gnfw_rs, a10, isobeta, uniform
            List of structures, by name, to copy.
        Returns
        -------
        Model : Model
            Model with a non-parametric representation of input model
        Raises
        ------
        ValueError
            If there are no models to copy
        """
        cur_model = deepcopy(
            self
        )  # Make a copy of model, we don't want to lose structures
        i = 0  # Make sure we keep at least one struct
        for structure in cur_model.structures:
            if structure.structure not in to_copy:
                cur_model.remove_struct(structure.name)
            else:
                i += 1
        if i == 0:
            raise ValueError("Error: no model structures in {}".format(to_copy))
        params = jnp.array(cur_model.pars)
        params = jnp.ravel(params)
        pressure, _ = core.model3D(
            cur_model.xyz, tuple(cur_model.n_struct), tuple(cur_model.n_rbins), params
        )
        pressure = pressure[
            ..., int(pressure.shape[2] / 2)
        ]  # Take middle slice. Close enough is good enough here, dont care about rounding

        pixsize = np.abs(cur_model.xyz[1][0][1] - cur_model.xyz[1][0][0])

        rs, bin1d, var1d = wu.bin_map(pressure, pixsize)

        rbins = nonparametric.get_rbins(cur_model)
        rbins = np.append(rbins, np.array([np.amax(rs)]))

        condlist = [
            np.array((rbins[i] <= rs) & (rs < rbins[i + 1]))
            for i in range(len(rbins) - 2, -1, -1)
        ]

        amps, pows, c = nonparametric.profile_to_broken_power(
            rs, bin1d, condlist, rbins
        )

        priors = (-1 * np.inf, np.inf)
        if n_rounds is None:
            n_rounds = self.n_rounds
        parameters = [
            Parameter(
                "rbins",
                tuple([False] * n_rounds),
                jnp.atleast_1d(jnp.array(rbins[:-1], dtype=float)),  # Drop last bin
                jnp.zeros_like(jnp.atleast_1d(jnp.array(rbins[:-1])), dtype=float),
                jnp.array(priors, dtype=float),
            ),
            Parameter(
                "amps",
                tuple([True] * n_rounds),
                jnp.atleast_1d(jnp.array(amps, dtype=float)),
                jnp.zeros_like(jnp.atleast_1d(jnp.array(amps)), dtype=float),
                jnp.array(priors, dtype=float),
            ),
            Parameter(
                "pows",
                tuple([True] * n_rounds),
                jnp.atleast_1d(jnp.array(pows, dtype=float)),
                jnp.zeros_like(jnp.atleast_1d(jnp.array(pows)), dtype=float),
                jnp.array(priors, dtype=float),
            ),
            Parameter(
                "dx",  # TODO: miscentering
                tuple([False] * n_rounds),
                jnp.atleast_1d(jnp.array(0, dtype=float)),
                jnp.zeros_like(jnp.atleast_1d(jnp.array(0)), dtype=float),
                jnp.array(priors, dtype=float),
            ),
            Parameter(
                "dy",  # TODO: miscentering
                tuple([False] * n_rounds),
                jnp.atleast_1d(jnp.array(0, dtype=float)),
                jnp.zeros_like(jnp.atleast_1d(jnp.array(0)), dtype=float),
                jnp.array(priors, dtype=float),
            ),
            Parameter(
                "dz",  # TODO: miscentering
                tuple([False] * n_rounds),
                jnp.atleast_1d(jnp.array(0, dtype=float)),
                jnp.zeros_like(jnp.atleast_1d(jnp.array(0)), dtype=float),
                jnp.array(priors, dtype=float),
            ),
            Parameter(
                "c",
                tuple([True] * n_rounds),
                jnp.atleast_1d(jnp.array(c, dtype=float)),
                jnp.zeros_like(jnp.atleast_1d(jnp.array(0)), dtype=float),
                jnp.array(priors, dtype=float),
            ),
        ]

        structures = [
            Structure(
                "nonpara_power", "nonpara_power", parameters, n_rbins=len(rbins) - 1
            )
        ]
        for structure in self.structures:
            if structure.name not in to_copy:
                structures.append(structure)

        return Model(
            name="test",
            structures=structures,
            xyz=self.xyz,
            dz=self.dz,
            beam=self.beam,
            n_rounds=n_rounds,
            cur_round=0,
        )

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
    def from_cfg(cls, cfg: dict, beam: Optional[jax.Array] = None) -> Self:
        """
        Create an instance of model from a witcher config.

        Parameters
        ----------
        cfg : dict
            The config loaded into a dict.
        beam : Optional[Array], default: None

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
        }  # pyright: ignore [reportUnusedVariable]

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

        # Make beam
        if beam is None:
            beam = jnp.ones((1, 1))
        beam = jax.device_put(beam, device)
        if beam is None:
            raise ValueError("Beam somehow still None!")

        n_rounds = cfg.get("n_rounds", 1)
        dz = dz * eval(str(cfg["model"]["unit_conversion"]))

        structures = []
        for name, structure in cfg["model"]["structures"].items():
            n_rbins = structure.get("n_rbins", 0)
            parameters = []
            for par_name, param in structure["parameters"].items():
                val = eval(str(param["value"]))
                fit = param.get("to_fit", [False] * n_rounds)
                if isinstance(fit, bool):
                    fit = [fit] * n_rounds
                if len(fit) != n_rounds:
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
                Structure(name, structure["structure"], parameters, n_rbins=n_rbins)
            )
        name = cfg["model"].get(
            "name", "-".join([structure.name for structure in structures])
        )

        return cls(name, structures, xyz, dz, beam, n_rounds)

    # Functions for making this a pytree
    # Don't call this on your own
    def tree_flatten(self) -> tuple[tuple, tuple]:
        children = (tuple(self.structures), self.xyz, self.dz, self.beam, self.chisq)
        aux_data = (
            self.name,
            self.n_rounds,
            self.cur_round,
        )

        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> Self:
        name, n_rounds, cur_round = aux_data
        structures, xyz, dz, beam, chisq = children

        return cls(name, list(structures), xyz, dz, beam, n_rounds, cur_round, chisq)


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
            self.beam,
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

    def to_tod_grad(self, dx: ArrayLike, dy: ArrayLike) -> None:
        """
        Project the model and gradient into a TOD. Currently computing
        grad for models with transfer function is not supported.

        Returns
        -------
        None
        """
        raise TypeError(
            "Error; Grad cannot currently be computed on Models with transfer function"
        )
        return None  # Shouldnt get here
