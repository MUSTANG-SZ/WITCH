"""
Data classes for describing models in a structured way.
"""

from dataclasses import dataclass, field
from functools import cached_property
from importlib import import_module
from typing import Optional, Sequence

import dill
import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from minkasi.tods import Tod
from numpy.typing import NDArray
from typing_extensions import Self

from . import core
from . import utils as wu
from .structure import STRUCT_N_PAR


@dataclass
class Parameter:
    """
    Dataclass to represent a single parameter of a model.

    Attributes
    ----------
    name : str
        The name of the parameter.
        This is used only for display purposes.
    fit : list[bool]
        Should be a a list with length `Model.n_rounds`.
        `fit[i]` is True if we want to fit this parameter in the `i`'th round.
    val : float
        The value of the parameter.
    err : float, default: 0
        The error on the parameter value.
    prior : Optional[tuple[float, float]], default: None
        The prior on this parameter.
        Set to None to have to prior,
        otherwise should be the tuple `(lower_bound, upper_bound)`.
    """

    name: str
    fit: list[bool]
    val: float
    err: float = 0
    prior: Optional[tuple[float, float]] = None  # Only flat for now

    def __post_init__(self):
        # If this isn't a float autograd breaks
        self.val = float(self.val)

    @property
    def fit_ever(self) -> bool:
        """
        Check if this parameter is set to ever be fit.

        Returns
        -------
        fit_ever : bool
            True if this parameter is ever fit.
        """
        return bool(np.any(self.fit))


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
    pix_size : float | None
        Pix size of corresponding map
    lims : tuple[float, float, float, float] | None
        List of ra_min, ra_max, dec_min, dec_max for map model was fit to
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
    pix_size: Optional[float] = None
    lims: Optional[tuple[float, float, float, float]] = None
    cur_round: int = 0
    chisq: float = np.inf
    original_order: list[int] = field(init=False)

    def __post_init__(self):
        # Make sure the structure is in the order that core expects
        structure_idx = np.argsort(
            [core.ORDER.index(structure.structure) for structure in self.structures]
        )
        self.structures = [self.structures[i] for i in structure_idx]
        self.original_order = list(np.sort(structure_idx))

    def __setattr__(self, name, value):
        if name == "cur_round":
            self.__dict__.pop("model_grad", None)
            self.__dict__.pop("model", None)
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
                    + str(par.prior) * (par.prior is not None)
                    + " = "
                    + str(par.val)
                    + " ± "
                    + str(par.err)
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

    @property
    def pars(self) -> list[float]:
        """
        Get the current parameter values.

        Returns
        -------
        pars : list[float]
            The parameter values in the order expected by `core.model`.
        """
        pars = []
        for structure in self.structures:
            pars += [parameter.val for parameter in structure.parameters]
        return pars

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
            par_names += [parameter.name for parameter in structure.parameters]
        return par_names

    @property
    def errs(self) -> list[float]:
        """
        Get the current parameter errors.

        Returns
        -------
        errs : list[float]
            The errors in the same order as vals.
        """
        errs = []
        for structure in self.structures:
            errs += [parameter.err for parameter in structure.parameters]
        return errs

    @cached_property
    def priors(self) -> list[Optional[tuple[float, float]]]:
        """
        Get the priors for all parameters.
        Note that this is cached.

        Returns
        -------
        priors : list[Optional[tuple[float, float]]]
            Parameter priors in the same order as `pars`.
        """
        priors = []
        for structure in self.structures:
            priors += [parameter.prior for parameter in structure.parameters]
        return priors

    @property
    def to_fit(self) -> list[bool]:
        """
        Get which parameters we want to fit for the current round.

        Returns
        -------
        to_fit : list[bool]
            `to_fit[i]` is True if we want to fit the `i`'th parameter
            in the current round.
            This is in the same order as `pars`.
        """
        to_fit = []
        for structure in self.structures:
            to_fit += [
                parameter.fit[self.cur_round] for parameter in structure.parameters
            ]
        return to_fit

    @cached_property
    def to_fit_ever(self) -> list[bool]:
        """
        Check which parameters we ever fit.
        Note that this is cached.

        Returns
        -------
        to_fit_ever : list[bool]
            `to_fit[i]` is True if we ever want to fit the `i`'th parameter.
            This is in the same order as `pars`.
        """
        to_fit = []
        for structure in self.structures:
            to_fit += [parameter.fit_ever for parameter in structure.parameters]
        return to_fit

    @cached_property
    def model(self) -> jax.Array:
        """
        The evaluated model, see `core.model` for details.
        Note that this is cached, but is automatically reset whenever
        `update` is called or `cur_round` changes.

        Returns
        -------
        model : jax.Array
            The model evaluted on `xyz` with the current values of `pars`.
        """
        return core.model(
            self.xyz,
            tuple(self.n_struct),
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
        return wu.bilinear_interp(
            dx, dy, self.xyz[0].ravel(), self.xyz[1].ravel(), self.model
        )

    def to_tod_grad(self, dx: ArrayLike, dy: ArrayLike) -> tuple[jax.Array, jax.Array]:
        """
        Project the model and gradint into a TOD.

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

    def update(self, vals: Sequence[float], errs: Sequence[float], chisq: float):
        """
        Update the parameter values and errors as well as the model chi-squared.
        This also resets the cache on `model` and `model_tod`.

        Parameters
        ----------
        vals : Sequence[float]
            The new parameter values.
            Should be in the same order as `pars`.
        errs : Sequence[float]
            The new parameter errors.
            Should be in the same order as `pars`.
        chisq : float
            The new chi-squared.
        """
        if not np.array_equal(self.pars, vals):
            self.__dict__.pop("model", None)
            self.__dict__.pop("model_grad", None)
        n = 0
        for struct in self.structures:
            for par in struct.parameters:
                par.val = vals[n]
                par.err = errs[n]
                n += 1
        self.chisq = chisq

    def minkasi_helper(
        self, params: NDArray[np.floating], tod: Tod
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        Helper function to work with minkasi fitting routines.
        You should never need to touch this yourself, just pass it to
        `minkasi.fitting.fit_timestreams_with_derivs_manyfun`.

        Parameters
        ----------
        params : NDArray[np.floating]
            Array of model parameters.
            Should be in the same order as `pars`
        tod : Tod
            A minkasi tod instance.
            'dx' and 'dy' must be in tod.info and be in radians.

        Returns
        -------
        grad : NDArray[np.floating]
            The gradient of the model with respect to the model parameters.
        pred : NDArray[np.floating]
            The model with the specified substructure.
        """
        self.update(list(params), self.errs, self.chisq)
        dx = tod.info["dx"] * wu.rad_to_arcsec
        dy = tod.info["dy"] * wu.rad_to_arcsec

        pred_tod, grad_tod = self.to_tod_grad(dx, dy)
        pred_tod = jax.device_get(pred_tod)
        grad_tod = jax.device_get(grad_tod)

        return grad_tod, pred_tod

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

        self.__dict__.pop("to_fit_ever")
        self.__dict__.pop("n_struct")
        self.__dict__.pop("priors")
        self.__dict__.pop("par_names")
        self.__dict__.pop("model")
        self.__post_init__()

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
        pix_size: Optional[float] = None,
        lims: Optional[tuple[float, float, float, float]] = None,
    ) -> Self:
        """
        Create an instance of model from a witcher config.

        Parameters
        ----------
        cfg : dict
            The config loaded into a dict.
        pix_size : float | None
            Pix size of corresponding map
        lims : tuple[float, float, float, float] | None
            List of ra_min, ra_max, dec_min, dec_max for map model was fit to

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

        xyz_host = wu.make_grid(
            r_map, dr, dr, dz, x0 * wu.rad_to_arcsec, y0 * wu.rad_to_arcsec
        )
        xyz = jax.device_put(xyz_host, device)
        xyz[0].block_until_ready()
        xyz[1].block_until_ready()
        xyz[2].block_until_ready()

        # Make beam
        beam = wu.beam_double_gauss(
            dr,
            eval(str(cfg["beam"]["fwhm1"])),
            eval(str(cfg["beam"]["amp1"])),
            eval(str(cfg["beam"]["fwhm2"])),
            eval(str(cfg["beam"]["amp2"])),
        )
        beam = jax.device_put(beam, device)

        n_rounds = cfg.get("n_rounds", 1)
        dz = dz * eval(str(cfg["model"]["unit_conversion"]))

        structures = []
        for name, structure in cfg["model"]["structures"].items():
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
                parameters.append(Parameter(par_name, fit, val, 0.0, priors))
            structures.append(Structure(name, structure["structure"], parameters))
        name = cfg["model"].get(
            "name", "-".join([structure.name for structure in structures])
        )

        return cls(name, structures, xyz, dz, beam, n_rounds, pix_size, lims)
