"""
Data classes for describing models in a structured way
"""

from dataclasses import dataclass, field
from typing import Optional

import dill
import jax
import numpy as np

# For eval statements
# TODO: Make this dynamic?
from astropy import units as u  # type: ignore [reportUnusedImport]
from astropy.coordinates import Angle  # pyright: ignore [reportUnusedImport]
from jax.typing import ArrayLike
from minkasi.tods import Tod
from numpy.typing import NDArray
from typing_extensions import Self

from . import core
from . import utils as wu
from .structure import STRUCT_N_PAR


@dataclass
class Parameter:
    name: str
    fit: list[bool]
    val: float
    err: float = 0
    prior: Optional[tuple[float, float]] = None  # Only flat for now

    @property
    def fit_ever(self) -> bool:
        return bool(np.any(self.fit))


@dataclass
class Structure:
    name: str
    structure: str
    parameters: list[Parameter]

    def __post_init__(self):
        self.structure = self.structure.lower()
        # Check that this is a valid structure
        if self.structure not in STRUCT_N_PAR.keys():
            raise ValueError("%s has invalid structure: %s", self.name, self.structure)
        # Check that we have the correct number of params
        if len(self.parameters) != STRUCT_N_PAR[self.structure]:
            raise ValueError(
                "%s has incorrect number of parameters, expected %d for %s but was given %d",
                self.name,
                STRUCT_N_PAR[self.structure],
                self.structure,
                len(self.parameters),
            )


@dataclass
class Model:
    name: str
    structures: list[Structure]
    xyz: tuple[jax.Array, jax.Array, jax.Array, float, float]  # arcseconds
    dz: float  # arcseconds * unknown
    beam: jax.Array
    n_rounds: int
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

    @property
    def n_struct(self) -> list[int]:
        n_struct = [0] * len(core.ORDER)
        for structure in self.structures:
            idx = core.ORDER.index(structure.structure)
            n_struct[idx] += 1
        return n_struct

    @property
    def pars(self) -> list[float]:
        pars = []
        for structure in self.structures:
            pars += [parameter.val for parameter in structure.parameters]
        return pars

    @property
    def par_names(self) -> list[str]:
        par_names = []
        for structure in self.structures:
            par_names += [parameter.name for parameter in structure.parameters]
        return par_names

    @property
    def priors(self) -> list[Optional[tuple[float, float]]]:
        priors = []
        for structure in self.structures:
            priors += [parameter.prior for parameter in structure.parameters]
        return priors

    @property
    def to_fit(self) -> list[bool]:
        to_fit = []
        for structure in self.structures:
            to_fit += [
                parameter.fit[self.cur_round] for parameter in structure.parameters
            ]
        return to_fit

    @property
    def to_fit_ever(self) -> list[bool]:
        to_fit = []
        for structure in self.structures:
            to_fit += [parameter.fit_ever for parameter in structure.parameters]
        return to_fit

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

    def update(self, vals, errs, chisq):
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

        Arguments:

            params: An array of model parameters.

            tod: A minkasi tod instance.
                'dx' and 'dy' must be in tod.info and be in radians.

        Returns:

            grad: The gradient of the model with respect to the model parameters.

            pred: The model with the specified substructure.
        """
        dx = tod.info["dx"] * wu.rad_to_arcsec
        dy = tod.info["dy"] * wu.rad_to_arcsec
        argnums = tuple(np.where(self.to_fit)[0] + core.ARGNUM_SHIFT_TOD)

        pred, grad = core.model_tod_grad(
            self.xyz,
            *self.n_struct,
            self.dz,
            self.beam,
            dx,
            dy,
            argnums,
            *params,
        )

        pred = jax.device_get(pred)
        grad = jax.device_get(grad)

        return grad, pred

    def save(self, path: str):
        """
        Serialize the model to a file with dill.

        Arguments:

            path: The file to save to.
        """
        with open(path, "wb") as f:
            dill.dump(self, f)

    @classmethod
    def load(cls, path: str) -> Self:
        """
        Load the model from a file with dill.

        Arguments:

            path: The path to the saved model
        """
        with open(path, "rb") as f:
            return dill.load(f)

    @classmethod
    def from_cfg(cls, cfg: dict) -> Self:
        """
        Create an instance of model from a witch config.

        Arguments:

            cfg: Config loaded into a dict.
        """
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
                        "to_fit has %d entries but we only have %d rounds",
                        len(fit),
                        n_rounds,
                    )
                priors = param.get("priors", None)
                if priors is not None:
                    priors = eval(str(priors))
                parameters.append(Parameter(par_name, fit, val, 0.0, priors))
            structures.append(Structure(name, structure["structure"], parameters))
        name = cfg["model"].get(
            "name", "-".join([structure.name for structure in structures])
        )

        return cls(name, structures, xyz, dz, beam, n_rounds)
