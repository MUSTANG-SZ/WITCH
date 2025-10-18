"""
Data classes for describing models in a structured way.
"""

from dataclasses import dataclass
from typing import Self

import jax
import numpy as np
from jax.tree_util import register_pytree_node_class
from typing_extensions import Self

from ..structure import STRUCT_N_PAR


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
    fit: tuple[bool, ...]  # 1d bool array
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
    structure_order : int
        The order of the structure.
        See `core` for details.
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
    structure_order: int
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
        children = (self.structure_order, tuple(self.parameters))
        aux_data = (self.name, self.structure, self.n_rbins)

        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> Self:
        name, structure, n_rbins = aux_data
        structure_order, parameters = children

        return cls(name, structure, structure_order, list(parameters), n_rbins)
