"""
Protocols that define the spec for external functions.
"""

from typing import Protocol, Self, runtime_checkable

from jax import Array
from jitkasi.solutions import SolutionSet
from jitkasi.tod import TODVec
from mpi4py import MPI

from .containers import Model

DataVec = TODVec | SolutionSet


@runtime_checkable
class GetFiles(Protocol):
    """
    Function that returns a list of files to be loaded for this dataset.
    Technically these do not have the be filepaths, just a list where each
    entry is the information needed to load the data.
    See docstring of `__call__` for details on the parameters and returns.
    """

    def __call__(self: Self, dset_name: str, cfg: dict) -> list:
        """
        Parameters
        ----------
        dset_name : str
            The name of the dataset to get file list for.
        cfg : dict
            The loaded `witcher` config.

        Returns
        -------
        file_list : list
            A list where each entry contains the information needed to load a
            discrete piece of data (ie: a TOD or map) for this dataset.
            The format of the entries are up to the dataset but the number of
            entries must match the number of things loaded for MPI planning
            purposes.
        """
        ...


@runtime_checkable
class Load(Protocol):
    """
    Function that loads data into a `jitkasi` container.
    See docstring of `__call__` for details on the parameters and returns.
    """

    def __call__(
        self: Self, dset_name: str, cfg: dict, fnames: list, comm: MPI.Intracomm
    ) -> DataVec:
        """
        Parameters
        ----------
        dset_name : str
            The name of the dataset to get file list for.
        cfg : dict
            The loaded `witcher` config.
        fnames : list
            Some subset of the output of `GetFiles`.
        comm : MPI.Intracomm
            The MPI communicator to pass to the `jitkasi` container.

        Returns
        -------
        datavec : DataVec
            The `jitkasi` container for the data.
            This is going to be `TODVec` for TODs
            and `SolutionSet` for maps.
        """
        ...


@runtime_checkable
class GetInfo(Protocol):
    """
    Function that gets information that will be used by other functions for this dataset.
    At the minimum this should contain:

    * `mode`: a string that is either `tod` or `map` that detemines how the dataset is treated.
    * `objective`: a function pointer to an objective function. See `witch.objective.ObjectiveFunc` for details.

    See docstring of `__call__` for details on the parameters and returns.
    """

    def __call__(self: Self, dset_name: str, cfg: dict, datavec: DataVec) -> dict:
        """
        Parameters
        ----------
        dset_name : str
            The name of the dataset to get file list for.
        cfg : dict
            The loaded `witcher` config.
        datavec : DataVec
            The `jitkasi` container for the data.
            This is going to be `TODVec` for TODs
            and `SolutionSet` for maps.


        Returns
        -------
        info : dict
            Dictionairy containing information.
            Must at least contain `mode` and `objective`.
        """
        ...


@runtime_checkable
class MakeBeam(Protocol):
    """
    Function that makes the beam array.
    If you don't need a beam just write a dummy function to return `jnp.array([[1]])`.
    See docstring of `__call__` for details on the parameters and returns.
    """

    def __call__(self: Self, dset_name: str, cfg: dict, info: dict) -> Array:
        """
        Parameters
        ----------
        dset_name : str
            The name of the dataset to get file list for.
        cfg : dict
            The loaded `witcher` config.
        info : dict
            Dictionairy containing dataset information.

        Returns
        -------
        beam : Array
            The beam to be convolved with the model.
            Should be a 2D array.
        """
        ...


@runtime_checkable
class PreProc(Protocol):
    """
    Function that runs before the data vector is processed.
    (see `witch.fitter.process_tods` and `witch.fitter.process_maps`).
    This is where you may want to compute something about the data's noise properties
    or some other statistic that may be useful to your analysis.
    You can also do nothing if you wish.
    See docstring of `__call__` for details on the parameters and returns.
    """

    def __call__(
        self: Self,
        dset_name: str,
        cfg: dict,
        datavec: DataVec,
        model: Model,
        info: dict,
    ):
        """
        Parameters
        ----------
        dset_name : str
            The name of the dataset to get file list for.
        cfg : dict
            The loaded `witcher` config.
        datavec : DataVec
            The `jitkasi` container for the data.
            This is going to be `TODVec` for TODs
            and `SolutionSet` for maps.
        model : Model
            The cluster model.
            At this point this will just be the initial state of the model.
        info : dict
            Dictionairy containing dataset information.
        """
        ...


@runtime_checkable
class PostProc(Protocol):
    """
    Function that runs after the data vector is processed.
    (see `witch.fitter.process_tods` and `witch.fitter.process_maps`).
    This is where you may want make some visualization or initial analysis of your data
    (ie. make a map from your TODs, improve the noise model estimation, etc.)
    You can also do nothing if you wish.
    See docstring of `__call__` for details on the parameters and returns.
    """

    def __call__(
        self: Self,
        dset_name: str,
        cfg: dict,
        datavec: DataVec,
        model: Model,
        info: dict,
    ):
        """
        Parameters
        ----------
        dset_name : str
            The name of the dataset to get file list for.
        cfg : dict
            The loaded `witcher` config.
        datavec : DataVec
            The `jitkasi` container for the data.
            This is going to be `TODVec` for TODs
            and `SolutionSet` for maps.
        model : Model
            The cluster model.
            At this point this will just be the initial state of the model.
        info : dict
            Dictionairy containing dataset information.
        """
        ...


@runtime_checkable
class PostFit(Protocol):
    """
    Function that runs after all fitting stages are over.
    This is where you may want make some visualization or initial analysis of your data
    (ie. plot residuals, check statistical significance, etc.)
    You can also do nothing if you wish.
    See docstring of `__call__` for details on the parameters and returns.
    """

    def __call__(
        self: Self,
        dset_name: str,
        cfg: dict,
        datavec: DataVec,
        model: Model,
        info: dict,
    ):
        """
        Parameters
        ----------
        dset_name : str
            The name of the dataset to get file list for.
        cfg : dict
            The loaded `witcher` config.
        datavec : DataVec
            The `jitkasi` container for the data.
            This is going to be `TODVec` for TODs
            and `SolutionSet` for maps.
        model : Model
            The cluster model.
            This will contain the final best fit parameters.
        info : dict
            Dictionairy containing dataset information.
        """
        ...
