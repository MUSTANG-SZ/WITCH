"""
Module for dataset container and protocols for defining
the spec of the required functions for all datasets.
"""

from dataclasses import dataclass, field
from typing import Protocol, Self, runtime_checkable

import numpy as np
from jax import Array
from jitkasi.noise import NoiseModel
from jitkasi.solutions import SolutionSet
from jitkasi.tod import TODVec
from mpi4py import MPI

from .containers import Model
from .objective import ObjectiveFunc

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


@dataclass
class DataSet:
    """
    Class for storing a dataset.

    Attributes
    ----------
    name : str
        The name of the dataset.
    get_files : GetFiles
        The function to get the file list for this dataset.
    load : Load
        The function to load data for this dataset.
    get_info : GetInfo
        The function to get the info dict for this dataset.
    make_beam : MakeBeam
        The function to make the beam for this dataset.
    preproc : PreProc
        The function to run preprocessing for this dataset.
    postproc : PostProc
        The function to run postprocessing for this dataset.
    postfit : PostFit
        The function to run after fitting this dataset.
    info : dict
        The info dict for this dataset.
        This field is not part of the initialization function.
    datavec : DataVec
        The data vector for this data.
        This will be a `jitkasi` container class.
        This field is not part of the initialization function.
    noise_class : NoiseModel
        The class of the noise model that will be used for this dataset.
        This field is not part of the initialization function.
    noise_args : tuple
        Positional arguments to be used by the noise model.
        This field is not part of the initialization function.
    noise_kwargs : dict
        Keyword arguments to be used by the noise model.
        This field is not part of the initialization function.
    """

    name: str
    get_files: GetFiles
    load: Load
    get_info: GetInfo
    make_beam: MakeBeam
    preproc: PreProc
    postproc: PostProc
    postfit: PostFit
    info: dict = field(init=False)
    datavec: DataVec = field(init=False)
    noise_class: NoiseModel = field(init=False)
    noise_args: tuple = field(init=False)
    noise_kwargs: dict = field(init=False)

    def __post_init__(self: Self):
        assert isinstance(self.get_files, GetFiles)
        assert isinstance(self.load, Load)
        assert isinstance(self.get_info, GetInfo)
        assert isinstance(self.make_beam, MakeBeam)
        assert isinstance(self.preproc, PreProc)
        assert isinstance(self.postproc, PostProc)
        assert isinstance(self.postfit, PostFit)

    def __setattr__(self, name, value):
        if name == "info":
            if "mode" not in value:
                raise ValueError("Cannot set dataset info without a 'mode' field")
            if value["mode"] not in ["tod", "map"]:
                raise ValueError("Dataset info contained invalid mode")
            if "objective" not in value:
                raise ValueError("Cannot set dataset info without an 'objective' field")
            if not isinstance(value["objective"], ObjectiveFunc):
                raise ValueError("Dataset info contained invalid objective function")
        return super().__setattr__(name, value)

    @property
    def mode(self: Self) -> str:
        """
        Get the mode for this dataset.
        Will be `tod` or `map`.

        Returns
        -------
        mode : str
            The dataset mode.
        """
        return self.info["mode"]

    @property
    def objective(self: Self) -> ObjectiveFunc:
        """
        Get the objective function for this dataset.

        Returns
        -------
        objective : ObjectiveFunc
            The objective function.
        """
        return self.info["objective"]

    def check_completeness(self: Self):
        """
        Check if all fields are actually populated and raise an error if not.

        Raises
        ------
        ValueError
            If the dataset is missing some fields.
            If `self.info` is missing some required info.
            If `self.mode` is not a valid mode.
            If `self.objective` is not a valid objective function.
        """
        missing = [
            fname
            for fname in self.__dataclass_fields__.keys()
            if fname not in self.__dict__
        ]
        if len(missing) > 0:
            raise ValueError(f"Datset is missing the following fields: {missing}")

        required_info = np.array(
            ["mode", "objective", "noise_class", "noise_args", "noise_kwargs"]
        )
        contained_info = list(self.info.keys())
        missing_info = required_info[~np.isin(required_info, contained_info)]
        if len(missing_info) > 0:
            raise ValueError(
                f"(Dataset info is missing the following fields: {missing_info}"
            )

        if self.info["mode"] not in ["tod", "map"]:
            raise ValueError("Dataset info contained invalid mode")
        if not isinstance(self.info["objective"], ObjectiveFunc):
            raise ValueError("Dataset info contained invalid objective function")
