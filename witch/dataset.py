"""
Module for dataset container and protocols for defining
the spec of the required functions for all datasets.
"""

from dataclasses import dataclass, field
from inspect import Parameter
from typing import TYPE_CHECKING, Protocol, Self, runtime_checkable

import numpy as np
from jax import Array
from jax.tree_util import register_pytree_node_class
from jitkasi.noise import NoiseModel
from jitkasi.solutions import SolutionSet
from jitkasi.tod import TODVec
from mpi4py import MPI

from . import utils as wu
from .objective import ObjectiveFunc
from .utils import beam_conv, beam_conv_vec

DataVec = TODVec | SolutionSet

if TYPE_CHECKING:
    from .containers import MetaModel


@register_pytree_node_class
@dataclass
class MetaData:
    """
    Class for storing and applying metdata to a dataset.
    You should subclass this for your individual metadata implementation.

    Attributes
    ----------
    include : tuple[str, ...]
        Used when computing metamodel mapping.
        If provided only models whose names are listed will
        have this metadata applied. If an empty tuple is provided
        then all models will have this metadata applied by default.
    exclude : tuple[str, ...]
        Used when computing metamodel mapping.
        If provided  models whose names are listed will not
        have this metadata applied. If an empty tuple is provided
        then no models will be excluded by default.
        This exclusion list is applied after the inclusion list,
        so a model listed in both will be excluded.
    """

    include: tuple[str, ...] = field(default_factory=tuple, kw_only=True)
    exclude: tuple[str, ...] = field(default_factory=tuple, kw_only=True)

    def check_apply(self, model_name) -> bool:
        """
        Check based on `self.include` and `self.exclude` if we should
        apply this metadata to a given model.

        Parameters
        ----------
        model_name : str
            The name of the model to apply.

        Returns
        -------
        check : bool
            True if this metadata should be applied.
            False if not.
        """
        if len(self.include) != 0 and model_name not in self.include:
            return False
        return model_name not in self.exclude

    def apply(self, model: Array) -> Array:
        """
        Apply the metdata to the model.
        This is the model prior to projection into the datavector.

        Parameters
        ----------
        model : Array
            The model as defined on a grid.

        Returns
        -------
        applied : Array
            The model with the metadata applied.
        """
        return model

    def apply_grad(self, model_grad: Array) -> Array:
        """
        Apply the metdata to the model gradient.
        This is the model gradient prior to projection into the datavector.

        Parameters
        ----------
        model_gradient : Array
            The model gradient defined on a grid.

        Returns
        -------
        applied : Array
            The model gradient with the metadata applied.
        """
        return model_grad

    def apply_proj(self, model_proj: Array) -> Array:
        """
        Apply the metdata to the model.
        This is the model after projection into the datavector.

        Parameters
        ----------
        model : Array
            The model projected to the datavector.

        Returns
        -------
        applied : Array
            The model with the metadata applied.
        """
        return model_proj

    def apply_grad_proj(self, model_grad_proj: Array) -> Array:
        """
        Apply the metdata to the model gradient.
        This is the model gradient after projection into the datavector.

        Parameters
        ----------
        model_gradient_proj : Array
            The model gradient projected to the datavector.

        Returns
        -------
        applied : Array
            The model gradient with the metadata applied.
        """
        return model_grad_proj

    # Functions for making this a pytree
    # Don't call this on your own
    def tree_flatten(self) -> tuple[tuple, tuple]:

        return (tuple(), (self.include, self.exclude))

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> Self:
        _ = children
        include, exclude = aux_data

        return cls(include=include, exclude=exclude)


@register_pytree_node_class
@dataclass
class BeamConvAndPrefac(MetaData):
    """
    Apply beam convolution and a prefactor (for unit conversion) to a model.
    Only attributes unique to this subclass are listed here.
    See `MetaData` for the rest.

    Attributes
    ----------
    beam : jax.Array
        The beam to convolve both the model and gradient with.
    prefactor : jax.Array
        Scalary array containing a value to multiply the model and gradients by.
    """

    beam: Array
    prefactor: Array

    def apply(self, model: Array) -> Array:
        """
        Convolve the model by the beam and then multiply by the prefactor.
        See `MetaData.apply` for details on parameters and returns.
        """
        return self.prefactor * beam_conv(model, self.beam)

    def apply_grad(self, model_grad: Array) -> Array:
        r"""
        Convolve the gradient by the beam and then multiply by the prefactor.
        Here we are taking advantage of the following:

        $$
        \frac{\partial}{\partial x}(p * (b \circledast m))
         = p\frac{\partial}{\partial x}(b \circledast m)
         = p * b \circledast \frac{\partial m}{\partial x}
        $$

        Where $m$ is the model, $x$ is some parameter, $p$ is the prefactor, and $b$ is the beam.

        See `MetaData.apply_grad` for details on parameters and returns.
        """
        return self.prefactor * beam_conv_vec(model_grad, self.beam)

    # Functions for making this a pytree
    # Don't call this on your own
    def tree_flatten(self) -> tuple[tuple, tuple]:

        children = (self.beam, self.prefactor)
        aux_data = (self.include, self.exclude)

        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> Self:
        include, exclude = aux_data

        return cls(*children, include=include, exclude=exclude)


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
    This function is also responsible for updating the comm object to
    the local one in any relevant libraries.
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
class MakeMetadata(Protocol):
    """
    Function that makes the metadata.
    If you don't need it just write a dummy function to return an empty tuple.
    See docstring of `__call__` for details on the parameters and returns.
    """

    def __call__(
        self: Self, dset_name: str, cfg: dict, info: dict
    ) -> tuple[MetaData, ...]:
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
        metadata : tuple[MetaData, ...]
            Tuple of MetaData instances.
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
        dset: "DataSet",
        cfg: dict,
        metamodel: "MetaModel",
    ):
        """
        Parameters
        ----------
        dset : str
            The dataset to preproc.
        cfg : dict
            The loaded `witcher` config.
        metamodel : MetaModel
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
        dset: "DataSet",
        cfg: dict,
        metamodel: "MetaModel",
    ):
        """
        Parameters
        ----------
        dset : str
            The dataset to postproc.
        cfg : dict
            The loaded `witcher` config.
        metamodel : MetaModel
            The cluster model.
            At this point this will just be the initial state of the model.
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
        dset: "DataSet",
        cfg: dict,
        metamodel: "MetaModel",
    ):
        """
        Parameters
        ----------
        dset : str
            The dataset to process after fitting.
        cfg : dict
            The loaded `witcher` config.
        metamodel : MetaModel
            The cluster model.
            This will contain the final best fit parameters.
        """
        ...


@register_pytree_node_class
@dataclass
class DataSetData:
    """
    Class for storing the data associated with a dataset.
    This is the minimal amount needed when actually fitting.

    Attributes
    ----------
    name : str
        The name of the dataset.
    mode : str
        The mode of the dataset, should be "tod" or "map".
    objective : ObjectiveFunc
        The objective function to use.
    datavec : DataVec
        The data vector for this data.
        This will be a `jitkasi` container class.
        This field is not part of the initialization function.
    metadata : tuple[MetaData]
        Tuple of `MetaData` instances to apply to model.
        This field is not part of the initialization function.
    noise_class : NoiseModel
        The noise model class used by this dataset.
    noise_args : tuple
        Positional arguments used when computing the noise model.
    noise_kwargs : dict
        Keyword arguments used when computing the noise model.
    """

    name: str
    mode: str
    objective: ObjectiveFunc
    datavec: DataVec
    metadata: tuple[MetaData, ...]
    noise_class: NoiseModel
    noise_args: tuple
    noise_kwargs: dict

    # Functions for making this a pytree
    # Don't call this on your own
    def tree_flatten(self) -> tuple[tuple, tuple]:
        children = (self.datavec, self.metadata)
        aux_data = (
            self.name,
            self.mode,
            self.objective,
            self.noise_class,
            self.noise_args,
            self.noise_kwargs,
        )

        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> Self:
        datavec, metadata = children
        name, mode, objective, noise_class, noise_args, noise_kwargs = aux_data
        return cls(
            name,
            mode,
            objective,
            datavec,
            metadata,
            noise_class,
            noise_args,
            noise_kwargs,
        )


@register_pytree_node_class
@dataclass
class DataSet:
    """
    Class for storing a dataset.
    Note that all atributes of `DataSet.data` are exposed as
    properties here, see `DataSetData` for details.

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
    make_metadata : MakeMetadata
        The function to make the metadata for this dataset.
    preproc : PreProc
        The function to run preprocessing for this dataset.
    postproc : PostProc
        The function to run postprocessing for this dataset.
    postfit : PostFit
        The function to run after fitting this dataset.
    global_comm : MPI.Comm | MPI.Intracomm
        The MPI communicator used for all datasets,
        not the local one just for this data.
    info : dict
        The info dict for this dataset.
        This field is not part of the initialization function.
    data : DataSetData
        The data needed for fitting. See `DataSetData` for details.
    """

    name: str
    get_files: GetFiles
    load: Load
    get_info: GetInfo
    make_metadata: MakeMetadata
    preproc: PreProc
    postproc: PostProc
    postfit: PostFit
    global_comm: MPI.Comm | MPI.Intracomm | wu.NullComm
    info: dict = field(default_factory=dict)
    data: DataSetData = field(init=False)

    def __post_init__(self: Self):
        assert isinstance(self.get_files, GetFiles)
        assert isinstance(self.load, Load)
        assert isinstance(self.get_info, GetInfo)
        assert isinstance(self.make_metadata, MakeMetadata)
        assert isinstance(self.preproc, PreProc)
        assert isinstance(self.postproc, PostProc)
        assert isinstance(self.postfit, PostFit)

    @property
    def mode(self):
        return self.data.mode

    @property
    def objective(self):
        return self.data.objective

    @property
    def datavec(self):
        return self.data.datavec

    @property
    def metadata(self):
        return self.data.metadata

    @property
    def noise_class(self):
        return self.data.noise_class

    @property
    def noise_args(self):
        return self.data.noise_args

    @property
    def noise_kwargs(self):
        return self.data.noise_kwargs

    def check_completeness(self: Self):
        """
        Check if all fields are actually populated and raise an error if not.

        Raises
        ------
        ValueError
            If the dataset is missing some fields.
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

        if self.data.mode not in ["tod", "map"]:
            raise ValueError("Dataset info contained invalid mode")
        if not isinstance(self.info["objective"], ObjectiveFunc):
            raise ValueError("Dataset info contained invalid objective function")

    # Functions for making this a pytree
    # Don't call this on your own
    def tree_flatten(self) -> tuple[tuple, tuple]:
        children = tuple()
        if "data" in self.__dict__:
            children = (self.data,)
        aux_data = (
            self.name,
            self.get_files,
            self.load,
            self.get_info,
            self.make_metadata,
            self.preproc,
            self.postproc,
            self.postfit,
            self.global_comm,
        )

        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> Self:
        dataset = cls(*aux_data)
        if len(children) > 0:
            dataset.data = children[0]
        return dataset
