# minkasi_jax

This repository contains tools for modeling and fitting SZ data of galaxy clusters.
While this code was originally written for [MUSTANG-2](https://greenbankobservatory.org/science/gbt-observers/mustang-2/) it is largely generic enough to work with data for other telescopes.

## The `minkasi_jax` Library

The core of this repository is the `minkasi_jax` library.
While the name contains `minkasi` it does not actually require the `minkasi` mapmaker to be used;
rather, it produces models of galaxy clusters and their gradients in a format that can be consumed by `minkasi`'s fitting code.

The core concept of `minkasi_jax` is to model the cluster as a 3D pressure profile and then apply modifications to that profile to represent substructure.
For example, a cavity can be modeled as a multiplicative suppression of the pressure within a certain region.
The profile is then integrated along the line of sight to produce a signal like we would observe via the SZ effect.
To produce gradients of the clusters and JIT expensive computations we employ [`jax`](https://github.com/google/jax).

This framework makes it very easy to add new types of models, see the [Contributing](#contributing) section for more.

## The `fitter.py` Script

The other main part of this repository is the `fitter.py` script.
It is generically a script to perform fit models and make maps using [`minkasi`](https://github.com/sievers/minkasi),
but there are a two key points that make it nice to use:

1. A flexible configuration system that allows the user to control mapmaking and fitting parameters, model specification, IO, etc. via yaml file.
2. First class support for models from the `minkasi_jax` library.

For the most part the config files are easy to make by using one of the files in the `configs` folder as a base.
However there are some subtleties and advanced configurations that will eventually get documented properly.

## Installation

To install the `minkasi_jax` library first clone this repository and from within it run:
```
pip install .
```
Note that this will only install `minkasi_jax` and its dependencies,
to also install dependencies for `fitter.py` do:
```
pip install .[fitter]
```
If you are going to be actively working on the `minkasi_jax` library you probably want to include the `-e` flag.

All the dependencies should be installed by `pip` with the one exception being `minkasi` itself (only needed for `fitter.py`).
Instructions on installing `minkasi` can be found [here](https://github.com/sievers/minkasi#installation).

## Contributing

All are welcome to contribute to this repository, be it code or config files.
A few things to keep in mind:

* In general contributions other than minor changes should follow the branch/fork -> PR -> merge workflow.
* The `minkasi_jax` library loosely follows [semantic versioning](https://semver.org/) so try to remember to update the version in `setup.py` when appropriate.
* Code should be [PEP8](https://peps.python.org/pep-0008/) wherever possible. Tools like [`black`](https://github.com/psf/black) are helpful for this. It is also good to setup a linter in your text editor that tells you when you violate PEP8.

If you are going to contribute regularly, contact one of us to get push access to the repository.

### Adding New Models 

When adding new models to `minkasi_jax`, be they profiles or substructure, there are some changes that need to be made to `core.py` to expose them properly.

1. A variable `N_PAR_{MODEL}` needs to be defined with the number of fittable parameters in the model. Do not include parameters like the grid here.
2. A parameter `n_{model}` needs to be added to the functions `helper`, `model`, and `model_grad`. Remember to update the `static_argnums` for `model` and `model_grad`. In `helper` set a default value of `0` for backwards compatibility.
3. A block grabbing the parameters for the model needs to be added to `model`. This can largely be copied from the other models, just remember to swap out the relevant variables.
4. A block applying model needs to be added to `model`. Pressure profiles should come first then substructure. This can largely be copied from the other models, just remember to swap out the relevant variables.

Adding a new model also (usually) means you should bump the minor version in the version number.
