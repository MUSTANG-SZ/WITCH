# WITCH (WHERE IS THAT CLUSTER HIDING) 

This repository contains tools for modeling and fitting SZ data of galaxy clusters.
While this code was originally written for [MUSTANG-2](https://greenbankobservatory.org/science/gbt-observers/mustang-2/) it is largely generic enough to work with data for other telescopes.

## The `WITCH` Library

The core of this repository is the `WITCH` library.
While the name contains `minkasi` it does not actually require the `minkasi` mapmaker to be used;
rather, it produces models of galaxy clusters and their gradients in a format that can be consumed by `minkasi`'s fitting code.

The core concept of `WITCH` is to model the cluster as a 3D pressure profile and then apply modifications to that profile to represent substructure.
For example, a cavity can be modeled as a multiplicative suppression of the pressure within a certain region.
The profile is then integrated along the line of sight to produce a signal like we would observe via the SZ effect.
To produce gradients of the clusters and JIT expensive computations we employ [`jax`](https://github.com/google/jax).

This framework makes it very easy to add new types of models, see the [Contributing](#contributing) section for more.

## The `fitter.py` Script

The other main part of this repository is the `fitter.py` script.
It is generically a script to perform fit models and make maps using [`minkasi`](https://github.com/sievers/minkasi),
but there are a two key points that make it nice to use:

1. A flexible configuration system that allows the user to control mapmaking and fitting parameters, model specification, IO, etc. via yaml files.
2. First class support for models from the `WITCH` library.

For the most part the config files are easy to make by using one of the files in the `configs` folder as a base.
However there are some subtleties and advanced configurations that will eventually get documented properly.

## Installation

To install the `WITCH` library first clone this repository and from within it run:
```
pip install .
```
Note that this will only install `WITCH` and its dependencies,
to also install dependencies for `fitter.py` do:
```
pip install .[fitter]
```
If you are going to be actively working on the `WITCH` library you probably want to include the `-e` flag.

All the dependencies should be installed by `pip` with the one exception being `minkasi` itself (only needed for `fitter.py`).
Instructions on installing `minkasi` can be found [here](https://github.com/sievers/minkasi#installation).

## Contributing

All are welcome to contribute to this repository, be it code or config files.
In general contributions other than minor changes should follow the branch/fork -> PR -> merge workflow.
If you are going to contribute regularly, contact one of us to get push access to the repository.

### Style and Standards
In general contributions should be [PEP8](https://peps.python.org/pep-0008/) with commits in the [conventional commits](https://www.conventionalcommits.org/en/v1.0.0/) format.
This library follows [semantic versioning](https://semver.org/), so changes that bump the version should do so by editing `pyproject.toml`.

In order to make following these rules easier this repository is setup to work with [commitizen](https://commitizen-tools.github.io/commitizen/) and [pre-commit](https://pre-commit.com/).
It is recommended that you make use of these tools to save time.

#### Getting Started
1. Install both tools with `pip install commitizen pre-commit`.
2. `cd` into the `WITCH` repository it you aren't already in it.
3. (Optional) Setup `commitizen` to automatically run when you run `git commit`. Follow instruction [here](https://commitizen-tools.github.io/commitizen/tutorials/auto_prepare_commit_message/).
4. Make sure the `pre-commit` hook is installed by running `pre-commit install`.

#### Example Workflow
1. Make a branch for the edits you want to make.
2. Code.
3. Commit your code with a [conventional commit message](https://www.conventionalcommits.org/en/v1.0.0/#summary).
  * `cz c` gives you a wizard that will do this for you, if you followed Step 3 above then `git commit` will also do this (but not `git commit -m`).
4. Repeat step 3 and 4 until the goal if your branch has been completed.
5. Put in a PR.
5. Once the PR is merged the repo version and tag will update [automatically](https://commitizen-tools.github.io/commitizen/tutorials/github_actions/).

### Adding New Models 

When adding new models to `WITCH`, be they profiles or substructure, there are some changes that need to be made to `core.py` to expose them properly.

1. A variable `N_PAR_{MODEL}` needs to be defined with the number of fittable parameters in the model. Do not include parameters like the grid here.
2. A parameter `n_{model}` needs to be added to the functions `helper`, `model`, and `model_grad`. Remember to update the `static_argnums` for `model` and `model_grad`. In `helper` set a default value of `0` for backwards compatibility.
3. A block grabbing the parameters for the model needs to be added to `model`. This can largely be copied from the other models, just remember to swap out the relevant variables.
4. A block applying model needs to be added to `model`. Pressure profiles should come first then substructure. This can largely be copied from the other models, just remember to swap out the relevant variables.

Adding a new model also (usually) means you should bump the minor version in the version number.

### Profiling Code

The script `scratch/profile.py` uses `jax` profiling tools to benchmark the library.
It outputs a trace file understandable [perfetto](https://ui.perfetto.dev/) as well as a text file containing
metadata about the software and hardware used while profiling.
To use non default settings use `python profile.py --help` but in most cases the default settings are fine.

The profiling script has some additional dependencies.
To install them run:
```
pip install .[profile]
```
