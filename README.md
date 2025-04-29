# WITCH (WHERE IS THAT CLUSTER HIDING) 

This repository contains tools for modeling and fitting SZ data of galaxy clusters.
While this code was originally written for [MUSTANG-2](https://greenbankobservatory.org/science/gbt-observers/mustang-2/) it is largely generic enough to work with data for other telescopes.

## The `WITCH` Library

The core of this repository is the `WITCH` library. `WITCH` produces models of galaxy clusters and their gradients in a format that can be consumed by [`minkasi`](https://github.com/sievers/minkasi)'s fitting code.

The core concept of `WITCH` is to model the cluster as a 3D pressure profile and then apply modifications to that profile to represent substructure.
For example, a cavity can be modeled as a multiplicative suppression of the pressure within a certain region.
The profile is then integrated along the line of sight to produce a signal like we would observe via the SZ effect.
To produce gradients of the clusters and JIT expensive computations we employ [`jax`](https://github.com/google/jax).

This framework makes it very easy to add new types of models, see the [Contributing](#contributing) section for more.

## End Users and the `fitter.py` Script

End users (i.e., users who will not be developing `WITCH`) should interact with the software entirely by writing configs yamls.
Once `WITCH` is built, the command line executable `witcher` will be made available, which basically wraps `fitter.py`. 
This fitter injests config files and performs the fitting. The usage is
```
mpirun -n N witcher /PATH/TO/my_config.yaml
```
where the `mpirun` call is optional and will automatically run the code in parallel. The screen output can of course optionally
be pipped to an output file, but necessary outputs (model parameters, uncertainties, etc.) will be saved as part of the `witcher` call.


## Writing Config Files

This section is an abreviation of the more extensive documentation available [here](https://mustang-sz.github.io/WITCH/latest/getting_started/).

For end users, config files are the only thing you need to change to run `WITCH`. In these files, you specify what data you would like to fit,
how you would like to fit it, and what model you would like to fit to that data. Config files can refer to other config files, in which case they
will be collated into one config before execution. This makes it easy to fit your data to many different models, by specifying one `_base.yaml` which
defines the data processing, and a series of `_model.yaml`s which specify the model to be fit and refer back to `_base.yaml`. When running `witcher`,
the highest level yaml should be passed as a commandline argument; in the example below, that is `RXJ1347_a10.yaml`. 

We will use the `yaml` files `RXJ1347_a10.yaml` and `base_unit.yaml`, both found in the `unit_tests` directory as examples. These files are well commented
and should be read through in addition to reading this section. Starting with `base_unit.yaml`, this file defines the data used in the fit and which data 
processing routines to use. These configs are a good place to start for writing your own configs, especially if you are fitting MUSTANG-2 data.

Firstly, `fit` and `sub` tell the script to fit the model to the data and to make residuals of the model to the data. `nrounds` 
tells the script how many rounds of fitting to perform. `contstants` defines a number of constants used in the fitting, including `Te` the electron 
temperature, `freq`, the frequency of observation, and `z`, the redshift of the cluster. `paths` specifies the path `witcher` should search to obtain data to fit, as
well as the outroot to save the data to. All paths are relative to the global environmental variables `WITCH_DATAROOT` (for data) and `WITCH_OUTROOT` (for outputs),
which you should set in your `.bashrc`.

`coords` defines the grid on which the model will be built. The larger and higher resolution the grid, the more accurate the results will be,
at the cost of computation time. In general, `r_map` should be larger than the largest scale in your model you are interested in. 
Keep in mind atmospheric effects restrict MUSTANG-2 to recover scales of 2.5' radisu and smaller. `dr` should be a few times smaller than
the instrument beam, which is 9" for MUSTANG-2. `dz` is the line-of-sight (LoS) resolution and is to integrate the model along the LoS. 
Generally the accuracy is less sensitive to `dz`, which can be set somewhat higher than `dr`. If you're unsure what to do for `grid`,
the defaults in `base_config.yaml` should be safe.

`datasets` defines the functions which load, process, and apply noise to the data used by `WITCH`. These are defined on a per-instrument basis 
for the purposes of joint fitting; if you are only fitting one data set, only one sub-header should be defined. The `noise` field defines what noise model
to use when fitting. `funcs` defines the experiment-specific functions which are used to load the data, perform preprocessing, etc.
These are defined in the `WITCH/external` submodule. `beam` defines the beam parameters, while `dograd` and `npass` are options for mapmakting
which can be left to their defaults. If you are only working with MUSTANG-2 data, all these values can be left at their defaults
from `base_config.yaml`.

`fitter` sets hyper-parameters for the fitting routine, specifically the maximum number of interations to take per step
and the `chitol` which sets delta-chi-squared minimum for a step to be considered converged. Finally, `imports` defines additional
packages to be loaded into the script at runtime. Again, these can all be left at their default values.

`RXJ1347_a10` defines a specific [Arnaud 2010](https://arxiv.org/abs/0910.1234) model which will be fit to the data loaded in `base_unit.yaml`. 
First, the `base` header tells `witcher` to load the config specified in `base` as part of the full config. `name` is self explanatory, although
note this has to match the name used for the TOD directory. `noise_map` and `model_map` tell `witcher` to make maps of just the noise and of
the model, respectively. 

The `model` contains the actual specification for the model that will be fit to the data. At the top, `unit_conversion` defines the conversion
from the units of the model (typically pressure in something like keV / cm^3) to map units (K Rayleigh–Jeans for MUSTANG-2). If you are
only fitting MUSTANG-2, you should stick to using the default specified in `RXJ1347_a10`. This unit conversion,
```
float(wu.get_da(constants['z'])*wu.y2K_RJ(constants['freq'], constants['Te'])*wu.XMpc/wu.me)
```
first gets the angular dymeter distance to convert from radians to Mpc, then multiplies by the conversion from Compton-y to K RJ, then finally
multiplies by the Thompson scattering cross section divided by the mass of the elctron, which is the conversion from integrated pressure
to Compton-y (see [Mroczkowski 2019, eqn. 5](https://arxiv.org/abs/1811.02310). 

Next the various model componants are specified. In this case, we are fitting an A10 model plus a point source. `structures` indicates
that we have moved on to specifying the model components, while `a10` names the first component. Under `a10`, `structure: a10` specifies that
this component is a `class a10` from `witch.structures`. Note the `a10` header is the name of the component, which is only used to refer to 
this particular componant, while `structure: a10` actually defines what structure it is. This is useful if you have, say, 2 A10s, in which case
you could have 
```
a10_1:
  structure: 10
  ...
a10_2:
  structure: 10
  ...
```
All available structures are defined in [`witch.structure`](https://mustang-sz.github.io/WITCH/latest/reference/structure/). The 
documenation available at that link specifies what arguments, units, etc are expected by each type of structure. After the type of
structure is defined, the `parameters` header indicates that each parameter will now be defined. Taking as an example `dx_1`:, the 
`value` header indicates the inital value. The `to_fit` header can be either a list of booleans of exactly `n_rounds` length or a boolean. 
If it is a list, for each round the parameter will either be fit or held constant at its last value depending on the value of the list
at that index. If it is a boolean, than it will either always be fit or always be held constant. `True` indicates fit while `False` indicates
held constant. Ffinally, the `priors` header is a list of two float, which define the lower and upper limits for a flat prior for that parameter.
Currently only flat priors supported. After `dx_1`, each parameter is specified in turn. Note `dx_1` is just the name of the paramter, and 
does not need to match the parameter of `a10`; the parameters will be fed as arguments to `a10` in the order they are specified.

Finally, we then specify another structural component, called `ps_gaus` is defined, this time a `structure: gaussian`. Again, all the paramters of
the component are specified. Note that the order of structures is unimportant; they can be specified in any order in the configuration file.
`RXJ1347_a0.yaml` can be run via `witcher RXJ1347_a0.yaml`; it should produce the results stored in `RXJ1347_unit.zip`.

## Installation

To install the `WITCH` library first clone this repository and from within it run:
```
pip install .
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
