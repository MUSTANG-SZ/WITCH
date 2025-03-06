# Getting Started

This is not a comprehensive guide to all of the capabilities of `WITCH`
but hopefully it should be enough to get a basic setup working.
Useful things that will not be covered in this guide are:

* How to add new datasets to `WITCH`
* Every single configuration file option
* Joint fitting
* Tuning HMC parameters
* Adding new substructures
* How to build your model
* How each dataset works

those topics and more will be covered in other guides down the line.

## Installation

Installing `WITCH` is very easy, simply clone the repository and
run `pip install .` from the root of the repository.
A [pypi](https://pypi.org/) release is on the todo list so this will get
even easier down the line.

## Environment Setup

!!! warning

    There is some backwards compatibility for older file organization schemes
    in parts of `WITCH` they wil not be covered here and are not reccomended
    going forward.

Technically you can organize your files however you like and provide `WITCH`
with absolute paths, but it is quite convenient to use the expected environment
variables to do this instead. This is because it facilitates sharing configuration
files across multiple users and compute resources far easier.

First you need to decide two things: where your input data will live and where the outputs will be placed.
It is often useful to place the input data in a directory that is readable by your collaborators and, if possible,
backed up (ie: not scratch), you can always regenerate outputs but requiring the input data is potentially non-trivial.

Once you decide these two things set the relevant environment variables like so:

```
export WITCH_DATROOT=WHERE_YOU_WANT_INPUTS
export WITCH_OUTROOT=WHERE_YOU_WANT_OUTPUTS
```

you most likely will want to place those lines in you `.bashrc` or equivalent (or in your job scripts on a cluster).

Next you will want to place your input files in the correct place in `$WITCH_DATROOT`,
technically the exact search path within is up to the dataset implementation but the recommended default is that
the files for a given dataset live in: `$WITCH_DATROOT/DATA_ROOT/DATASET/`, where `DATA_ROOT` is a path specified
in the configuration file by the `data` field in the `paths` section (example later on) and `DATASET` is the name of
the dataset you are working with. The data files can be placed in that directory, with the structure within the directory
being dictated by the specific dataset.

`WITCH` will handle building a path within your output directory for you and will print it out for convenience as needed,
but you can customize some specifics by setting this `outroot` and `subdir` fields in the `paths` section of the configuration file.
With those set you outputs will be found in a path along the lines of `$WITCH_OUTROOT/OUTROOT/NAME/MODEL_NAME/SUBDIR`, where
`NAME` and `MODEL_NAME` are specified elsewhere in the configuration file and will be discussed in the next section.

## Building a Basic Configuration File

!!! info

    A full reference to the configuration file is under construction,
    here we just discuss what is needed for a simple setup.

A key feature of the `WITCH` configuration file is that any configuration file may reference a `base`
configuration. Any fields populated in the `base` configuration will be known when loading the actual configuration
file and any fields included in both will be overwriten by the actual configuration.
This can be done any number of times, so the `base` configuration can reference its own `base` configuration if you wish.

This is useful for placing portions of the configuration that are largely static in a file that doesn't need to be changed
and then referencing it in downstream files that modify various things for testing
(ie: configurations for different modeling assumptions).

Lets first build up a typical `base` configuration file,
starting with some very basic information that we can can use as defaults for the downstream configurations:


```yaml
name: "RXJ1347" # The name of the cluster/project
fit: True # If True fit the cluster, overridden by command line
sub: True # If True the cluster before mapmaking, overridden by command line
n_rounds: 4 # How many rounds of fitting to try
sim: False # Set to True of we are running a simulation
```

Next lets add in the paths section, the important fields here were disused above:

```yaml
paths:
    data: "RXJ1347" # Usually we will want the data in a directory
    outroot: "" # The default for this is an empty string
    subdir: "" # Same here
```

The next two sections allow us to define things that can be referenced by the rest of the configuration.
The variables defined in the `constants` section can be accessed via the `constants` dictionary in later fields,
all of the constants are passed through `eval` so some basic math is possible.
The `imports` section allows us to import modules that can be used elsewhere in the configuration file.
The syntax equivalent to `import module.a as b` is `module.a: m` and `from module import a, b` is `module: [a, b]`.
Below is an example of these two sections:

```yaml
constants:
  Te: 5.0
  freq: "90e9"
  z: 0.451
imports:
  astropy.units: u
  astropy.coordinates: [Angle]
  witch.external.minkasi.funcs: mf
  jitkasi.noise: jn
```

The next section that is useful to include is the coordinate grid, all of the fields
in the section are passed through `eval` so you can include some math if you wish.
The coordinate grid is the set of points at which the model is evaluated.
This is not the same as the map (if such a thing is relevant to your dataset).
The model called anywhere outside the grid will evaluate to 0, so make sure the grid covers the model
where there is significant power. Smaller grids have the advantage of better performance.
`dr` sets the x/y grid size and should not be smaller than half the pixel size to avoid sub-pixel interpolation errors.
`dz` is the line of sight spacing and only needs to be set small enough to avoid excess uncertainty in the numerical integration.
`x0` and `y0` should be set to the center of the cluster (or other feature of interest) and defines the grid center,
all model space references will be made with respect to this point.
An example of this section with units annotated can be seen below:

```yaml
coords:
  r_map: "3.0*60" # Radial size of grid in arcseconds
  dr: "1.0" # Pixel size of grid in x and y in arcseconds
  dz: 1.0 # Pixel size along the LOS, if not provided dr is used
  x0: "(206.8776*u.degree).to(u.radian).value" # RA of grid origin in radians
  y0: "(-11.7528*u.degree).to(u.radian).value" # Dec of grid origin in radians
```

Two other sections commonly used desired in the `base` unit are those that control the settings
for the Levenberg–Marquardt fit and Hamiltonian Monte Carlo.
The details of these two algorithms and how to tune them will be discussed in a later guide but the settings
below encompass all options for both:

```yaml
# Setting for the Levenberg–Marquardt fitter
fitting:
  maxiter: 10 # Maximum fit iterations per round of fitting
  chitol: 1e-5 # Change in chisq that we consider to be converged
# Settings for HMC
mcmc:
  run: True # If this is false the chain will not be run
  num_steps: 1000 # The number of samples in the chain
  num_leaps: 10 # The number of iterations of the leapfrog algorithm to run at each step
  step_size: .02 # The step size to use in the chain
  sample_which: -2 # Which parameters to sample, -2 will sample any parameters that we have tried to fit
```

to better understand what these fields are read the documentation for [`fit_dataset`](https://mustang-sz.github.io/WITCH/latest/reference/fitting/#witch.fitting.fit_dataset) and [`run_mcmc`](https://mustang-sz.github.io/WITCH/latest/reference/fitting/#witch.fitting.run_mcmc)

The final section that you will usually want in the `base` config is one that defines out datasets.
This can be quite complicated and will vary depending on your dataset.
This topic be covered extensively in its own guide so here we simply provide an annotated example:

```yaml
datasets:
  mustang2:
    # These fields need to be set regardless of the dataset
    # This can rely on the imports section
    noise: # Noise to use while fitting
      class: "jn.NoiseSmoothedSVD"
      args: "[]"
      kwargs: "{'fwhm':10}"
    funcs: # Functions needed to interact with your dataset
      # All of these need to exist and be importable
      # If you don't have them in a library add the relevant file to your $PYTHONPATH
      # Check docs for the needed signatures
      get_files: "mf.get_files"
      load_tods: "mf.load_tods"
      get_info: "mf.get_info"
      make_beam: "mf.make_beam"
      preproc: "mf.preproc"
      postproc: "mf.postproc"
      postfit: "mf.postfit"
    # The rest are user specified and will depend on your dataset
    # These should only ever be needed in the function set in "funcs" above
    # Since they are only called from the specified "funcs" make sure the scope
    # of things referenced here is based on the module(s) that "funcs" is from.
    minkasi_noise:
      class: "minkasi.mapmaking.noise.NoiseSmoothedSVD" # Noise class to use
      args: "[]" # Arguments to pass to apply_noise
      kwargs: "{'fwhm':10}" # kwargs to pass to apply_noise
    # Defines the beam
    # All are passed through eval
    # Note that these define a double gaussian
    beam:
      fwhm1: "9.735" # FWHM in arcseconds of the first gaussian
      amp1: 0.9808 # Amplitude of the first gaussian
      fwhm2: "32.627" # FWHM in arcseconds of the second gaussian
      amp2: 0.0192 # Amplitude of the second gaussian
    copy_noise: False # If true then fitting noise just wraps minkasi noise, may make this automatic later
    dograd: False # If True then use gradient priors when mapmaking
    npass: 1 # How many passes of mapmaking to run
```

Putting all of this together we have the following `base` configuration file:

```yaml
name: "RXJ1347" # The name of the cluster/project
fit: True # If True fit the cluster, overridden by command line
sub: True # If True the cluster before mapmaking, overridden by command line
n_rounds: 4 # How many rounds of fitting to try
sim: False # Set to True of we are running a simulation
paths:
    data: "RXJ1347" # Usually we will want the data in a directory
    outroot: "" # The default for this is an empty string
    subdir: "" # Same here
constants:
  Te: 5.0
  freq: "90e9"
  z: 0.451
imports:
  astropy.units: u
  astropy.coordinates: [Angle]
  witch.external.minkasi.funcs: mf
  jitkasi.noise: jn
coords:
  r_map: "3.0*60" # Radial size of grid in arcseconds
  dr: "1.0" # Pixel size of grid in x and y in arcseconds
  dz: 1.0 # Pixel size along the LOS, if not provided dr is used
  x0: "(206.8776*u.degree).to(u.radian).value" # RA of grid origin in radians
  y0: "(-11.7528*u.degree).to(u.radian).value" # Dec of grid origin in radians
# Setting for the Levenberg–Marquardt fitter
fitting:
  maxiter: 10 # Maximum fit iterations per round of fitting
  chitol: 1e-5 # Change in chisq that we consider to be converged
# Settings for HMC
mcmc:
  run: True # If this is false the chain will not be run
  num_steps: 1000 # The number of samples in the chain
  num_leaps: 10 # The number of iterations of the leapfrog algorithm to run at each step
  step_size: .02 # The step size to use in the chain
  sample_which: -2 # Which parameters to sample, -2 will sample any parameters that we have tried to fit
datasets:
  mustang2:
    # These fields need to be set regardless of the dataset
    # This can rely on the imports section
    noise: # Noise to use while fitting
      class: "jn.NoiseSmoothedSVD"
      args: "[]"
      kwargs: "{'fwhm':10}"
    funcs: # Functions needed to interact with your dataset
      # All of these need to exist and be importable
      # If you don't have them in a library add the relevant file to your $PYTHONPATH
      # Check docs for the needed signatures
      get_files: "mf.get_files"
      load_tods: "mf.load_tods"
      get_info: "mf.get_info"
      make_beam: "mf.make_beam"
      preproc: "mf.preproc"
      postproc: "mf.postproc"
      postfit: "mf.postfit"
    # The rest are user specified and will depend on your dataset
    # These should only ever be needed in the function set in "funcs" above
    # Since they are only called from the specified "funcs" make sure the scope
    # of things referenced here is based on the module(s) that "funcs" is from.
    minkasi_noise:
      class: "minkasi.mapmaking.noise.NoiseSmoothedSVD" # Noise class to use
      args: "[]" # Arguments to pass to apply_noise
      kwargs: "{'fwhm':10}" # kwargs to pass to apply_noise
    # Defines the beam
    # All are passed through eval
    # Note that these define a double gaussian
    beam:
      fwhm1: "9.735" # FWHM in arcseconds of the first gaussian
      amp1: 0.9808 # Amplitude of the first gaussian
      fwhm2: "32.627" # FWHM in arcseconds of the second gaussian
      amp2: 0.0192 # Amplitude of the second gaussian
    copy_noise: False # If true then fitting noise just wraps minkasi noise, may make this automatic later
    dograd: False # If True then use gradient priors when mapmaking
    npass: 1 # How many passes of mapmaking to run
```

This can be saved to a file somewhere, for conveniences sake lets assume the file is called `base.yaml`.
Now we are ready a write a downstream configuration file.

The most important field here is `base` since this tells `WITCH` where to find the base configuration file.
You can either provide an absolute path to the file or (often more conveniently) a path relative to the directory
that your downstream configuration is in.
In our example case here this field will simply be:

```yaml
base: base.yaml
```

Now we can overwrite anything from `base.yaml` in this configuration if we wish.

The main section you will usually want in each of your downstream configurations is one defining the model.
The general layout of the model section is as follows:

```yaml
model:
    unit_conversion: ... # A prefactor to apply to go from compton y to whatever your data is
    structures:
        structure1:
            structure: ... # The type of structure this is
            parameters:
                par1:
                    value: ... # The value of the parameter, if we are fitting it this is where we start
                    to_fit: ... # Whether or not to fit the parameter, this can be a single bool or a list with one value per round. False by default.
                    priors: ... # A two element list [low, high] of the bounds on the parameter, this is optional
                par2:
                    ...
        structure2:
            ...
```

To know what parameters a specific structure takes see [this documentation](https://mustang-sz.github.io/WITCH/latest/reference/structure/),
note that the names of the parameters in the configuration file do not need to match the function parameters but the order must be correct.

Below is an example model:

```yaml
model:
  unit_conversion: "float(wu.get_da(constants['z'])*wu.y2K_RJ(constants['freq'], constants['Te'])*wu.XMpc/wu.me)"
  structures:
    a10:
      structure: "a10"
      parameters:
        dx_1:
          value: 0.0
          to_fit: [True, True, False, True]
          priors: [-9.0, 9.0]
        dy_1:
          value: 0.0
          to_fit: [True, True, False, True]
          priors: [-9.0, 9.0]
        dz_1:
          value: 0.0
        theta:
          value: 0.0
        P0:
          value: 8.403
        c500:
          value: 1.177
        m500:
          value: "1.5e15"
          to_fit: True
        gamma:
          value: .3081
        alpha:
          value: 1.551
        beta:
          value: 5.4905
        z:
          value: 0.97
    ps_gauss:
      structure: "gaussian"
      parameters:
        dx_g:
          value: 0.0
          to_fit: [True, True, False, True]
          priors: [-9.0, 9.0]
        dy_g:
          value: 0.0
          to_fit: [True, True, False, True]
          priors: [-9.0, 9.0]
        sigma:
          value: 4
        amp_g:
          value: 0.002
          to_fit: [True, False, True, True]
```

A complete guide to how to think about and construct models is planned for the future but it is worth
briefly discussing the core ideas of how `WITCH` models things here.
The model is constructed in several stages:

* Stage -1: adds non parametric models to the grid. This is an advanced feature and not recommended for beginners.
* Stage 0: add 3d parametric models to the grid. This is typically where you would add cluster profiles (ie: A10s, isobetas, etc.).
* Stage 1: modify the 3d grid. This is where substructure like bubbles and shocks are added.
* At this point the model is integrated along the line of sight, reducing it to 2d.
* Stage 2: add 2d models to the model. This is can be used to add profiles that don't need to interact with cluster substructure.
* At this point the model is convolved with the beam.
* Stage 3: add 2d models to the model that we didn't want to beam convolved. This is really only used for adding point sources.

So lets say you wanted to model a merging cluster pair with a shock in the merger and a AGN in one,
you would do this by adding two cluster profiles at Stage 0, adding the shock at Stage 1, and then adding the AGN
at Stage 3.

## Running `WITCH`

Once you have your configuration file to run `WITCH` all you need to do is call the `witcher` command like so:

```
witcher PATH_TO_CONFIG.yaml
```

depending on your dataset the outputs will vary but the key things to look out for is the fit parameters after each
round of fitting. For example below is the output from fitting a simple gaussian:

```
ps_gauss:
Round 1 out of 1
        ps_gauss:
                dx_g* [-9.0, 9.0] = [-0.09016553] ± [0.12899712] ([0.69897324] σ)
                dy_g* [-9.0, 9.0] = [-0.14666429] ± [0.12809413] ([1.1449728] σ)
                sigma* [1.0, 10.0] = [4.03847427] ± [0.06408161] ([63.02080388] σ)
                amp_g [0.0, 0.005] = [0.002] ± [0.] ([inf] σ)
chisq is 3578656.971706165
````

After each round of fitting the model will be saved using [`dill`](https://pypi.org/project/dill/) and
can be loaded with [`Model.load`](https://mustang-sz.github.io/WITCH/latest/reference/containers/#witch.containers.Model.load),
the path to each file will be printed after each round. A `yaml` file with the fit model parameters will also be saved
along side the `dill` file. In the case of HMC, the full sample chain will be saved as a `npz` file in the output directory.
The output directory also contains a copy of the configuration file including all the fields from the `base` configuration.

The path to the output directory can sometimes be cumbersome as it includes the names of all fit parameters,
luckily the last thing printed by `witcher` is the path with all your outputs!
Any dataset specific outputs (ie: residual maps) will be in subdirectories labeled with the dataset name.
