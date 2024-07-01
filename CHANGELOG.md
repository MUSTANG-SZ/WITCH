## 4.2.0 (2024-06-17)

### Feat

- adding aplpy plotting notebook

## 4.1.2 (2024-06-14)

### Fix

- floats in config

## 4.1.1 (2024-06-13)

### Fix

- gradient priors are a ts model so they only need to be in the mapset
- make sure parameter value is float

## 4.1.0 (2024-06-12)

### Feat

- switching presets by source to minkasi

## 4.0.0 (2024-06-10)

### Feat

- solve for signal map before fitting
- cache model computation

### Refactor

- model function now calls structure functions dynamically, introduces the concepts of structure stages
- removed depracted tod functions

## 3.3.0 (2024-06-10)

### Feat

- let users specify imports in config file

### Fix

- better printing and error messages

## 3.2.1 (2024-06-09)

### Fix

- handle grid in absolute coords and apply cosdec properly

## 3.2.0 (2024-06-04)

### Feat

- wnoise in cfg

## 3.1.0 (2024-05-29)

### Feat

- allow base config to be relative

## 3.0.0 (2024-05-29)

### Feat

- recursive configs and relative paths
- set coord units
- add serialization of model dclass
- repr for model
- add some convenience classes in prep for new fitter script
- switch to bilinear interp
- added unit tests
- adding itterative fitting
- print starting params for sims. Also switching to sci notation for params
- adding white noise option for sims
- do static argnums and argnum shift via function inspection
- adding no sub functionality
- adding downsampling to fitter and small parameter changes

### Fix

- many small bugs
- also allow kwargs
- more npars issues :)
- ensure that samples outside of grid are 0
- do inspections after function def
- jnp.trapz depracation
- undoing default
- only change variables that are being fit
- Dont start at the right value when simming

### Refactor

- streamlined fitter and made it a part of the core library
- move some inspections around and add variables to make working with things dynamically easier

## 2.4.0 (2024-05-27)

### Feat

- switch to bilinear interp

## 2.3.1 (2024-03-14)

### Fix

- ensure that samples outside of grid are 0

## 2.3.0 (2024-03-08)

### Feat

- print starting params for sims. Also switching to sci notation for params

## 2.2.1 (2024-03-08)

### Fix

- do inspections after function def

## 2.2.0 (2024-03-07)

### Feat

- do static argnums and argnum shift via function inspection

### Fix

- jnp.trapz depracation

## 2.1.0 (2024-01-22)

### Feat

- Added centering to make_grid_from_skymap

## 2.0.4 (2024-01-21)

### Fix

- make sure grid is floats and return the modified arrays

## 2.0.3 (2024-01-21)

### Fix

- use jax syntax for setting arrays

## 2.0.2 (2024-01-19)

### Fix

- various indexing bugs, also switch to ij indexing

## 2.0.1 (2024-01-15)

### Fix

- make mcmc check consistent with sampler branch
- several minor fixes, mostly from merge errors but also use seperate ARGNUM_SHIFTS for the two cases

## 2.0.0 (2024-01-15)

### Feat

- spilt creation of model from indexing it onto a tod
- modify tod pixilization to work on any grid
- add in function to get grid from skymap
- faster get_chis
- new gaussian modeling
- new streamlined sampler
- adding mcmc check
- working towards being able to fix parameters
- deved constructor function for sampler

### Fix

- fixing merge issue
- include da in smapler call
- finally sorted sim issues and other sim improvements
- restoring jit
- updating to work with minkasi refactor
- do sparse grid creation even when using a skymap
- fixing dr/r_map units
- updating imports to conform to new minkasi standards

### Perf

- typing idx/idy for robustness
- removing dx/dy seems to help fitting

## 1.5.0 (2024-01-08)

### Feat

- adding a10 profile
- adding fix_r500 functionality

### Fix

- don't double divide r500
- don't allow negetive indices
- fixing variable name
- making compatible with minkasi update and small dr fix

## 1.4.0 (2023-10-16)

### Feat

- add helper to profiling and make things more realistic

### Fix

- ARGNUM_SHIFT somehow wrong again

### Perf

- expicitly commit arrays to devices
- prevent unneeded mem transfers

## 1.3.1 (2023-09-26)

### Fix

- make ntod cut MPI aware rather than per proc
