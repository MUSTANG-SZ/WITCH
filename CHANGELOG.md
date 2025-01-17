## 8.2.1 (2025-01-17)

### Fix

- fix using wrong kT for ultra relativistic

## 8.2.0 (2025-01-16)

### Feat

- re-adding supfactor

## 8.1.2 (2025-01-06)

### Fix

- small fixes to minkasi postfit

## 8.1.1 (2025-01-03)

### Fix

- make sure todvec isnt converted to minkasi todvec before processing

## 8.1.0 (2024-12-16)

### Feat

- adding 2D cylindrical beta:

## 8.0.0 (2024-12-16)

### Feat

- added tool for updating config files
- abstract away beam interface
- abstract away assumption that we are using an minkasi based dataset
- switch to backending everything in jax

### Fix

- cast priors to array
- jax array when doing wn sims
- don;t try to hash tracers

## 7.0.0 (2024-12-16)

### Feat

- update to match colormap changes
- changed colormap handling to be more sensible
- add model map
- recompute model when xyz is updated
- add model plotting to fitter
- removing bowling from configs
- added noise estimation. Also started removing bowling
- get_radial_mask util
- return maps from make_maps
- add units to doc strings
- allow no model in config
- save final fit pars in a seperate yaml
- adding act fitting and non-witch formating
- files for checking r500 concordance
- r500_elip now corresponds to r500_sph
- ability to turn off structure when sim fitting
- adding cylindrical beta profile
- switching to plot_cluster. Need to clean this workbook up more
- adding lims and pix_size to model
- base plotting function
- add example of 2nd stage plotting
- changing cos powerlaw to simpler powerlaw
- plotting models is now ez
- adding aplpy plotting notebook
- switching presets by source to minkasi
- solve for signal map before fitting
- cache model computation
- let users specify imports in config file
- wnoise in cfg
- allow base config to be relative
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
- Added centering to make_grid_from_skymap
- spilt creation of model from indexing it onto a tod
- modify tod pixilization to work on any grid
- add in function to get grid from skymap
- faster get_chis
- new gaussian modeling
- new streamlined sampler
- adding mcmc check
- working towards being able to fix parameters
- deved constructor function for sampler
- adding a10 profile
- adding fix_r500 functionality
- add helper to profiling and make things more realistic

### Fix

- cyl_unit has wrong structure name
- writting on multiple threads:
- bug where it wouldnt model mapmake
- update grid_from_wcs to match make_grid
- removing debug print statements
- bug where it always made noise
- correcting units pixel to arcsec
- applying black
- raise error when we don't have MPI and handle having more procs than TODs
- adjust cylindical beta name in ORDER
- making function name and id string agree for cylindrical_beta
- changing parse args
- fix misnamed setattr
- small typing issues
- updating colormap to matplotlib 3.9.0
- missed a config_path reference
- formating
- applied black formating
- adding markdown
- clear naive map after each iter
- updating a10 pars
- formating
- r500 now uses fit value
- output cfg no longer has base dependency
- floats in config
- gradient priors are a ts model so they only need to be in the mapset
- make sure parameter value is float
- better printing and error messages
- handle grid in absolute coords and apply cosdec properly
- many small bugs
- also allow kwargs
- more npars issues :)
- ensure that samples outside of grid are 0
- do inspections after function def
- jnp.trapz depracation
- undoing default
- only change variables that are being fit
- Dont start at the right value when simming
- make sure grid is floats and return the modified arrays
- use jax syntax for setting arrays
- various indexing bugs, also switch to ij indexing
- make mcmc check consistent with sampler branch
- several minor fixes, mostly from merge errors but also use seperate ARGNUM_SHIFTS for the two cases
- fixing merge issue
- include da in smapler call
- finally sorted sim issues and other sim improvements
- restoring jit
- updating to work with minkasi refactor
- do sparse grid creation even when using a skymap
- fixing dr/r_map units
- updating imports to conform to new minkasi standards
- don't double divide r500
- don't allow negetive indices
- fixing variable name
- making compatible with minkasi update and small dr fix
- ARGNUM_SHIFT somehow wrong again
- make ntod cut MPI aware rather than per proc

### Refactor

- many changes to utils:
- move bowling to scratch
- model function now calls structure functions dynamically, introduces the concepts of structure stages
- removed depracted tod functions
- streamlined fitter and made it a part of the core library
- move some inspections around and add variables to make working with things dynamically easier

### Perf

- typing idx/idy for robustness
- removing dx/dy seems to help fitting
- expicitly commit arrays to devices
- prevent unneeded mem transfers

## 6.5.0 (2024-11-11)

### Feat

- update to match colormap changes

## 6.4.0 (2024-11-11)

### Feat

- changed colormap handling to be more sensible

## 6.3.3 (2024-10-15)

### Fix

- cyl_unit has wrong structure name

## 6.3.2 (2024-10-15)

### Fix

- writting on multiple threads:

## 6.3.1 (2024-10-15)

### Fix

- bug where it wouldnt model mapmake

## 6.3.0 (2024-10-14)

### Feat

- add model map
- recompute model when xyz is updated
- add model plotting to fitter

### Fix

- update grid_from_wcs to match make_grid

## 6.2.1 (2024-10-09)

### Fix

- removing debug print statements

## 6.2.0 (2024-10-09)

### Feat

- removing bowling from configs
- added noise estimation. Also started removing bowling
- get_radial_mask util
- return maps from make_maps

### Fix

- bug where it always made noise

## 6.1.1 (2024-10-08)

### Fix

- correcting units pixel to arcsec

## 6.1.0 (2024-10-08)

### Feat

- add units to doc strings

### Fix

- applying black

## 6.0.4 (2024-10-02)

### Fix

- raise error when we don't have MPI and handle having more procs than TODs

## 6.0.3 (2024-10-01)

### Fix

- adjust cylindical beta name in ORDER

## 6.0.2 (2024-09-26)

### Fix

- making function name and id string agree for cylindrical_beta

## 6.0.1 (2024-09-24)

### Fix

- changing parse args

## 6.0.0 (2024-09-24)

### Refactor

- many changes to utils:

## 5.8.1 (2024-09-19)

### Fix

- fix misnamed setattr

## 5.8.0 (2024-08-14)

### Feat

- allow no model in config

## 5.7.0 (2024-07-30)

### Feat

- save final fit pars in a seperate yaml

### Fix

- small typing issues

## 5.6.2 (2024-07-24)

### Fix

- updating colormap to matplotlib 3.9.0

## 5.6.1 (2024-07-23)

### Fix

- missed a config_path reference

## 5.6.0 (2024-07-23)

### Feat

- adding act fitting and non-witch formating

## 5.5.0 (2024-07-18)

### Feat

- files for checking r500 concordance
- r500_elip now corresponds to r500_sph

## 5.4.0 (2024-07-17)

### Feat

- ability to turn off structure when sim fitting

### Fix

- formating

## 5.3.0 (2024-07-15)

### Feat

- adding cylindrical beta profile

### Fix

- applied black formating

## 5.2.0 (2024-07-12)

### Feat

- switching to plot_cluster. Need to clean this workbook up more
- adding lims and pix_size to model
- base plotting function

### Fix

- adding markdown
- clear naive map after each iter
- updating a10 pars
- formating
- r500 now uses fit value
- output cfg no longer has base dependency

## 5.1.0 (2024-07-11)

### Feat

- add example of 2nd stage plotting
- changing cos powerlaw to simpler powerlaw

## 5.0.0 (2024-07-09)

### Refactor

- move bowling to scratch

## 4.3.0 (2024-07-01)

### Feat

- plotting models is now ez

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
