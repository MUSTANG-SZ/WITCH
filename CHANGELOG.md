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
