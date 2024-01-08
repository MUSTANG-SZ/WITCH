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
