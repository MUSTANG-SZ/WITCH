#!/bin/bash
# Batch script for the full A399-A401 ACT+M2 joint fit (configs/A399_401/A399_401_joint_act_m2.yaml).
# One MPI task is the ACT map, the others split the 304 M2 TODs evenly. Each TOD needs ~1.5 GB, so ask for
# enough memory in total (~480 GB with some headroom), ie 2 nodes of 256 GB or more. Adjust the SLURM lines to the machine.
# Submit with: sbatch configs/A399_401/run_A399_401_joint_act_m2.sbatch
# For the MCMC afterwards add the `mcmc:` block of A399_401_ymap_isobeta_mesa_new_mcmc.yaml in a wrapper config.
#SBATCH --job-name=a399_401_joint
#SBATCH --nodes=2
#SBATCH --ntasks=65           # 1 ACT + 64 M2 tasks, ~5 TODs each
#SBATCH --cpus-per-task=2
#SBATCH --mem=0               # all the memory on each node
#SBATCH --time=06:00:00
#SBATCH --output=a399_401_joint_%j.log

set -euo pipefail
source ~/venvs/general/bin/activate
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
# WITCH_DATROOT has to contain bridge/ (the TODs) and A399_401/act_y/ (the ACT cutout and noise spectrum)
: "${WITCH_DATROOT:?set WITCH_DATROOT}"
: "${WITCH_OUTROOT:?set WITCH_OUTROOT, an unset one writes into HOME}"
CFG=${CFG:-configs/A399_401/A399_401_joint_act_m2.yaml}

srun --mpi=pmix witcher "$CFG" -cpu True
