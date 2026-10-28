#!/bin/bash
# Batch script for the full A399-A401 ACT+M2 joint fit (configs/A399_401/A399_401_joint_act_m2.yaml) on 4 Trillium nodes.
# Rank 0 is the ACT map and the other ranks split the 304 M2 TODs evenly.
# Both clusters are elliptical isobetas, so every rank builds the full 400x400x200 3D grid and its jacfwd.
# That cost is the same per rank whatever its TOD count, so fewer ranks with more threads beat 1 TOD per rank.
# Layout: 76 M2 ranks x 4 TODs + 1 ACT rank = 77 ranks, 20 per node, 9 threads each (180 of 192 cores).
# scripts/bench_joint_layout.sh (2026-10-28, one node packed like a production node) measured per LM eval:
#   8 TODs/rank x 19 thr 2.60 s, 4 x 9 2.62 s, 2 x 4 2.97 s, 1 x 2 3.90 s. 4 x 9 ties for fastest and has the
#   quickest map making (12 s vs 19 s). Memory ~5.8 GB/rank + 0.32 GB/TOD, 138 GB per node here.
#   Estimated fit: 0.1-0.2 h (7 LM iterations per round up to maxiter), so --time has a ~5x margin.
# To change the layout keep ntasks = 1 + 304 / TODs per rank and ntasks-per-node * cpus-per-task <= 192.
# Submit with: sbatch configs/A399_401/run_A399_401_joint_act_m2.sbatch
# For the MCMC afterwards add the `mcmc:` block of A399_401_ymap_isobeta_mesa_new_mcmc.yaml in a wrapper config.
#SBATCH --job-name=a399_401_joint
#SBATCH --account=def-rbond
#SBATCH --nodes=4
#SBATCH --ntasks=77
#SBATCH --ntasks-per-node=20
#SBATCH --cpus-per-task=9
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/jorlo/%x_%j.txt
#SBATCH --mail-type=FAIL

set -euo pipefail
module load StdEnv/2023 python/3.12.4 gcc/14.3 openmpi/5.0.8 mpi4py/4.1.0
source /home/jorlo/venvs/witch/bin/activate

# Trillium's sbatch uses --export=NONE, so set the roots here.
# WITCH_DATROOT has to contain bridge/ (the TODs) and A399_401/act_y/ (the ACT cutout and noise spectrum)
export WITCH_DATROOT=/scratch/jorlo/MUSTANG/M2-TODs
export WITCH_OUTROOT=/scratch/jorlo/MUSTANG/Reductions

# One thread pool per rank, sized to its cores
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export FLEXIBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=$SLURM_CPUS_PER_TASK"
export JAX_PLATFORMS=cpu

# Home is read-only on compute nodes
export MPLBACKEND=Agg MPLCONFIGDIR=/scratch/jorlo/.cache/matplotlib
mkdir -p "$MPLCONFIGDIR"

# Config from the first argument (sbatch doesn't pass the environment), eg the MCMC wrapper config
CFG=${1:-/home/jorlo/dev/WITCH/configs/A399_401/A399_401_joint_act_m2.yaml}
cd /scratch/jorlo

# block:cyclic keeps each rank's cores on one socket and alternates ranks between the sockets
srun --cpus-per-task="$SLURM_CPUS_PER_TASK" --distribution=block:cyclic --cpu-bind=cores \
    witcher "$CFG" --cpu True
