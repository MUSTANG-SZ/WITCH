#!/bin/bash
# Per dataset posterior predictive chi-squared and Q_DMAP for a joint fit and a fit of one of its datasets alone,
# see scripts/joint_consistency.py. Results go to /scratch/jorlo/joint_consistency_<jobid>.yaml (+ _draws.npz), read by
# notebooks/joint_consistency.ipynb.
# Usage: sbatch scripts/joint_consistency.sbatch [<joint outdir> <M2 only outdir>] [script options, e.g. --noise initial]
# Defaults to the MOOJ1142 joint and M2 only fits on the 10' grid. The run directories are given before any options.
# Same layout as the joint fits: 107 ranks = 1 per M2 TOD + 1 for the ACT map.
#SBATCH --job-name=witch_joint_consistency
#SBATCH --account=def-rbond
#SBATCH --nodes=3
#SBATCH --ntasks=107
#SBATCH --ntasks-per-node=36
#SBATCH --cpus-per-task=5
#SBATCH --time=00:30:00               # ~6 min for 3 noise models; short so it backfills
#SBATCH --output=/scratch/jorlo/%x_%j.txt
#SBATCH --mail-type=FAIL,END
#SBATCH --exclude=tri[1096-1097]

source /cvmfs/soft.computecanada.ca/config/profile/bash.sh 2>/dev/null
module load python/3.12.4
module load StdEnv/2023 gcc/14.3 openmpi/5.0.8
module load mpi4py/4.1.0
source /home/jorlo/venvs/witch/bin/activate

# Trillium's sbatch uses --export=NONE, so set the data root explicitly
export WITCH_DATROOT=/scratch/jorlo/MUSTANG/M2-TODs

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export FLEXIBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=$SLURM_CPUS_PER_TASK"
export JAX_PLATFORMS=cpu
export MPLBACKEND=Agg
export MPLCONFIGDIR=/scratch/jorlo/.cache/matplotlib
export JAX_COMPILATION_CACHE_DIR=/scratch/jorlo/.cache/jax
mkdir -p "$MPLCONFIGDIR" "$JAX_COMPILATION_CACHE_DIR"

R=/scratch/jorlo/MUSTANG/Reductions/MOOJ1142
JOINT_DEFAULT=$R/mustang2+act/cluster_model_sph_isobeta-gnfw_rs+ps_model_ps_gauss+ps_act_ps_gauss/joint_mcmc_rmap10/r-beta-amp-P0-r_s-beta-dx_g-dy_g-sigma-amp_g-amp_g
M2_DEFAULT=$R/mustang2/cluster_model_sph_isobeta-gnfw_rs+ps_model_ps_gauss/xray_peak_rs_control_rmap10/r-beta-amp-P0-r_s-beta-dx_g-dy_g-sigma-amp_g
if [ $# -ge 2 ] && [ "${1#-}" = "$1" ]; then
    JOINT=$1
    M2=$2
    shift 2
else
    JOINT=$JOINT_DEFAULT
    M2=$M2_DEFAULT
fi

cd /scratch/jorlo
# --cpu-bind=none: binding ranks to cores can deadlock XLA's CPU runtime
srun --cpu-bind=none python /home/jorlo/dev/WITCH/scripts/joint_consistency.py "$JOINT" "$M2" \
    --out "/scratch/jorlo/joint_consistency_${SLURM_JOB_ID}.yaml" "$@"
