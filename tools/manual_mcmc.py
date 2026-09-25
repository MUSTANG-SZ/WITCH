"""Command-line driver for running and saving WITCH MCMC chains.

By default this tool evaluates one dataset and runs independent chains in
parallel. Use ``--joint`` to evaluate the MPI-aware joint likelihood and run
the chains serially.
"""

import argparse
import os

import numpy as np

from witch.fitter import fit_loop, load_config, comm
from witch.cfg_loader import load_cfg
from witch.metropolis_hastings import run_chains_parallel, run_chains_serial

import matplotlib.pyplot as plt


def parse_args():
    """Parse command-line options for the MCMC run."""
    parser = argparse.ArgumentParser(description="Run independent MCMC chains.")
    parser.add_argument(
        "--config",
        help="Path to the WITCH configuration file.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10000,
        help="Number of samples per chain.",
    )
    parser.add_argument(
        "--num-chains",
        type=int,
        default=4,
        help="Number of independent MCMC chains to run.",
    )
    parser.add_argument(
        "--dataset-index",
        type=int,
        default=0,
        help="Dataset index for the parallel local-objective mode.",
    )
    parser.add_argument(
        "--joint",
        action="store_true",
        help="Use the MPI-aware joint likelihood and run chains serially.",
    )
    parser.add_argument(
        "--bound",
        type=float,
        default=2,
        help="Factor used to replace unbounded priors. Used with flat priors only.",
    )
    parser.add_argument(
        "--prior-type",
        choices=("uniform", "normal"),
        default="uniform",
        help="Sampling prior: uniform bounds or normal errors from the metamodel.",
    )
    parser.add_argument(
        "--prior-err-scale",
        type=float,
        default=1.0,
        help="Scale factor to multiply gaussian prior errors (and normal proposal widths).",
    )
    parser.add_argument(
        "--output",
        default="mcmc_chains.npz",
        help="Output filename relative to WITCH outdir, or an absolute path.",
    )
    return parser.parse_args()


def main():
    """Load, fit, sample, save, and plot the configured WITCH model."""
    import pandas as pd
    from chainconsumer import Chain, ChainConsumer

    args = parse_args()
    cfg = load_config({}, args.config)
    _, outdir, _, metamodel = load_cfg(cfg)

    # First fit the data to get a reasonable starting point. Saves a lot of burn in.
    metamodel = fit_loop(metamodel, cfg, comm)

    # Split behavior between running chains serially if we are doing the joint likelihood across
    # different datasets (i.e. M2+X-ray) or parallel if doing only one data set (i.e. just M2).
    # The comm.reduce in joint_objective get in the way of doing parallel chains.
    if args.joint:
        chains = run_chains_serial(
            metamodel,
            num_samples=args.num_samples,
            num_chains=args.num_chains,
            bound=args.bound,
            prior_type=args.prior_type,
            prior_err_scale=args.prior_err_scale,
        )
    else:
        chains = run_chains_parallel(
            metamodel,
            num_samples=args.num_samples,
            num_chains=args.num_chains,
            bound=args.bound,
            dataset_ind=args.dataset_index,
            prior_type=args.prior_type,
            prior_err_scale=args.prior_err_scale,
        )
    # Get the acceptance_rates from the chains
    acceptance_rates = [acceptance_rate for _, acceptance_rate in chains]
    # Filter out chains to only look at fit parameters.
    fit_mask = np.asarray(metamodel.to_fit)
    chain_samples = np.stack([chain_samples for chain_samples, _ in chains])
    output_path = (
        args.output if os.path.isabs(args.output) else os.path.join(outdir, args.output)
    )
    np.savez_compressed(
        output_path,
        samples=chain_samples,
        acceptance_rates=np.asarray(acceptance_rates),
        par_names=np.asarray(metamodel.par_names),
        to_fit=fit_mask,
    )
    print(f"Saved chains to {output_path}")

    consumer = ChainConsumer()
    for chain_id, (chain_samples, _) in enumerate(chains):
        consumer.add_chain(
            Chain(
                samples=pd.DataFrame(
                    chain_samples[:, fit_mask],
                    columns=np.asarray(metamodel.par_names)[fit_mask],
                ),
                name=f"RXJ1347 chain {chain_id}",
            )
        )
    consumer.plotter.plot()
    plt.savefig(output_path+"mcmc.pdf")
    print(f"Acceptance rates: {acceptance_rates}")


if __name__ == "__main__":
    main()
