"""Command-line driver for running and saving WITCH MCMC chains.

By default this tool evaluates one dataset and runs independent chains in
parallel. Use ``--joint`` to evaluate the MPI-aware joint likelihood and run
the chains serially.
"""

import argparse
import os
import warnings
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np

from witch.cfg_loader import load_cfg
from witch.fitter import comm, fit_loop, load_config
from witch.metropolis_hastings import run_chains_parallel, run_chains_serial


def sanitize_chains(
    chain_samples: np.ndarray,
    par_names: Sequence[str] | np.ndarray,
    fit_mask: Sequence[bool] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Make parameter names unique and identify immobile parameters.

    Duplicate parameter names are renamed with numeric suffixes, preserving
    the first occurrence. Fitted parameters with no movement in any chain are
    excluded from the returned plotting mask.

    Parameters
    ----------
    chain_samples : np.ndarray
        Samples with shape ``(num_chains, num_samples, num_parameters)``.
    par_names : Sequence[str] or np.ndarray
        Name for each parameter in the final axis of ``chain_samples``.
    fit_mask : Sequence[bool] or np.ndarray
        Boolean mask indicating which parameters are fitted. Must have one
        entry per parameter.

    Returns
    -------
    unique_names : np.ndarray
        Parameter names made unique where duplicates were present.
    plot_mask : np.ndarray
        Copy of ``fit_mask`` with parameters that have no movement in at least
        one chain set to ``False``.

    Warns
    -----
    UserWarning
        If parameter names are duplicated or fitted parameters have no movement.
        The no-movement warning identifies the affected parameters and chains.
    """
    unique_names = []
    used_names = set()
    occurrences = {}
    renamed = []
    for name in map(str, par_names):
        occurrences[name] = occurrences.get(name, 0) + 1
        suffix = occurrences[name]
        candidate = name if suffix == 1 else f"{name}_{suffix}"
        while candidate in used_names:
            suffix += 1
            candidate = f"{name}_{suffix}"
        used_names.add(candidate)
        unique_names.append(candidate)
        if candidate != name:
            renamed.append(f"{name} -> {candidate}")

    if renamed:
        warnings.warn(
            "Duplicate parameter names found; renamed duplicates: "
            + ", ".join(renamed),
            UserWarning,
            stacklevel=2,
        )

    plot_mask = np.asarray(fit_mask, dtype=bool).copy()
    immobile = []
    for parameter_index in np.flatnonzero(plot_mask):
        frozen_chains = np.flatnonzero(
            np.ptp(chain_samples[:, :, parameter_index], axis=1) == 0
        )
        if frozen_chains.size:
            plot_mask[parameter_index] = False
            chain_labels = ", ".join(f"Chain {index}" for index in frozen_chains)
            immobile.append(f"{unique_names[parameter_index]} ({chain_labels})")

    if immobile:
        warnings.warn(
            "Parameters with no movement were removed from the plot: "
            + "; ".join(immobile),
            UserWarning,
            stacklevel=2,
        )

    return np.asarray(unique_names), plot_mask


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
    par_names, plot_mask = sanitize_chains(chain_samples, metamodel.par_names, fit_mask)
    output_path = (
        args.output if os.path.isabs(args.output) else os.path.join(outdir, args.output)
    )
    np.savez_compressed(
        output_path,
        samples=chain_samples,
        acceptance_rates=np.asarray(acceptance_rates),
        par_names=par_names,
        to_fit=fit_mask,
    )
    print(f"Saved chains to {output_path}")

    consumer = ChainConsumer()
    if np.any(plot_mask):
        for chain_id, (chain_samples, _) in enumerate(chains):
            consumer.add_chain(
                Chain(
                    samples=pd.DataFrame(
                        chain_samples[:, plot_mask],
                        columns=par_names[plot_mask],
                    ),
                    name=f"RXJ1347 chain {chain_id}",
                )
            )
        consumer.plotter.plot()
        plot_path = os.path.splitext(output_path)[0] + ".pdf"
        plt.savefig(plot_path)
    else:
        warnings.warn(
            "No fitted parameters with movement remain; skipping the MCMC plot.",
            UserWarning,
            stacklevel=2,
        )
    print(f"Acceptance rates: {acceptance_rates}")


if __name__ == "__main__":
    main()
