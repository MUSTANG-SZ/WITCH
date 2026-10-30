"""
Compare fits of the same data with KS tests on their posteriors and the AIC.

The AIC uses the `aic.yaml` each fit writes (`witch.fitter.aic_refit`): the maximum likelihood
under one noise model shared by every fit of the same data, the one estimated from the raw
data before any model subtraction, so the chi-squared of different models are comparable.
Runs that predate it can be given one with `scripts/aic_from_run.py`.

Two levels of checks guard the comparisons:
* comparable (`check_comparable`, needed for the AIC): same data and likelihood configuration,
  same structures and parameter ties, and parameters fixed in both runs have the same value.
  Raises `ComparabilityError`.
* nested (`check_nested`, needed for the KS tests): comparable, plus one run's free parameters
  are a strict subset of the other's and its fixed values lie inside the other's priors.
  Raises `NestingError` (a `ComparabilityError`).
`ChainQualityError` is raised when a chain is too short or too correlated for a KS test.
"""

import itertools
import os
from dataclasses import dataclass, field

import numpy as np
import yaml
from scipy import stats

from .containers import MetaModel

# Config keys that only set where outputs go, these can differ between nested runs
OUTPUT_PATH_KEYS = ("subdir", "outroot")
# Config sections that define the data and the likelihood, these must match exactly
DATA_KEYS = ("name", "sim", "wnoise", "coords", "constants", "datasets")


class ComparabilityError(ValueError):
    """Raised when two runs are not fits of the same model family to the same data."""


class NestingError(ComparabilityError):
    """Raised when two runs are not nested models of the same data."""


class ChainQualityError(ValueError):
    """Raised when an MCMC chain can't support a KS test."""


@dataclass
class Run:
    """
    A finished WITCH run loaded from its output directory.

    Attributes
    ----------
    label : str
        Short name for the run.
    outdir : str
        The output directory of the run.
    cfg : dict
        The fully merged config the run used (`config.yaml` in `outdir`).
    names : list[str]
        Fully qualified parameter names (`model.structure.parameter`) in `MetaModel.parameters` order.
        Tied parameters are named after their first occurrence.
    structures : dict[str, str]
        Map of `model.structure` to the structure type.
    values : np.ndarray
        Parameter values the run was set up with, fixed parameters keep these.
    priors : np.ndarray
        `(npar, 2)` prior bounds.
    free : np.ndarray
        Boolean mask of the parameters sampled by the MCMC.
    chains : np.ndarray
        `(nchain, nstep, nfree)` post burn-in MCMC samples of the free parameters.
    parameter_map : tuple
        The `MetaModel.parameter_map` of the run.
    aic : dict | None
        Contents of the run's `aic.yaml` (see `witch.fitter.aic_refit`), None if it has none.
    """

    label: str
    outdir: str
    cfg: dict
    names: list[str]
    structures: dict[str, str]
    values: np.ndarray
    priors: np.ndarray
    free: np.ndarray
    chains: np.ndarray
    parameter_map: tuple = field(repr=False)
    aic: dict | None = field(default=None, repr=False)

    @property
    def k(self) -> int:
        """Number of free parameters."""
        return int(self.free.sum())

    @property
    def free_names(self) -> list[str]:
        """Names of the free parameters, in the order of the chain columns."""
        return [n for n, f in zip(self.names, self.free) if f]

    @property
    def medians(self) -> np.ndarray:
        """Full parameter vector with the free parameters at their posterior medians."""
        pars = self.values.copy()
        pars[self.free] = np.median(self.chains.reshape(-1, self.k), axis=0)
        return pars


def full_par_names(metamodel: MetaModel) -> tuple[list[str], dict[str, str]]:
    """
    Get fully qualified parameter names and structure types of a metamodel.

    Parameters
    ----------
    metamodel : MetaModel
        The metamodel to name the parameters of.

    Returns
    -------
    names : list[str]
        `model.structure.parameter` for each entry of `metamodel.parameters`.
        Tied parameters are named after the first model that holds them.
    structures : dict[str, str]
        Map of `model.structure` to the structure type, ie: `gnfw_rs`.
    """
    names = [""] * len(metamodel.parameters)
    structures = {}
    for model, par_map in zip(metamodel.models, metamodel.parameter_map):
        full = [
            f"{model.name}.{struct}.{par}"
            for struct, par in zip(model.par_struct_names, model.par_names)
        ]
        for idx, name in zip(par_map, full):
            names[int(idx)] = names[int(idx)] or name
        for structure in model.structures:
            structures[f"{model.name}.{structure.name}"] = structure.structure
    return names, structures


def load_run(
    outdir: str, label: str | None = None, burn_in: float | None = None
) -> Run:
    """
    Load a finished MCMC run from its output directory.

    Parameters
    ----------
    outdir : str
        Output directory containing `config.yaml`, `results_mcmc.dill`, and `samples_mcmc.npz`.
    label : str | None, default: None
        Short name for the run, defaults to the last two path components.
    burn_in : float | None, default: None
        Fraction of each chain to drop, defaults to `mcmc.burn_in` from the config (or 0.2).

    Returns
    -------
    run : Run
        The loaded run.

    Raises
    ------
    FileNotFoundError
        If any of the needed files are missing.
    ChainQualityError
        If the saved chains don't line up with the saved metamodel.
    """
    for fname in ("config.yaml", "results_mcmc.dill", "samples_mcmc.npz"):
        if not os.path.isfile(os.path.join(outdir, fname)):
            raise FileNotFoundError(f"{fname} not found in {outdir}, is the MCMC done?")
    with open(os.path.join(outdir, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    metamodel, _ = MetaModel.load(os.path.join(outdir, "results_mcmc.dill"))
    names, structures = full_par_names(metamodel)

    npz = np.load(os.path.join(outdir, "samples_mcmc.npz"))
    free = np.asarray(npz["to_fit"], dtype=bool)
    if "chains" not in npz.files or len(free) != len(names):
        raise ChainQualityError(
            f"{outdir}: samples_mcmc.npz has no per-chain samples or doesn't match the metamodel"
        )
    if list(npz["par_names"]) != list(metamodel.par_names):
        raise ChainQualityError(
            f"{outdir}: chain parameter order differs from the metamodel"
        )
    if burn_in is None:
        burn_in = cfg.get("mcmc", {}).get("burn_in", 0.2)
    chains = npz["chains"]
    chains = chains[:, int(burn_in * chains.shape[1]) :, free]
    aic = None
    if os.path.isfile(os.path.join(outdir, "aic.yaml")):
        with open(os.path.join(outdir, "aic.yaml")) as f:
            aic = yaml.safe_load(f)

    return Run(
        label or os.path.join(*outdir.rstrip("/").split("/")[-2:]),
        outdir,
        cfg,
        names,
        structures,
        np.asarray(metamodel.parameters, dtype=float),
        np.column_stack([np.asarray(p, dtype=float) for p in metamodel.priors]),
        free,
        chains,
        metamodel.parameter_map,
        aic,
    )


def data_signature(cfg: dict) -> dict:
    """
    The parts of a config that define the data and likelihood.

    Parameters
    ----------
    cfg : dict
        A merged WITCH config.

    Returns
    -------
    signature : dict
        Config subset that has to be identical for two runs to be comparable.
    """
    sig = {key: cfg.get(key) for key in DATA_KEYS}
    sig["paths"] = {
        k: v for k, v in cfg.get("paths", {}).items() if k not in OUTPUT_PATH_KEYS
    }
    return sig


def _dict_diff(a, b, prefix: str = "") -> list[str]:
    """
    List the keys where two nested dicts differ.

    Parameters
    ----------
    a, b : Any
        Values to compare.
    prefix : str, default: ""
        Key path of `a` and `b`.

    Returns
    -------
    diffs : list[str]
        Dotted key paths that differ.
    """
    if isinstance(a, dict) and isinstance(b, dict):
        return [
            d
            for key in sorted(set(a) | set(b), key=str)
            for d in _dict_diff(a.get(key), b.get(key), f"{prefix}{key}.")
        ]
    return [] if a == b else [prefix.rstrip(".")]


def check_comparable(a: Run, b: Run, rtol: float = 1e-10):
    """
    Check that two runs fit the same data with the same family of models.

    This is what the AIC needs: same data and likelihood configuration, the same model
    structures and parameter ties, and parameters fixed in both runs held at the same value.
    The runs don't have to be nested, ie: one can free alpha and the other gamma.

    Parameters
    ----------
    a, b : Run
        The runs to compare.
    rtol : float, default: 1e-10
        Relative tolerance when comparing fixed values.

    Raises
    ------
    ComparabilityError
        If any of the conditions above are not met.
    """
    pair = f"{a.label} and {b.label}"
    diffs = _dict_diff(data_signature(a.cfg), data_signature(b.cfg))
    if diffs:
        raise ComparabilityError(f"{pair}: data/likelihood config differs at {diffs}")
    if a.structures != b.structures or a.names != b.names:
        raise ComparabilityError(f"{pair}: model structures or parameters differ")
    if a.parameter_map != b.parameter_map:
        raise ComparabilityError(f"{pair}: parameter ties differ")
    both_fixed = ~a.free & ~b.free
    bad = ~np.isclose(a.values, b.values, rtol=rtol, atol=0) & both_fixed
    if bad.any():
        raise ComparabilityError(
            f"{pair}: fixed values differ for {list(np.array(a.names)[bad])}"
        )


def check_nested(small: Run, big: Run, rtol: float = 1e-10) -> list[str]:
    """
    Check that `small` is nested in `big`.

    Nested here means: comparable (see `check_comparable`), every parameter free in `small`
    is free in `big`, `big` has at least one more free parameter, and every parameter that
    `big` frees but `small` fixes is held at a value inside `big`'s prior so `small` is a
    point in `big`'s parameter space.

    Parameters
    ----------
    small : Run
        The run with fewer free parameters.
    big : Run
        The run with more free parameters.
    rtol : float, default: 1e-10
        Relative tolerance when comparing fixed values.

    Returns
    -------
    extra : list[str]
        The parameters free in `big` but fixed in `small`.

    Raises
    ------
    NestingError
        If any of the conditions above are not met.
    """
    pair = f"{small.label} in {big.label}"
    try:
        check_comparable(small, big, rtol)
    except NestingError:
        raise
    except ComparabilityError as e:
        raise NestingError(str(e)) from e

    not_free = [n for n, s, b in zip(big.names, small.free, big.free) if s and not b]
    if not_free:
        raise NestingError(f"{pair}: {not_free} are free in the smaller model only")
    extra = [n for n, s, b in zip(big.names, small.free, big.free) if b and not s]
    if not extra:
        raise NestingError(f"{pair}: both runs free the same parameters")

    only_big = big.free & ~small.free
    lo, hi = big.priors[:, 0], big.priors[:, 1]
    outside = only_big & ((small.values < lo) | (small.values > hi))
    if outside.any():
        raise NestingError(
            f"{pair}: fixed values of {list(np.array(big.names)[outside])} are outside the larger model's priors"
        )
    return extra


def order_nested(runs: list[Run]) -> list[Run]:
    """
    Sort runs by number of free parameters and check every pair is nested.

    Parameters
    ----------
    runs : list[Run]
        Runs to compare.

    Returns
    -------
    ordered : list[Run]
        The runs from fewest to most free parameters.

    Raises
    ------
    NestingError
        If any pair of runs is not nested.
    """
    ordered = sorted(runs, key=lambda run: run.k)
    for small, big in itertools.combinations(ordered, 2):
        check_nested(small, big)
    return ordered


def is_nested(small: Run, big: Run) -> bool:
    """
    Whether `small` is nested in `big`, see `check_nested`.

    Parameters
    ----------
    small, big : Run
        The runs to check.

    Returns
    -------
    nested : bool
        True if `small` is nested in `big`.
    """
    try:
        check_nested(small, big)
    except NestingError:
        return False
    return True


def sort_comparable(runs: list[Run]) -> list[Run]:
    """
    Sort runs by number of free parameters and check every pair is comparable.

    Parameters
    ----------
    runs : list[Run]
        Runs to compare.

    Returns
    -------
    ordered : list[Run]
        The runs from fewest to most free parameters.

    Raises
    ------
    ComparabilityError
        If any pair of runs is not comparable.
    """
    ordered = sorted(runs, key=lambda run: run.k)
    for a, b in itertools.combinations(ordered, 2):
        check_comparable(a, b)
    return ordered


def nested_pairs(runs: list[Run]) -> list[tuple[Run, Run]]:
    """
    All `(small, big)` pairs of runs where `small` is nested in `big`.

    Parameters
    ----------
    runs : list[Run]
        Runs to search.

    Returns
    -------
    pairs : list[tuple[Run, Run]]
        Nested pairs, ordered by the number of free parameters of `small` then `big`.
    """
    ordered = sorted(runs, key=lambda run: run.k)
    return [(a, b) for a, b in itertools.combinations(ordered, 2) if is_nested(a, b)]


def ess(chains: np.ndarray) -> float:
    """
    Effective sample size of one parameter, summed over chains.
    Uses Geyer's initial positive sequence estimate of the autocorrelation time.

    Parameters
    ----------
    chains : np.ndarray
        `(nchain, nstep)` samples.

    Returns
    -------
    ess : float
        Effective number of independent samples.
    """
    total = 0.0
    for chain in chains:
        x = chain - chain.mean()
        n = len(x)
        f = np.fft.rfft(x, 2 * n)
        acf = np.fft.irfft(f * np.conj(f))[:n]
        if acf[0] == 0:
            continue
        acf /= acf[0]
        tau = 1.0
        for lag in range(1, n - 1, 2):
            pair = acf[lag] + acf[lag + 1]
            if pair < 0:
                break
            tau += 2 * pair
        total += n / tau
    return total


def rhat(chains: np.ndarray) -> float:
    """
    Split R-hat (Gelman-Rubin) of one parameter.
    Each chain is split in half so drifts within a chain show up as well as disagreement between chains.
    Values above ~1.01-1.1 mean the chains have not converged to a common distribution.

    Parameters
    ----------
    chains : np.ndarray
        `(nchain, nstep)` samples.

    Returns
    -------
    rhat : float
        The split R-hat.
    """
    half = chains.shape[1] // 2
    split = np.concatenate([chains[:, :half], chains[:, half : 2 * half]])
    within = split.var(axis=1, ddof=1).mean()
    between = half * split.mean(axis=1).var(ddof=1)
    return float(np.sqrt(((half - 1) / half * within + between / half) / within))


def chain_diagnostics(run: Run) -> list[dict]:
    """
    Convergence diagnostics of each free parameter of a run.

    Parameters
    ----------
    run : Run
        The run to check.

    Returns
    -------
    rows : list[dict]
        Per parameter effective sample size, autocorrelation time in steps
        (total samples / ESS), and split R-hat.
    """
    nsamp = run.chains.shape[0] * run.chains.shape[1]
    rows = []
    for i, name in enumerate(run.free_names):
        n_eff = ess(run.chains[..., i])
        rows.append(
            {
                "run": run.label,
                "parameter": name,
                "ESS": n_eff,
                "tau": nsamp / n_eff,
                "Rhat": rhat(run.chains[..., i]),
            }
        )
    return rows


def ks_compare(small: Run, big: Run, min_ess: float = 50.0) -> list[dict]:
    """
    Two-sample KS test of the shared free parameters of two nested runs.
    The p-value uses the effective sample sizes, since MCMC samples are correlated
    and the nominal KS p-value would be wildly overconfident.

    Parameters
    ----------
    small : Run
        The nested (smaller) run.
    big : Run
        The larger run.
    min_ess : float, default: 50.0
        Minimum effective sample size per parameter for the test to be meaningful.

    Returns
    -------
    rows : list[dict]
        One row per shared free parameter with the medians, KS statistic `D`,
        the effective sample sizes, and the ESS corrected p-value.

    Raises
    ------
    NestingError
        If the runs are not nested.
    ChainQualityError
        If any shared parameter has fewer than `min_ess` effective samples.
    """
    check_nested(small, big)
    rows = []
    for name in small.free_names:
        xa = small.chains[..., small.free_names.index(name)]
        xb = big.chains[..., big.free_names.index(name)]
        na, nb = ess(xa), ess(xb)
        if min(na, nb) < min_ess:
            raise ChainQualityError(
                f"{name}: ESS {na:.0f} ({small.label}) / {nb:.0f} ({big.label}) is below {min_ess}"
            )
        d = stats.ks_2samp(xa.ravel(), xb.ravel()).statistic
        rows.append(
            {
                "parameter": name,
                f"median {small.label}": np.median(xa),
                f"median {big.label}": np.median(xb),
                "D": d,
                f"ESS {small.label}": na,
                f"ESS {big.label}": nb,
                "p": stats.kstwobign.sf(d * np.sqrt(na * nb / (na + nb))),
            }
        )
    return rows


def aic_chisqs(runs: list[Run], which: str = "refit") -> list[float]:
    """
    The chi-squared of each run from its `aic.yaml`.

    Parameters
    ----------
    runs : list[Run]
        Runs loaded with `load_run`.
    which : str, default: "refit"
        "refit" for the maximum likelihood (what the AIC is defined at) or
        "start" for the starting point (MCMC medians, or the LM best fit).

    Returns
    -------
    chisqs : list[float]
        Chi-squared of each run, in the order of `runs`.

    Raises
    ------
    FileNotFoundError
        If a run has no `aic.yaml`.
    ComparabilityError
        If an `aic.yaml` refit different parameters than the run's chain sampled.
    """
    missing = [run.outdir for run in runs if run.aic is None]
    if missing:
        raise FileNotFoundError(
            f"No aic.yaml in {missing}, make one with scripts/aic_from_run.py"
        )
    for run in runs:
        refit = [par["index"] for par in run.aic["parameters"]]
        if run.aic["k"] != run.k or refit != list(np.flatnonzero(run.free)):
            raise ComparabilityError(
                f"{run.label}: aic.yaml refit parameters {refit} but the chain samples "
                f"{list(np.flatnonzero(run.free))}"
            )
    return [float(run.aic[f"chisq_{which}"]) for run in runs]


def aic_table(runs: list[Run], chisqs: list[float]) -> list[dict]:
    """
    AIC of each run, `chisq + 2k`, with the difference to the best run and Akaike weights.

    Parameters
    ----------
    runs : list[Run]
        Comparable runs, checked with `sort_comparable`. They don't need to be nested.
    chisqs : list[float]
        Chi-squared of each run against the same data and noise,
        ideally at each run's maximum likelihood (`aic_chisqs(runs, "refit")`).

    Returns
    -------
    rows : list[dict]
        One row per run, in the order of `runs`.

    Raises
    ------
    ComparabilityError
        If the runs are not comparable.
    """
    sort_comparable(runs)
    aic = np.array([c + 2 * run.k for run, c in zip(runs, chisqs)])
    delta = aic - aic.min()
    weights = np.exp(-delta / 2) / np.exp(-delta / 2).sum()
    return [
        {"run": run.label, "k": run.k, "chisq": c, "AIC": a, "dAIC": d, "weight": w}
        for run, c, a, d, w in zip(runs, chisqs, aic, delta, weights)
    ]


def chisq_order_warnings(runs: list[Run], chisqs: list[float]) -> list[str]:
    """
    Flag nested pairs where the larger model has the higher chi-squared, non-nested pairs are skipped.
    At the maximum likelihood this can't happen, at the posterior medians it can
    if a posterior is skewed or a chain is poorly converged, so it is worth a look.

    Parameters
    ----------
    runs : list[Run]
        Runs in the same order as `chisqs`.
    chisqs : list[float]
        Chi-squared of each run.

    Returns
    -------
    warnings : list[str]
        One message per offending pair.
    """
    chisq = {run.label: c for run, c in zip(runs, chisqs)}
    return [
        f"{s.label} (k={s.k}) has chisq {chisq[s.label]:.2f} <= {b.label} (k={b.k}) at {chisq[b.label]:.2f}"
        for s, b in nested_pairs(runs)
        if chisq[b.label] > chisq[s.label]
    ]
