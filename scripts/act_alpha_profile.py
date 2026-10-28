"""
Profile the ACT-only chi-squared over the gNFW outer slope `alpha`.

For each fixed `alpha` the other free parameters (m500 and the point source amplitude)
are refit with `witcher`. A flat profile means the map cannot constrain alpha.

Usage
-----
    python scripts/act_alpha_profile.py configs/MOOJ1142_ACT/MOOJ1142_ACT_gnfw.yaml
    python scripts/act_alpha_profile.py unit_tests/act/act_sim_cmb.yaml --debug
"""

import argparse
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def fit_fixed_alpha(config: Path, alpha: float, n_rounds: int, out: Path) -> tuple:
    """
    Fit with `alpha` fixed and return the final chi-squared and fitted parameters.

    Parameters
    ----------
    config : Path
        ACT-only config with a gnfw structure in `cluster_model`.
    alpha : float
        Value to fix `alpha` to.
    n_rounds : int
        Number of LM rounds.
    out : Path
        Directory for the wrapper config and outputs.

    Returns
    -------
    chisq : float
        Final chi-squared.
    m500 : float
        Fitted m500.
    amp : float
        Fitted point source amplitude.
    """
    out.mkdir(parents=True, exist_ok=True)
    wrapper = out / "wrapper.yaml"
    wrapper.write_text(
        f'base: "{config.resolve()}"\nn_rounds: {n_rounds}\nmap: False\nmodel_map: False\n'
        "fitting:\n  maxiter: 50\n  chitol: 1e-2\n"
        "cluster_model:\n  structures:\n    gnfw:\n      parameters:\n"
        f"        alpha:\n          value: {alpha}\n          to_fit: False\n"
    )
    env = os.environ | {"WITCH_OUTROOT": str(out / "output")}
    res = subprocess.run(
        [shutil.which("witcher"), str(wrapper)],
        capture_output=True,
        env=env,
        text=True,
        check=False,
    )
    if res.returncode:
        raise RuntimeError(res.stderr[-1500:])
    chisq = float(re.findall(r"chisq is ([\d.eE+-]+)", res.stdout)[-1])
    m500 = float(re.findall(r"m500\* .*= \[([\d.eE+-]+)\]", res.stdout)[-1])
    amp = float(re.findall(r"amp_act\* .*= \[([\d.eE+-]+)\]", res.stdout)[-1])
    return chisq, m500, amp


def main() -> None:
    """Print the chi-squared profile over alpha."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config", type=Path)
    ap.add_argument(
        "--alphas", type=float, nargs="+", default=[0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5]
    )
    ap.add_argument("--n-rounds", type=int, default=3)
    ap.add_argument("--debug", action="store_true", help="Three alpha values")
    args = ap.parse_args()
    if args.debug:
        args.alphas = [1.0, 2.0, 3.0]
    tmp = Path(tempfile.mkdtemp(prefix="act_alpha_"))
    rows = [
        (a, *fit_fixed_alpha(args.config, a, args.n_rounds, tmp / str(a)))
        for a in args.alphas
    ]
    best = min(r[1] for r in rows)
    print(f"{'alpha':>6}{'chisq':>11}{'dchisq':>9}{'m500':>11}{'amp_act':>9}")
    for a, c, m, p in rows:
        print(f"{a:6.2f}{c:11.2f}{c - best:9.2f}{m:11.3g}{p:9.1f}")
    print("min at alpha =", rows[int(np.argmin([r[1] for r in rows]))][0])


if __name__ == "__main__":
    main()
