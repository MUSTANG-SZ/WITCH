"""
How does fitting with the wrong CMB covariance bias an ACT-only fit?

Simulates `unit_tests/act/act_sim_cmb.yaml` with the CMB covariance scaled by `cmb_scale`
(and optionally a different foreground amplitude) while the fit keeps the default noise model,
for several seeds. Prints the mean fitted value, scatter, mean reported error and
reduced chi-squared for each case.

Usage
-----
    python scripts/act_noise_mismatch.py --scales 0.5 1 2 --nseeds 5
    python scripts/act_noise_mismatch.py --debug
"""

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import dill
import numpy as np

CONFIG = Path(__file__).resolve().parents[1] / "unit_tests" / "act" / "act_sim_cmb.yaml"
TRUTH = {"m500": 5e14, "alpha": 1.551, "amp_act": 50.0}


def run_case(cmb_scale: float, fg: float, seed: int, n_rounds: int, out: Path) -> dict:
    """
    Simulate and fit one noise realisation.

    Parameters
    ----------
    cmb_scale : float
        Multiplier on the CMB covariance used for the simulated noise.
    fg : float
        Foreground D_l at l=3000 in uK^2 for the simulated noise.
    seed : int
        Seed of the simulated noise.
    n_rounds : int
        Number of LM rounds.
    out : Path
        Directory for the wrapper config and outputs.

    Returns
    -------
    dict
        Fitted parameters and errors.
    """
    out.mkdir(parents=True, exist_ok=True)
    wrapper = out / "wrapper.yaml"
    wrapper.write_text(
        f'base: "{CONFIG}"\nsim_seed: {seed}\nn_rounds: {n_rounds}\nmap: False\nmodel_map: False\n'
        "fitting:\n  maxiter: 50\n  chitol: 1e-2\n"
        f"datasets:\n  act:\n    sim_noise_kwargs: {{cmb_scale: {cmb_scale}, fg_dl3000: {fg}}}\n"
    )
    env = os.environ | {"WITCH_OUTROOT": str(out / "output")}
    res = subprocess.run(
        [shutil.which("witcher"), str(wrapper)],
        capture_output=True,
        env=env,
        text=True,
        check=False,
    )
    files = list((out / "output").rglob("par_results_final_fit.dill"))
    if res.returncode or not files:
        raise RuntimeError(res.stderr[-1500:])
    with files[0].open("rb") as f:
        r = dill.load(f)
    pars = {
        n: (float(p), float(e))
        for n, p, e, fit in zip(
            r["par_names"], r["parameters"], r["errs"], r["to_fit"], strict=True
        )
        if fit
    }
    return {"pars": pars}


def main() -> None:
    """Scan the simulated CMB covariance scale and print the fit bias."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scales", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    ap.add_argument("--fg", type=float, default=100.0, help="Simulated fg D_l(3000)")
    ap.add_argument("--nseeds", type=int, default=5)
    ap.add_argument("--n-rounds", type=int, default=3)
    ap.add_argument("--debug", action="store_true", help="Two scales, two seeds")
    args = ap.parse_args()
    if args.debug:
        args.scales, args.nseeds = [1.0, 2.0], 2
    tmp = Path(tempfile.mkdtemp(prefix="act_mismatch_"))
    print(
        f"{'scale':>6}{'par':>9}{'truth':>10}{'mean':>11}{'std':>10}{'<err>':>10}{'bias/SEM':>9}"
    )
    for scale in args.scales:
        runs = [
            run_case(scale, args.fg, s, args.n_rounds, tmp / f"{scale}_{s}")
            for s in range(args.nseeds)
        ]
        for name, truth in TRUTH.items():
            v = np.array([r["pars"][name][0] for r in runs])
            e = np.array([r["pars"][name][1] for r in runs])
            std = v.std(ddof=1) if len(v) > 1 else np.nan
            print(
                f"{scale:6.2f}{name:>9}{truth:10.4g}{v.mean():11.4g}{std:10.3g}"
                f"{e.mean():10.3g}{(v.mean() - truth) / std * np.sqrt(len(v)):9.2f}"
            )


if __name__ == "__main__":
    main()
