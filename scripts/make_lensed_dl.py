"""
Make the lensed TT CMB spectrum table used by `witch.noise`, with CAMB.

The default table in `witch/data/lensed_tt_planck2018.npz` was made with the defaults of this
script. Use it to store a spectrum for another cosmology, which can then be loaded with
`np.load(path)["dl"]` and passed as `dl` to `witch.noise.cmb_cl`, `cmb_covariance` or
`NoiseCMBMap`. Needs the optional `camb` dependency.

Usage
-----
    python scripts/make_lensed_dl.py --out witch/data/lensed_tt_planck2018.npz
    python scripts/make_lensed_dl.py --out lensed_tt_h70.npz --H0 70.0
    python scripts/make_lensed_dl.py --out /tmp/test.npz --debug
"""

import argparse
import json
from importlib.metadata import version

import numpy as np

from witch.noise import PLANCK2018, generate_lensed_dl


def main() -> None:
    """Parse the command line, run CAMB and write the table."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", required=True, help="Output .npz file")
    parser.add_argument("--lmax", type=int, default=10000, help="Maximum multipole")
    for name, val in PLANCK2018.items():
        parser.add_argument(f"--{name}", type=float, default=val, help=f"CAMB {name}")
    parser.add_argument("--debug", action="store_true", help="Use lmax=2000")
    args = parser.parse_args()

    lmax = 2000 if args.debug else args.lmax
    cosmology = {name: getattr(args, name) for name in PLANCK2018}
    dl = generate_lensed_dl(lmax, **cosmology)
    np.savez_compressed(
        args.out,
        ell=np.arange(len(dl)),
        dl=dl,
        cosmology=json.dumps(cosmology),
        camb_version=version("camb"),
        lens_potential_accuracy=1,
    )
    print(f"Wrote D_l for l=0..{lmax} to {args.out}")


if __name__ == "__main__":
    main()
