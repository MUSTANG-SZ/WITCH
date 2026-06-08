import os
import argparse as ap
from glob import glob
from minkasi.tools import presets_by_source as pbs
from minkasi.tods.io import cut_blacklist
import shutil
from pathlib import Path


def _make_parser() -> ap.ArgumentParser:
    parser = ap.ArgumentParser()
    parser.add_argument(
        "-r",
        "--idir",
        type=str,
        required=True,
        help="Base path that maps are relative to",
    )

    parser.add_argument(
        "-g",
        "--glob",
        type=str,
        required=True,
        help="Glob pattern below relative-to that lists the _map.fits files",
    )

    parser.add_argument(
        "-n",
        "--name",
        type=str,
        help="Name of cluster for getting bad TODs."
    )

    parser.add_argument(
        "-s",
        "--splits",
        type=int,
        help="Number of splits to do."
    )

    return parser

def main():
    parser = _make_parser()
    args = parser.parse_args()

    fnames = glob(args.idir + args.glob)
    bad_tods, _ = pbs.get_bad_tods(name=args.name, ndo=False, odo=False)

    fnames = cut_blacklist(fnames, bad_tods)
    fnames.sort()

    for i in range(args.splits):
        odir = os.path.dirname(os.path.dirname(args.idir))
        cur_odir = Path(odir + "_{}/mustang2".format(i))
        cur_odir.mkdir(parents=True, exist_ok=True)
        for j, fname in enumerate(fnames):
            if not ((i + j) % 4):
                continue
            shutil.copy(fname, cur_odir)

if __name__ == "__main__":
    main()





