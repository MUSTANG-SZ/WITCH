import argparse as argp

import numpy as np

parser = argp.ArgumentParser()
parser.add_argument("chis_cyl", type=float)
parser.add_argument("--chis_cls", default=11751366.0, type=float)
parser.add_argument("--nsamp", default=12165304.0, type=float)
parser.add_argument("--ddof", default=2.0, type=float)

args = parser.parse_args()

fstat = ((args.chis_cls - args.chis_cyl) / (args.ddof)) / (args.chis_cyl / args.nsamp)
print(fstat)
print(np.sqrt(fstat))
