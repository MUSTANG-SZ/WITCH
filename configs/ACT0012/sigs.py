import numpy as np
import argparse as argp

parser = argp.ArgumentParser()
parser.add_argument("chis_cyl", type=float)
parser.add_argument("--chis_cls", default=11751366., type=float)
parser.add_argument("--nsamp", default = 12165304., type=float)
parser.add_argument("--ddof", default = 2., type=float)

args = parser.parse_args()

fstat = ((args.chis_cls - args.chis_cyl)/(args.ddof)) / (args.chis_cyl / args.nsamp)
print(fstat)
print(np.sqrt(fstat))




