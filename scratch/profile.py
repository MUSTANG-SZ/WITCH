import argparse as argp
import datetime as dt
import os
import platform
import shutil
import subprocess

import astropy.units as u
import git
import jax
import jaxlib
import numpy as np
import yaml
from astropy.coordinates import Angle
from minkasi_jax import core
from minkasi_jax.utils import *


class FakeTod:
    def __init__(self, info):
        self.info = info


parser = argp.ArgumentParser(description="Profile the minkasi_jax library")
parser.add_argument(
    "--config",
    "-c",
    default="../configs/profiling/ms0735_like.yaml",
    help="Config file containing model to profile",
)
parser.add_argument(
    "--log_dir", "-d", default="profiles", help="Directory to store profile in"
)
parser.add_argument(
    "--keep_old",
    "-k",
    action="store_false",
    help="Don't overwrite an old profile with the same git hash",
)
parser.add_argument(
    "--no_meta",
    "-nm",
    action="store_true",
    help="If set, a metadata file won't be created",
)
parser.add_argument(
    "--no_link", "-nl", action="store_false", help="Don't create perfetto link"
)
args = parser.parse_args()

jax.config.update("jax_transfer_guard", "log_explicit")

# Figure out output file
os.makedirs(args.log_dir, exist_ok=True)
repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha
base_name = f"profile_{dt.date.today().isoformat().replace('-', '')}_{sha[:6]}_{jax.devices()[0].device_kind}"
log_file = os.path.join(args.log_dir, base_name + ".json.gz")
append = ""
if (not args.keep_old) and os.path.isfile(log_file):
    while os.path.isfile(log_file + append):
        append += ".new"
    log_file += append
meta_path = os.path.join(args.log_dir, base_name + ".meta" + append)

if args.no_meta and (not args.keep_old) and os.path.isfile(meta_path):
    os.remove(meta_path)

# Write out metadata
if not args.no_meta:
    meta = f"Config file: {args.config}\n\n"
    # Git stuff
    meta += "Git information:\n"
    meta += f"Full SHA: {sha}\n"
    meta += f"Staged files: {[item.a_path for item in repo.index.diff('HEAD')]}\n"
    meta += f"Unstaged files: {[item.a_path for item in repo.index.diff(None)]}\n"
    meta += f"Untracked files: {repo.untracked_files}\n\n"

    # Software into
    meta += "Software information:\n"
    meta += f"OS: {platform.platform()}\n"
    meta += f"jax: {jax.__version__}\n"
    meta += f"jaxlib: {jaxlib.__version__}\n\n"

    # Hardware info
    lscpu = subprocess.run(["lscpu"], capture_output=True).stdout.decode()
    lsmem = subprocess.run(["lsmem"], capture_output=True).stdout.decode()
    lsgpu = ""
    lspci = subprocess.run(["lspci", "-v"], capture_output=True)
    grep = subprocess.run(
        ["grep", "VGA"], input=lspci.stdout, capture_output=True
    ).stdout.decode()
    if len(grep) > 11:
        vgas = grep.split("\n")
        for vga in vgas:
            if len(vga) < 11:
                continue
            if vga[8:11] != "VGA":
                continue
            dev = vga[:7]
            gpuinfo = subprocess.run(
                ["lspci", "-vvnns", dev], capture_output=True
            ).stdout.decode()
            if len(gpuinfo):
                lsgpu += gpuinfo
    meta += "Hardware information:\n"
    meta += f"CPU information:\n {lscpu}\n"
    meta += f"RAM information:\n {lsmem}\n"
    meta += f"GPU information:\n {lsgpu}"

    with open(meta_path, "w") as file:
        print(meta, file=file)

with open(args.config, "r") as file:
    cfg = yaml.safe_load(file)

# Get device
# TODO: multi device setups
dev_id = cfg.get("jax_device", 0)
device = jax.devices()[dev_id]

# Setup coordindate stuff
z = eval(str(cfg["coords"]["z"]))
da = get_da(z)
r_map = eval(str(cfg["coords"]["r_map"]))
dr = eval(str(cfg["coords"]["dr"]))
xyz = make_grid(r_map, dr)
xyz_decoupled = make_grid(r_map, 2 * dr, 2 * dr, dr)
coord_conv = eval(str(cfg["coords"]["conv_factor"]))
x0 = eval(str(cfg["coords"]["x0"]))
y0 = eval(str(cfg["coords"]["y0"]))

x = np.arange(0, len(xyz[1]) / 100, dtype=int)
y = np.arange(0, len(xyz[1]) / 100, dtype=int)
X, Y = np.meshgrid(x, y)
X = X.ravel()
Y = Y.ravel()


Te = eval(str(cfg["cluster"]["Te"]))
freq = eval(str(cfg["cluster"]["freq"]))
beam = beam_double_gauss(
    dr,
    eval(str(cfg["beam"]["fwhm1"])),
    eval(str(cfg["beam"]["amp1"])),
    eval(str(cfg["beam"]["fwhm2"])),
    eval(str(cfg["beam"]["amp2"])),
)
dx = float(y2K_RJ(freq, Te) * dr * XMpc / me)

labels = []
params = []
to_fit = []
re_eval = []
par_idx = {}
for mname, model in cfg["models"].items():
    # We only want helper minkasi_jax functions
    if model["func"][:15] != "partial(helper,":
        continue
    _to_fit = []
    _re_eval = []
    _par_idx = {}
    for name, par in model["parameters"].items():
        labels.append(name)
        par_idx[mname + "-" + name] = len(params)
        _par_idx[mname + "-" + name] = len(_to_fit)
        params.append(eval(str(par["value"])))
        _to_fit.append(eval(str(par["to_fit"])))
        if "re_eval" in par and par["re_eval"]:
            _re_eval.append(str(par["value"]))
        else:
            _re_eval.append(False)
    to_fit = to_fit + _to_fit
    re_eval = re_eval + _re_eval
    # Only use the first model
    break
labels = np.array(labels)
params = np.array(params)
to_fit = np.array(to_fit, dtype=bool)
argnums = np.where(to_fit)[0]

print(f"params shape {params.shape}")
print(f"grid shape {xyz[0].shape}")
print(f"beam shape {beam.shape}")
print(f"X shape {X.shape}")
print(f"Y shape {Y.shape}")

# Make a fake TOD
info = {}
idu, id_inv = np.unique(np.vstack((X.ravel(), Y.ravel())), axis=1, return_inverse=True)
info["idx"] = jax.device_put(idu[0], device)
info["idy"] = jax.device_put(idu[1], device)
info["id_inv"] = id_inv
info["dx"] = X
tod = FakeTod(info)

# Now profile
jax.profiler.start_trace(
    "/tmp/jax-trace", create_perfetto_link=args.no_link, create_perfetto_trace=True
)
with jax.profiler.TraceAnnotation("Moving data"):
    xyz = jax.device_put(xyz)
    xyz[0].block_until_ready()
    xyz[1].block_until_ready()
    xyz[2].block_until_ready()
    xyz_decoupled = jax.device_put(xyz_decoupled, device)
    xyz_decoupled[0].block_until_ready()
    xyz_decoupled[1].block_until_ready()
    xyz_decoupled[2].block_until_ready()
    beam = jax.device_put(beam, device)
    beam.block_until_ready()
    X = jax.device_put(X, device)
    X.block_until_ready()
    Y = jax.device_put(Y, device)
    Y.block_until_ready()
    params = jax.device_put(params)
    params.block_until_ready()

with jax.profiler.TraceAnnotation("Standard grid"):
    with jax.profiler.TraceAnnotation("No JIT"):
        with jax.disable_jit():
            profile = core.model(xyz, 2, 0, 0, 0, 3, 0, 0, 0, dx, beam, params)
            profile.block_until_ready()

    with jax.profiler.TraceAnnotation("JITing"):
        profile = core.model(xyz, 2, 0, 0, 0, 3, 0, 0, 0, dx, beam, params)
        profile.block_until_ready()

    with jax.profiler.TraceAnnotation("JITed"):
        profile = core.model(xyz, 2, 0, 0, 0, 3, 0, 0, 0, dx, beam, params)
        profile.block_until_ready()

with jax.profiler.TraceAnnotation("Grid xy 4x coarser"):
    with jax.profiler.TraceAnnotation("JITing"):
        profile = core.model(xyz_decoupled, 2, 0, 0, 0, 3, 0, 0, 0, dx, beam, params)
        profile.block_until_ready()

    with jax.profiler.TraceAnnotation("JITed"):
        profile = core.model(xyz_decoupled, 2, 0, 0, 0, 3, 0, 0, 0, dx, beam, params)
        profile.block_until_ready()

for i in range(2):
    grad, pred = core.helper(
        params,
        tod,
        xyz,
        dx,
        beam,
        np.arange(5),
        [False] * len(params),
        par_idx,
        n_isobeta=2,
        n_uniform=3,
    )
jax.profiler.stop_trace()


tracedir = "/tmp/jax-trace/plugins/profile/"
all_subdirs = [
    os.path.join(tracedir, d)
    for d in os.listdir(tracedir)
    if os.path.isdir(os.path.join(tracedir, d))
]
moved = shutil.move(
    os.path.join(max(all_subdirs, key=os.path.getmtime), "perfetto_trace.json.gz"),
    log_file,
)
print(f"Trace can be found at {moved}")
if not args.no_meta:
    print(f"Metadata can be found at {meta_path}")
print("Upload trace to https://ui.perfetto.dev/ to view.")
