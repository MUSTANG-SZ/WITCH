"""
Generate a grid of config files for LM fitter grid search (Issue 212)

Takes a setup yaml that specifies parameter grids and writes one config yaml per
combination of parameter values. Useful for mitigating the
local minima problem in LM fitting by starting from many initial points.

Example setup yaml:
        base: "configs/RXJ1347/RXJ1347_a10.yaml"
        output_dir: "configs/RXJ1347/grid_search"
        grid:
                cluster_model:
                        structures:
                                a10:
                                        parameters:
                                                m500:
                                                        val: "np.linspace(1e14, 1e15, 5)"
                                                dx_1:
                                                        val: [-5.0, 0.0. 5.0]

Usage:
        python tools/grid_search.py setup.yaml
"""

import argparse
import itertools
import os

import numpy as np
import yaml


def parse_val(val):
    """
    Parse a parameter value which can be:
            - a python list
            - a numpy expression string
            - a scalar (fixed value)
    """

    if isinstance(val, list):
        return val
    elif isinstance(val, str):
        result = eval(val, {"np": np})
        if hasattr(result, "__iter__"):
            return list(result)
        else:
            return [result]
    else:
        return [val]


def deep_set(d, keys, value):
    # set a nested dict value given a list of keys
    for key in keys[:-1]:
        d = d[key]
    d[keys[-1]] = value


def get_grid_params(grid):
    """
    Walk grid spec and return:
            - param_keys: list of (model_name, struct_name, param_name) tuples
            - param_values: list of lists of values
    """

    param_keys = []
    param_values = []

    for model_name, model_spec in grid.items():
        for struct_name, struct_spec in model_spec["structures"].items():
            for param_name, param_spec in struct_spec["parameters"].items():
                values = parse_val(param_spec["val"])
                param_keys.append((model_name, struct_name, param_name))
                param_values.append(values)

    return param_keys, param_values


def load_base_cfg(path):
    # Load and merge configs following base chain
    with open(path) as f:
        cfg = yaml.safe_load(f)

    if "base" in cfg:
        base_path = cfg["base"]
        if not os.path.isabs(base_path):
            base_path = os.path.join(os.path.dirname(path), base_path)
        base_cfg = load_base_cfg(base_path)
        # merge: cfg takes priority over base
        for k, v in base_cfg.items():
            if k not in cfg:
                cfg[k] = v
    return cfg


def main():
    parser = argparse.ArgumentParser(
        description="Generate grid search configsd for LM fitter (Issue 212)"
    )
    parser.add_argument("setup", help="Path to the setup yaml")
    args = parser.parse_args()

    # Load setup yaml
    with open(args.setup) as f:
        setup = yaml.safe_load(f)

    base_path = setup["base"]
    if not os.path.isabs(base_path):
        base_path = os.path.join(os.path.dirname(args.setup), base_path)

    output_dir = setup["output_dir"]
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(os.path.dirname(args.setup), output_dir)

    grid = setup["grid"]

    # Parse the grid
    param_keys, param_values = get_grid_params(grid)

    if not param_keys:
        raise ValueError("No parameters found in grid spec!")

    # Generate all combinations
    combinations = list(itertools.product(*param_values))
    print(f"Generating {len(combinations)} configs for {len(param_keys)} parameters")
    print(f"Parameters: {[f'{m}/{s}/{p}' for m, s, p in param_keys]}")
    print(f"Output directory: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    for i, combo in enumerate(combinations):
        # Load a fresh copy of the base config for each combination
        cfg = load_base_cfg(base_path)

        # Remove base key since we've already merged
        cfg.pop("base", None)

        # Apply this combination's parameter values
        for (model_name, struct_name, param_name), value in zip(param_keys, combo):
            cfg[model_name]["structures"][struct_name]["parameters"][param_name][
                "value"
            ] = (float(value) if not isinstance(value, str) else value)

        # Build a good filename
        parts = [
            (
                f"{param_name}_{value:.4g}"
                if isinstance(value, float)
                else f"{param_name}_{value}"
            )
            for (_, _, param_name), value in zip(param_keys, combo)
        ]
        fname = f"grid_{i:04d}_{'_'.join(parts)}.yaml"
        outpath = os.path.join(output_dir, fname)

        with open(outpath, "w") as f:
            yaml.dump(cfg, f, sort_keys=False)

    print(f"Done~ Wrote {len(combinations)} configs to {output_dir}")


if __name__ == "__main__":
    main()
