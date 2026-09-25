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
                                                        value: "np.linspace(1e14, 1e15, 5)"
                                                dx_1:
                                                        value: [-5.0, 0.0. 5.0]

Usage:
        python tools/grid_search.py setup.yaml
"""

import argparse
import itertools
import os

import numpy as np
import yaml


def parse_value(value):
    """
    Parse a parameter value which can be:
            - A python list of numbers
            - a numpy expression string
            - a scalar

    returns a list of numeric values
    """

    if isinstance(value, list):
        return [_to_number(v, value) for v in value]
    elif isinstance(value, str):
        result = eval(value, {"np": np})
        if hasattr(result, "__iter__"):
            return [_to_number(v, value) for v in result]
        else:
            return [_to_number(result, value)]
    elif isinstance(value, (int, float)):
        return [value]
    else:
        raise TypeError(
            f"Could not interpret grid value {value!r} (type {type(value)})"
        )


def _to_number(v, original):
    # Force a single grid entry to a float, with an error if it cant be
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            raise ValueError(
                f"Could not parse {v!r} as a number from gid value {original!r}. "
                "This usually means there is a missing comma in the list"
            )
    raise TypeError(f"Unexpected value type {type(v)} in {original!r}")


def deep_set(d, keys, value):
    # Set a nested dictionary value given a list of keys
    for key in keys[:-1]:
        d = d[key]
    d[keys[-1]] = value


def get_grid_params(grid):
    """
    Walk the grid spec and return:
            - param_keys: list of (model_name, struct_name, param_name) tuples
              for every parameter explicitly given a value in the grid spec
            - param_values: list of lists of values, one per entry in param_keys
            - grid_structs: set of (model_name, struct_name) tuples referenced by the grid,
              used to default any other parameters in those structures to to_fit: False
    """

    param_keys = []
    param_values = []
    grid_structs = set()

    for model_name, model_spec in grid.items():
        for struct_name, struct_spec in model_spec["structures"].items():
            grid_structs.add((model_name, struct_name))
            for param_name, param_spec in struct_spec["parameters"].items():
                values = parse_value(param_spec["value"])
                param_keys.append((model_name, struct_name, param_name))
                param_values.append(values)

    return param_keys, param_values, grid_structs


def get_unspecified_params(base_cfg, grid_structs, specified):
    """
    for each (model_name, struct_name) referenced in the grid, find any
    parameters in the base config that were not given a value in the
    grid spec. these get to_fit forced to False in generated configs.
    """

    unspecified = []
    for model_name, struct_name in grid_structs:
        all_params = base_cfg[model_name]["structures"][struct_name]["parameters"]
        for param_name in all_params:
            if (model_name, struct_name, param_name) not in specified:
                unspecified.append((model_name, struct_name, param_name))
    return unspecified


def format_value_for_filename(value):
    # format a numeric value so it is always filesystem/shell safe (no spaces)
    if isinstance(value, float):
        s = f"{value:.4g}"
    else:
        s = str(value)
    return s.replace(" ", "")


def load_base_cfg(path):
    # load and merge configs following the base chain
    with open(path) as f:
        cfg = yaml.safe_load(f)

    if "base" in cfg:
        base_path = cfg["base"]
        if not os.path.isabs(base_path):
            base_path = os.path.join(os.path.dirname(path), base_path)
        base_cfg = load_base_cfg(base_path)
        for k, v in base_cfg.items():
            if k not in cfg:
                cfg[k] = v
    return cfg


def main():
    parser = argparse.ArgumentParser(
        description="Generate grid search configs for LM fitter (issue 212)"
    )
    parser.add_argument("setup", help="Path to the setup yaml")
    args = parser.parse_args()

    with open(args.setup) as f:
        setup = yaml.safe_load(f)

    base_path = setup["base"]
    if not os.path.isabs(base_path):
        base_path = os.path.join(os.path.dirname(args.setup), base_path)

    output_dir = setup["output_dir"]
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(os.path.dirname(args.setup), output_dir)

    grid = setup["grid"]

    param_keys, param_values, grid_structs = get_grid_params(grid)

    if not param_keys:
        raise ValueError("No parameters found in grid spec!")

    specified = set(param_keys)
    template_cfg = load_base_cfg(base_path)
    unspecified_params = get_unspecified_params(template_cfg, grid_structs, specified)

    combinations = list(itertools.product(*param_values))
    print(f"Generating {len(combinations)} configs for {len(param_keys)} parameters")
    print(f"Parameters: {[f'{m}/{s}/{p}' for m, s, p in param_keys]}")
    if unspecified_params:
        print(
            "Forcing to_fit=False for unspecified parameters: "
            f"{[f'{m}/{s}/{p}' for m, s, p in unspecified_params]}"
        )
    print(f"Output directory: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    for i, combo in enumerate(combinations):
        cfg = load_base_cfg(base_path)
        cfg.pop("base", None)

        for (model_name, struct_name, param_name), value in zip(param_keys, combo):
            cfg[model_name]["structures"][struct_name]["parameters"][param_name][
                "value"
            ] = value

        for model_name, struct_name, param_name in unspecified_params:
            cfg[model_name]["structures"][struct_name]["parameters"][param_name][
                "to_fit"
            ] = False

        parts = [
            f"{param_name}_{format_value_for_filename(value)}"
            for (_, _, param_name), value in zip(param_keys, combo)
        ]
        fname = f"grid_{i:04d}_{'_'.join(parts)}.yaml"
        outpath = os.path.join(output_dir, fname)

        with open(outpath, "w") as f:
            yaml.dump(cfg, f, sort_keys=False)

    print(f"Done! Wrote {len(combinations)} configs to {output_dir}")


if __name__ == "__main__":
    main()
