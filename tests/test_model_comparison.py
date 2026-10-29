from copy import deepcopy

import numpy as np
import pytest

from witch.model_comparison import (
    ChainQualityError,
    NestingError,
    Run,
    aic_table,
    check_nested,
    chisq_order_warnings,
    ess,
    ks_compare,
    order_nested,
)

NAMES = ["m.gnfw.P0", "m.gnfw.r_s", "m.gnfw.alpha", "ps.g.amp"]
CFG = {
    "name": "MOOJ1142",
    "sim": False,
    "coords": {"r_map": "3.0*60", "dr": 1.0},
    "datasets": {"mustang2": {"ntods": 106}},
    "paths": {"tods": "MOOJ1142", "subdir": "a"},
}


def _run(label, free, values=(1.0, 4.33, 2.26, 1e-3), cfg=CFG, nstep=4000, seed=0):
    """A synthetic run with independent gaussian chains for the free parameters."""
    free = np.array(free, dtype=bool)
    rng = np.random.default_rng(seed)
    chains = rng.normal(size=(4, nstep, int(free.sum())))
    return Run(
        label,
        f"/fake/{label}",
        deepcopy(cfg),
        list(NAMES),
        {"m.gnfw": "gnfw_rs", "ps.g": "gaussian"},
        np.array(values, dtype=float),
        np.array([[0.01, 4.0], [1.0, 50.0], [0.1, 10.0], [0.0, 1.0]]),
        free,
        chains,
        ((0, 1, 2), (3,)),
    )


def test_nested_pair_returns_extra_parameters():
    small = _run("fixed", [1, 0, 0, 1])
    big = _run("free", [1, 1, 0, 1])
    assert check_nested(small, big) == ["m.gnfw.r_s"]


def test_output_paths_may_differ():
    cfg = deepcopy(CFG)
    cfg["paths"]["subdir"] = "b"
    check_nested(_run("fixed", [1, 0, 0, 1]), _run("free", [1, 1, 0, 1], cfg=cfg))


@pytest.mark.parametrize(
    "small_free, big_free, big_values, big_cfg, match",
    [
        ([1, 1, 0, 1], [1, 0, 1, 1], None, None, "free in the smaller model only"),
        ([1, 1, 0, 1], [1, 1, 0, 1], None, None, "same parameters"),
        (
            [1, 0, 0, 1],
            [1, 1, 0, 1],
            (1.0, 4.33, 2.5, 1e-3),
            None,
            "fixed values differ",
        ),
        (
            [1, 0, 0, 1],
            [1, 1, 0, 1],
            None,
            {"datasets": {"mustang2": {"ntods": 10}}},
            "datasets.mustang2.ntods",
        ),
        (
            [1, 0, 0, 1],
            [1, 1, 0, 1],
            None,
            {"coords": {"r_map": "7.0*60", "dr": 1.0}},
            "coords.r_map",
        ),
    ],
)
def test_not_nested_raises(small_free, big_free, big_values, big_cfg, match):
    cfg = deepcopy(CFG)
    cfg.update(big_cfg or {})
    big = _run(
        "big", big_free, cfg=cfg, **({"values": big_values} if big_values else {})
    )
    with pytest.raises(NestingError, match=match):
        check_nested(_run("small", small_free), big)


def test_fixed_value_outside_larger_prior_raises():
    small = _run("small", [1, 0, 0, 1], values=(1.0, 60.0, 2.26, 1e-3))
    big = _run("big", [1, 1, 0, 1], values=(1.0, 60.0, 2.26, 1e-3))
    with pytest.raises(NestingError, match="outside the larger model's priors"):
        check_nested(small, big)


def test_different_structures_raise():
    big = _run("big", [1, 1, 0, 1])
    big.structures = {"m.gnfw": "a10", "ps.g": "gaussian"}
    with pytest.raises(NestingError, match="structures"):
        check_nested(_run("small", [1, 0, 0, 1]), big)


def test_order_nested_sorts_and_checks_all_pairs():
    runs = [
        _run("all", [1, 1, 1, 1]),
        _run("fixed", [1, 0, 0, 1]),
        _run("rs", [1, 1, 0, 1]),
    ]
    assert [r.label for r in order_nested(runs)] == ["fixed", "rs", "all"]
    runs.append(_run("alpha", [1, 0, 1, 1]))  # Not nested with "rs"
    with pytest.raises(NestingError):
        order_nested(runs)


def test_ess_of_independent_samples_is_close_to_n():
    chains = np.random.default_rng(1).normal(size=(4, 5000))
    assert 0.8 * 20000 < ess(chains) < 1.2 * 20000


def test_ks_same_distribution_is_not_rejected():
    rows = ks_compare(
        _run("fixed", [1, 0, 0, 1], seed=1), _run("free", [1, 1, 0, 1], seed=2)
    )
    assert [r["parameter"] for r in rows] == ["m.gnfw.P0", "ps.g.amp"]
    assert all(r["p"] > 1e-3 for r in rows)


def test_ks_stuck_chain_raises():
    big = _run("free", [1, 1, 0, 1])
    big.chains = np.repeat(big.chains[:, :4], 1000, axis=1)  # Chains that barely move
    with pytest.raises(ChainQualityError, match="ESS"):
        ks_compare(_run("fixed", [1, 0, 0, 1]), big)


def test_aic_and_chisq_ordering():
    runs = [_run("fixed", [1, 0, 0, 1]), _run("free", [1, 1, 0, 1])]
    rows = aic_table(runs, [100.0, 90.0])
    # k = 2 and 3 free parameters
    assert [r["AIC"] for r in rows] == [104.0, 96.0]
    assert rows[1]["dAIC"] == 0 and rows[0]["dAIC"] == 8.0
    np.testing.assert_allclose(sum(r["weight"] for r in rows), 1.0)
    assert chisq_order_warnings(runs, [100.0, 90.0]) == []
    assert len(chisq_order_warnings(runs, [90.0, 100.0])) == 1
