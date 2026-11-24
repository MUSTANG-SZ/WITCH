"""
Datasets left out of the fit (`datasets.<name>.fit: False`): their objective is zero, and the config check refuses
free parameters that only they would constrain.
"""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from witch.cfg_loader import check_fit_datasets
from witch.containers import MetaModel
from witch.objective import null_objective
from witch.utils import NullComm


def test_null_objective_is_zero():
    """The objective of an excluded dataset adds nothing, with the shapes joint_objective sums."""
    npar = 4
    metamodel = MetaModel(
        NullComm(),
        (),
        (),
        (),
        (),
        (),
        jnp.ones(npar),
        jnp.zeros(npar),
        jnp.zeros((npar, npar)),
        jnp.asarray(0.0),
    )
    chisq, grad, curve = null_objective(metamodel, 0)
    assert float(chisq) == 0.0
    np.testing.assert_array_equal(grad, np.zeros(npar))
    np.testing.assert_array_equal(curve, np.zeros((npar, npar)))


def _model(name: str, pars: list[tuple[str, str]]) -> SimpleNamespace:
    """
    A stub model with the parameter bookkeeping check_fit_datasets reads.

    Parameters
    ----------
    name : str
        Model name.
    pars : list[tuple[str, str]]
        (structure, parameter) names.

    Returns
    -------
    SimpleNamespace
        With `name`, `par_struct_names` and `par_names`.
    """
    return SimpleNamespace(
        name=name,
        par_struct_names=[s for s, _ in pars],
        par_names=[p for _, p in pars],
    )


def _setup(
    act_fit: bool, act_amp_free: bool, m2_fit: bool = True
) -> tuple[dict, SimpleNamespace]:
    """
    Config and stub metamodel shaped like the MOOJ1142 joint config: the cluster model in both datasets, an M2 and
    an ACT point source, the ACT one's position tied to the M2 one's.

    Parameters
    ----------
    act_fit : bool
        datasets.act.fit.
    act_amp_free : bool
        Whether the ACT point source amplitude is free.
    m2_fit : bool, default: True
        datasets.mustang2.fit.

    Returns
    -------
    cfg : dict
        The config.
    metamodel : SimpleNamespace
        Stub with `models`, `parameter_map` and `to_fit_ever`.
    """
    cfg = {
        "datasets": {"mustang2": {"fit": m2_fit}, "act": {"fit": act_fit}},
        "metamodel": {
            "models": ["cluster_model", "ps_model", "ps_act"],
            "model_map": {
                "mustang2": ["cluster_model", "ps_model"],
                "act": ["cluster_model", "ps_act"],
            },
        },
    }
    models = [
        _model("cluster_model", [("gnfw", "P0"), ("gnfw", "r_s")]),
        _model("ps_model", [("ps", "dx_g"), ("ps", "amp_g")]),
        _model("ps_act", [("ps", "dx_g"), ("ps", "amp_g")]),
    ]
    # metamodel slots: 0 P0, 1 r_s, 2 dx_g (shared by both point sources), 3 M2 amp_g, 4 ACT amp_g
    parameter_map = ((0, 1), (2, 3), (2, 4))
    to_fit = np.array([True, True, True, True, act_amp_free])
    return cfg, SimpleNamespace(
        models=models, parameter_map=parameter_map, to_fit_ever=to_fit
    )


def test_all_datasets_fit_passes():
    """Nothing to check when every dataset is fit."""
    check_fit_datasets(*_setup(act_fit=True, act_amp_free=True))


def test_excluded_dataset_with_its_params_fixed_passes():
    """ACT out of the fit with its amplitude fixed: the tied position is constrained through the M2 point source."""
    check_fit_datasets(*_setup(act_fit=False, act_amp_free=False))


def test_free_param_of_excluded_dataset_raises():
    """A free parameter only the excluded dataset sees is refused, by its full name."""
    with pytest.raises(ValueError, match=r"ps_act\.ps\.amp_g") as err:
        check_fit_datasets(*_setup(act_fit=False, act_amp_free=True))
    assert "dx_g" not in str(err.value)


def test_no_fitted_dataset_raises():
    """Leaving every dataset out of the fit is refused."""
    with pytest.raises(ValueError, match="Every dataset"):
        check_fit_datasets(*_setup(act_fit=False, act_amp_free=False, m2_fit=False))
