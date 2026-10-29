import jax.numpy as jnp
import numpy as np

from witch.containers import MetaModel
from witch.utils import NullComm


def test_bookkeeping_properties_without_models():
    """
    The bookkeeping properties work and keep their types.
    Regression test for a global np -> jnp replace that broke all of them while every
    other test still passed. `to_fit_idx` has to stay numpy so it is concrete inside jit,
    and `par_names` holds strings, which JAX arrays can't.
    """
    metamodel = MetaModel(
        NullComm(),
        (),
        (),
        (),
        (),
        (),
        jnp.zeros(3),
        jnp.zeros(3),
        jnp.zeros((3, 3)),
        jnp.asarray(0.0),
    )

    assert len(metamodel.par_names) == 3
    assert isinstance(metamodel.to_fit_idx, np.ndarray)
    assert len(metamodel.to_fit_idx) == 0
    assert not np.any(np.asarray(metamodel.to_fit))
    assert not np.any(np.asarray(metamodel.to_fit_ever))
    low, high = metamodel.priors
    assert low.shape == high.shape == (3,)
