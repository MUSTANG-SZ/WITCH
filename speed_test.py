from minkasi_jax import conv_int_gnfw
import jax
import jax.numpy as jnp
import numpy as np
import timeit

#compile jit gradient function
jit_gnfw_deriv = jax.jacfwd(conv_int_gnfw, argnums = 0)

def helper(params, tod):
    x, y = tod
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    
    
    pred = conv_int_gnfw(params, x, y, 10., 0.5)

    derivs = jit_gnfw_deriv(params, x, y, 10., 0.5)
    
    #derivs = np.moveaxis(derivs, 2, 0)

    return derivs, pred

#compile vectorized functions
vmap_conv_int_gnfw = jax.vmap(conv_int_gnfw, in_axes = (None, 0, 0, None, None))
vmap_jit_gnfw_deriv = jax.jacfwd(vmap_conv_int_gnfw)

def vmap_helper(params, tod):
    x, y = tod
    x = jnp.asarray(x)
    y = jnp.asarray(y)


    pred = vmap_conv_int_gnfw(params, x, y, 10., 0.5)

    derivs = vmap_jit_gnfw_deriv(params, x, y, 10., 0.5)
    
    #derivs = np.moveaxis(derivs, 2, 0)

    return derivs, pred

test_tod = np.random.rand(2, int(1e4))
pars = np.array([0, 0, 1., 1., 1.5, 4.3, 0.7,3e14])

#Run both once to compile
_, __ = helper(pars, test_tod)
_,__ = vmap_helper(pars, test_tod)

def timeit_helper():
    return helper(pars, test_tod)

def vmap_timeit_helper():
    return vmap_helper(pars, test_tod)

print(timeit.timeit(timeit_helper, number = 100))
print(timeit.timeit(vmap_timeit_helper, number = 100))


