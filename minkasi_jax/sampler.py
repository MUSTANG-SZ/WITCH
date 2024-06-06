import minkasi_jax.forward_modeling as mfm
import numpy as np

import emcee

def parhelper(theta,fixed_pars,fix_pars_idx):
    _theta = np.empty(len(theta) + len(fixed_pars))

    _theta[ fix_pars_idx] = fixed_pars
    _theta[~fix_pars_idx] = theta

    return _theta


class sample:
    def __init__(self,model,todvec,skymap,nwalk=100,nstep=1000,nburn=500,pinit=None):
        
        params = np.array(model.pars)
        priors = model.priors

        fixed_pars_idx = ~np.array(model.to_fit)
        fixed_pars_val = params[fixed_pars_idx]

        ndims = params.shape[0]

        tods = mfm.make_tod_stuff(todvec,skymap,x0=model.x0,y0=model.y0)
        
        jitlike = mfm.jax.jit(lambda x: mfm.loglike(model,x,tods))

        def loglike(theta):
            _theta = parhelper(theta,fixed_pars_val,fixed_pars_idx)
            return jitlike(_theta)
        
        def logprior(theta):
            prob = np.array([priors[pi].logpdf(theta[pi]) for pi in range(ndims)])
            return np.sum(prob)
        
        def logpost(theta):
            lp = logprior(theta)
            if np.isfinite(lp):
                return lp+loglike(theta)
            return -np.inf

        self.sampler = emcee.EnsembleSampler(nwalk,ndims,logpost)

        print(f'* Running burn-in stage [{nburn} steps]')

        if pinit is not None:
            state = pinit[None,:]*(1.00+0.01*np.random.rand(nwalk,ndims))
        else:
            state = np.array([pp.rvs(size=nwalk) for pp in priors]).T
        state = self.sampler.run_mcmc(state,nburn,progress=True)

        print(f'* Running sampling stage [{nstep} steps]')
        self.sampler.reset()
        self.sampler.run_mcmc(state,nstep,progress=True)

        model.cur_round += 1

    def get_chain(self,thin=1,flat=False):
        return self.sampler.get_chain(flat=flat,thin=thin)