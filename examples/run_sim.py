# author: Peter Wijeratne (p.wijeratne@pm.me)
# example LEMING training on simulated data
import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path

from leming import LEMING
from utils import gen_data

try:
    seed = int(sys.argv[1])
except IndexError:
    seed = 42
np.random.seed(seed)

if __name__ == "__main__":
    
    # model parameters
    n_sinkhorn = 20 # number of Sinkhorn-Knopp iterations
    temperature_start = 1E1 # temperature hyperparameter - start value (for annealing)
    temperature_end = 1E0 # temperature hyperparameter - end value
    temperature_prior = 1E0 # temperature prior hyperparameter
    gumbel_scale = 0 # Gumbel noise hyperparameter
    if gumbel_scale > 0:
        n_mc_samples = 100 # number of Monte Carlo samples
    else:
        n_mc_samples = 1
    n_iters = 100 # number of ADAM iterations
    step_size = 1E-1 # step size for ADAM
        
    # simulate data
    n_ppl = 100 # number of individuals
    n_fts = 10 # number of features
    n_obs = 1 # number of observations per individual
    sigma_noise = 0.1 # standard deviation of noise    
    print ('Generating simulated data...')
    # X is observed data for each individual and each observation: shape (n_ppl, n_features, n_obs)
    # X0 is the first observation for each individual only: shape (n_ppl, n_features)
    # labels is the control ("con") or case ("case") labels: shape (n_ppl)
    # seq_true is the true simulated sequence, used for post-hoc comparison: shape (n_fts+1)
    X, _, _, labels, X0, _, _, seq_true, _, _, _ = gen_data(n_ppl, n_fts, n_obs, sigma_noise)
    print ('n_ppl {} n_fts {} n_iters {} step_size {} n_sinkhorn {} temperature_start {} temperature_end {} temperature_prior {} gumbel_scale {} n_mc_samples {} sigma_noise {}'.format(n_ppl, n_fts, n_iters, step_size, n_sinkhorn, temperature_start, temperature_end, temperature_prior, gumbel_scale, n_mc_samples, sigma_noise))
    
    # run model
    print("Training LEMING...")
    model = LEMING(X=X0,
                   labels=labels,
                   n_sinkhorn=n_sinkhorn,
                   temperature_start=temperature_start,
                   temperature_end=temperature_end,
                   temperature_prior=temperature_prior,
                   gumbel_scale=gumbel_scale,
                   n_mc_samples=n_mc_samples,
                   n_iters=n_iters,
                   step_size=step_size,
                   use_em=True,
                   verbose=True)
    model.train()
    S, S_samples = model.get_sequence(n_samples=10)
    conf_soft = model.confusion_soft()
    fig, ax = model.plot_confusion(conf_soft, S)
    conf_hard = model.confusion_hard(S, S_samples)
    fig, ax = model.plot_confusion(conf_hard, S)
    plt.show()
