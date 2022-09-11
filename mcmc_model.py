import torch
import pyro
import pyro.distributions as distr
import numpy as np


def model(K, dataset):
    hyperparameters = { \
        "mean_scale":min(dataset.float().var(), torch.tensor(1000).float()), \
        "mean_loc":dataset.float().max() / 2, \
        
        # mean and sd for the Normal prior of the variance
        "var_loc":torch.tensor(120).float(), \
        "var_scale":torch.tensor(130).float(), \
        "min_var":torch.tensor(5).float(), \
        
        "eta":torch.tensor(1).float(), \
        
        # slope and intercepts for the variance constraints
        "slope":torch.tensor(0.15914).float(), "intercept":torch.tensor(23.70988).float()}

    N, T = dataset.shape

    weights = pyro.sample("weights", distr.Dirichlet(torch.ones(K)))  # mixing proportions for each component sample the mixing proportion

    mean_scale = hyperparameters["mean_scale"]
    mean_loc = hyperparameters["mean_loc"]
    var_loc = hyperparameters["var_loc"]
    var_scale = hyperparameters["var_scale"]
    eta = hyperparameters["eta"]

    with pyro.plate("time_plate", T):
        with pyro.plate("comp_plate", K):
            mean = pyro.sample("mean", distr.HalfNormal(mean_scale)).add(mean_loc)

    with pyro.plate("time_plate2", T):
        with pyro.plate("comp_plate3", K):
            sigma_vector = pyro.sample("sigma_vector", distr.HalfNormal(var_scale)).add(var_loc)  # sampling sigma, the sd

    with pyro.plate("comp_plate2", K):
        sigma_chol = pyro.sample("sigma_chol", distr.LKJCholesky(T, eta))

    Sigma = compute_Sigma(sigma_chol, sigma_vector, K, T)

    with pyro.plate("data_plate", N):
        z = pyro.sample("z", distr.Categorical(weights), infer={"enumerate":"parallel"})
        x = pyro.sample("obs", distr.MultivariateNormal(loc=mean[z], \
            scale_tril=Sigma[z]), obs=dataset)
            

def compute_Sigma(sigma_chol, sigma_vector, K, T):
    '''
    Function to compute the sigma_tril used in the Normal likelihood
    '''
    Sigma = torch.zeros((K, T, T))
    for k in range(K):
        Sigma[k,:,:] = torch.mm(sigma_vector[k,:].diag_embed(), \
            sigma_chol[k]).add(torch.eye(T))
    return Sigma


def run(K, dataset, model=model, n_samples=100, n_chains=1, warmup=50):
    kernel = pyro.infer.mcmc.NUTS(model)

    mcmc = pyro.infer.mcmc.api.MCMC(kernel, 
                                    num_samples=n_samples, 
                                    num_chains=n_chains, 
                                    warmup_steps=warmup)
    
    mcmc.run(K=K, dataset=torch.tensor(dataset.values))
    
    return mcmc
