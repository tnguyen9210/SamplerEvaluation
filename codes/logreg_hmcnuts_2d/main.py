

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

color_list = ["Green", "Blue"]
color_map = mcolors.ListedColormap(["Green", "Blue"])

import seaborn as sns
sns.set()
sns.set_palette("tab10")

import random as rnd
import numpy as np
rnd.seed(0)
np.random.seed(0)

import scipy.stats as stats
import bayes_logistic

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import pyro
import pyro.distributions as dist
import pyro.distributions.constraints as constraints
from pyro.nn import PyroModule, PyroSample

# 
from config import parse_args
from utils import print_config

def run_program(config=None):

    # get params
    args = parse_args()
    if config:
        args.update(config)
    print_config(args)
        
    # generate data_x
    data_x = generate_data_x(args)
    # print(data_x.shape)
    # stop
    
    # generate prior and posterior samples
    samples_a_weights_prior, samples_b_weights_prior, samples_a_weights_posterior = \
        generate_prior_and_posterior_samples(data_x, args)
    # print(samples_a_weights_prior.shape)
    # print(samples_a_weights_prior[:5])
    # print(samples_b_weights_prior[:5])
    # print(samples_a_weights_posterior[:5])
    # stop

    # Visualize the generated prior and posterior samples
    fig, axes = plt.subplots(
        nrows=3, ncols=1, sharex=True, sharey=True, figsize=(10,10))
    # axes = axes.flatten()

    sns.kdeplot(x=samples_a_weights_prior[:,0], y=samples_a_weights_prior[:,1],
                n_levels=20, cmap="inferno", shade=False, cbar=True, ax=axes[0])

    sns.kdeplot(x=samples_b_weights_prior[:,0], y=samples_b_weights_prior[:,1],
                n_levels=20, cmap="inferno", shade=False, cbar=True, ax=axes[1])

    sns.kdeplot(x=samples_a_weights_posterior[:,0], y=samples_a_weights_posterior[:,1],
                n_levels=20, cmap="inferno", shade=False, cbar=True, ax=axes[2])
    axes[0].set_title("sample_a_prior")
    axes[1].set_title("sample_b_prior")
    axes[2].set_title("sample_a_posterior")
    plt.show()

def generate_data_x(args):
    # Generate data_x
    num_data = args["num_data"]

    data_x_marginal_dists = [
        stats.multivariate_normal(mu, sigma, seed=12345) \
        for mu, sigma in args["data_x_marginal_params"]]  # fixed random seed

    c0_x = data_x_marginal_dists[0].rvs(size=(num_data))
    c1_x = data_x_marginal_dists[1].rvs(size=(num_data))
    data_x = np.vstack((c0_x, c1_x))

    return data_x

def generate_prior_and_posterior_samples(data_x, args):
    
    # np.random.seed(0)

    # params
    num_feats = args["num_feats"]
    num_samples = args["num_samples"]
    
    # weights' prior distribution
    weights_prior_params = args["weights_prior_params"]

    weights_prior_dist_a = stats.multivariate_normal(
        weights_prior_params[0], weights_prior_params[1], seed=args['seed'])
    
    weights_prior_dist_b = stats.multivariate_normal(
        weights_prior_params[0], weights_prior_params[1], seed=args['seed']+10)

    def logistic_regression_model(x, y):
        num_data, num_feats = x.shape
        # set prior  
        w = pyro.sample("w", dist.MultivariateNormal(
            torch.tensor(weights_prior_params[0], dtype=torch.float).reshape(1, num_feats),
            torch.tensor(weights_prior_params[1], dtype=torch.float)).to_event(1))
    
        # compute ymean and sample
        ymean = torch.sigmoid(torch.matmul(x, w.squeeze(1).T)).squeeze(-1)
        with pyro.plate("data", x.shape[0]):
            pyro.sample("y", dist.Bernoulli(ymean), obs=y)
        
    nuts_kernel = pyro.infer.NUTS(logistic_regression_model)
    mcmc = pyro.infer.MCMC(
        nuts_kernel, num_samples=1, warmup_steps=200, disable_progbar=False)


    samples_a_weights_prior = []
    samples_b_weights_prior = []
    samples_a_weights_posterior = []
    for i in range(num_samples):
        # sample two set of weights' priors 
        sample_a_weights_prior = weights_prior_dist_a.rvs(1)[None,:]
        sample_b_weights_prior = weights_prior_dist_b.rvs(1)[None,:]
        samples_a_weights_prior.append(sample_a_weights_prior)
        samples_b_weights_prior.append(sample_b_weights_prior)
    
        # generate sample y_i from theta_i in A
        sample_a_logit = 1.0 / (1 + np.exp(
            -np.matmul(data_x, sample_a_weights_prior.T)))
        sample_a_y = stats.bernoulli.rvs(sample_a_logit)
        # print(sample_a_y.shape)
    
        # sample weights' posterior using hmc
        mcmc.run(torch.tensor(data_x, dtype=torch.float),
                 torch.tensor(sample_a_y, dtype=torch.float))
        mcmc_samples = mcmc.get_samples()["w"].reshape(1, num_feats)
        samples_a_weights_posterior.append(mcmc_samples.data.cpu().numpy())  
        del mcmc_samples
        torch.cuda.empty_cache()
        pyro.clear_param_store() 
        

    samples_a_weights_prior = np.vstack(samples_a_weights_prior)
    samples_b_weights_prior = np.vstack(samples_b_weights_prior)
    samples_a_weights_posterior = np.vstack(samples_a_weights_posterior)
    # print(samples_a_weights_prior.shape)
    # print(samples_b_weights_prior)
    # print(samples_a_weights_posterior)
    return samples_a_weights_prior, samples_b_weights_prior, \
        samples_a_weights_posterior
    
if __name__ == '__main__':
    run_program()
