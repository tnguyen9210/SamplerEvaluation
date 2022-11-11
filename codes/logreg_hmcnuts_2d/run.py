# Import packages
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


# Set parameters
# num_data
num_data = 2
num_data_half = num_data // 2
num_feats = 2

# num_samples 
num_samples = 1000

# data_x marginal distribution parameters
data_x_marginal_params = [
    [[1.0, 5.0], [(1.0, 0.0), (0.0, 1.0)]],
    [[-5.0, 1.0], [(3.0, 0.0), (0.0, 3.0)]]]

# weights prior distribution parameters
weights_prior_params = [[0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]]]

# Generate data_x
data_x_marginal_dists = [
    stats.multivariate_normal(mu, sigma, seed=12345) \
        for mu, sigma in data_x_marginal_params]

# c0_x = data_x_marginal_dists[0].rvs(size=(num_data_half))
# c1_x = data_x_marginal_dists[1].rvs(size=(num_data_half))
# data_x = np.vstack((c0_x, c1_x))

### Generate prior and posterior samples

# weights' prior distribution
weights_prior_dist_a = stats.multivariate_normal(
    weights_prior_params[0], weights_prior_params[1], seed=1)
weights_prior_dist_b = stats.multivariate_normal(
    weights_prior_params[0], weights_prior_params[1], seed=11)


samples_a_weights_prior = []
samples_b_weights_prior = []
samples_a_y = []
samples_a_x = []
samples_a_weights_posterior = []
for i in range(num_samples):
    # sample two set of weights' priors 
    sample_a_weights_prior = weights_prior_dist_a.rvs(1)[None,:]
    sample_b_weights_prior = weights_prior_dist_b.rvs(1)[None,:]
    samples_a_weights_prior.append(sample_a_weights_prior)
    samples_b_weights_prior.append(sample_b_weights_prior)
    # print(sample_a_weights_prior.shape)
    # print(sample_b_weights_prior.shape)
    
    sample_c0_x = data_x_marginal_dists[0].rvs(size=(num_data_half))
    sample_c1_x = data_x_marginal_dists[1].rvs(size=(num_data_half))
    sample_a_x = np.vstack((sample_c0_x, sample_c1_x))
    samples_a_x.append(sample_a_x)

    # generate sample y_i from theta_i in A
    sample_a_logit = 1.0 / (1 + np.exp(-np.matmul(sample_a_x, sample_a_weights_prior.T)))
    sample_a_y = stats.bernoulli.rvs(sample_a_logit)
    # print(sample_a_y.shape)
    # stop
    samples_a_y.append(sample_a_y)
    
samples_a_weights_prior = np.vstack(samples_a_weights_prior)
samples_b_weights_prior = np.vstack(samples_b_weights_prior)
print(samples_a_weights_prior.shape)
print(samples_b_weights_prior.shape)

samples_a_y = np.vstack(samples_a_y)
samples_a_x = np.vstack(samples_a_x)
print(samples_a_y.shape)
print(samples_a_x.shape)

def logistic_regression_model(x, y):
    num_data, num_feats = x.shape
    # set prior  
    # w = pyro.sample("w", dist.MultivariateNormal(
    #     torch.tensor(weights_prior_params[0], dtype=torch.float).reshape(1, num_feats),
    #     torch.tensor(weights_prior_params[1], dtype=torch.float)).to_event(1))
    w = pyro.sample(
        "w", dist.Normal(
            torch.zeros(num_feats), torch.ones(num_feats)).to_event(1))
    
    # compute ymean and sample
    ymean = torch.sigmoid(torch.matmul(x, w)).squeeze(-1)
    with pyro.plate("data", x.shape[0]):
        pyro.sample("y", dist.Bernoulli(ymean), obs=y)
        
nuts_kernel = pyro.infer.NUTS(logistic_regression_model)
mcmc = pyro.infer.MCMC(nuts_kernel, num_samples=num_samples,
                       warmup_steps=200, num_chains=1)

# sample weights' posterior using hmc
mcmc.run(torch.tensor(samples_a_x, dtype=torch.float), torch.tensor(samples_a_y, dtype=torch.float))
mcmc_samples = mcmc.get_samples()
samples_a_weights_posterior = mcmc_samples["w"].reshape(num_samples, num_feats)
print(samples_a_weights_posterior.shape)

# Visualize the generated prior and posterior samples 
fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=True, figsize=(10,10))
axes = axes.flatten()

sns.kdeplot(x=samples_a_weights_prior[:,0], y=samples_a_weights_prior[:,1], n_levels=20, 
            cmap="inferno", shade=False, cbar=True, ax=axes[0])

sns.kdeplot(x=samples_b_weights_prior[:,0], y=samples_b_weights_prior[:,1], n_levels=20, 
            cmap="inferno", shade=False, cbar=True, ax=axes[1])

sns.kdeplot(x=samples_a_weights_posterior[:,0], y=samples_a_weights_posterior[:,1], n_levels=20, 
            cmap="inferno", shade=False, cbar=True, ax=axes[2])
axes[0].set_title("sample_a_prior")
axes[1].set_title("sample_b_prior")
axes[2].set_title("sample_a_posterior")
plt.show()
