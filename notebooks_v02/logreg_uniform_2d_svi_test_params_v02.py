# Import packages
import matplotlib.pyplot as plt

import seaborn as sns
sns.set()
sns.set_palette("tab10")

import random as rnd
import numpy as np
rnd.seed(0)
np.random.seed(0)

import scipy.stats as stats

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import pyro
import pyro.distributions as dist
import pyro.distributions.constraints as constraints
from pyro.nn import PyroModule, PyroSample

from utils import *
from cores import generate_prior_and_posterior_samples, sample_svi_approx

def main():
    # Set parameters
    # data_x parameters
    args = {}
    args["seed"] = 1
    args["num_feats"] = 2
    args["num_data"] = 2
    args["num_data_half"] = args['num_data'] // 2

    # num_samples
    args["num_samples"] = 400

    # weights prior distribution parameters
    args["weights_prior_params"] = [[0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]]]

    # parameters for inference model
    args["svi_num_iters"] = 200

    # Generate data_x
    # data_x = stats.uniform.rvs(0, 1, size=(num_data, num_feats), random_state=12345)
    data_x = np.array([[0, 1], [1, 0]])
    data_x = torch.tensor(data_x, dtype=torch.float)
    print(data_x)
    print(data_x.shape)

    weights_prior_params_list = [
        [[0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]]],
        [[0.0, 0.0], [[4.0, -1.0], [-1.0, 4.0]]],
        [[0.0, 0.0], [[25.0, 0.0], [0.0, 25.0]]],
        [[0.0, 0.0], [[49.0, 0.0], [0.0, 49.0]]],
    ]
    
    for weights_prior_params in weights_prior_params_list:
        print("\n-> weights_prior_params")
        print(weights_prior_params)
        args["weights_prior_params"] = weights_prior_params

        def logreg_model(x, y):
            num_data, num_feats = x.shape
            # set prior  
            w = pyro.sample("w", dist.MultivariateNormal(
                torch.tensor(weights_prior_params[0], dtype=torch.float).reshape(1, num_feats),
                torch.tensor(weights_prior_params[1], dtype=torch.float)).to_event(1))
    
            # compute ymean and sample
            ymean = torch.sigmoid(torch.matmul(x, w.squeeze(1).T)).squeeze(-1)
            with pyro.plate("data", x.shape[0]):
                pyro.sample("y", dist.Bernoulli(ymean), obs=y)

        def logreg_guide(x, y):
            num_data, num_feats = x.shape
            w_loc = pyro.param("w_loc", torch.tensor(weights_prior_params[0], dtype=torch.float).reshape(1, num_feats))
            w_scale = pyro.param('w_scale', torch.tensor(weights_prior_params[1], dtype=torch.float))
            w = pyro.sample('w', dist.MultivariateNormal(w_loc, w_scale).to_event(1))
        
        args["logreg_model"] = logreg_model
        args["logreg_guide"] = logreg_guide
            
        samples_a_weights_prior, samples_b_weights_prior, samples_a_weights_posterior = \
            generate_prior_and_posterior_samples(data_x, sample_svi_approx, args)
            
        samples_a_weights_prior = np.vstack(samples_a_weights_prior)
        samples_b_weights_prior = np.vstack(samples_b_weights_prior)
        samples_a_weights_posterior = np.vstack(samples_a_weights_posterior)
        
        # Maximum mean distance with RBF kernel
        mmd_rbf_prior_a_prior_b = compute_mmd_rbf(samples_a_weights_prior, samples_b_weights_prior)
        mmd_rbf_posterior_a_prior_b = compute_mmd_rbf(samples_a_weights_posterior, samples_b_weights_prior)
        print(f"MMD between prior a and prior b: {mmd_rbf_prior_a_prior_b:0.5f}")
        print(f"MMD between posterior a and prior b: {mmd_rbf_posterior_a_prior_b:0.5f}")
    
        # Wasserstein distance with RBF kernel
        wd_prior_a_prior_b = compute_wasserstein_distance(samples_a_weights_prior, samples_b_weights_prior)
        wd_posterior_a_prior_b = compute_wasserstein_distance(samples_a_weights_posterior, samples_b_weights_prior)
        print(f"Wasserstein distance between prior a and prior b: {wd_prior_a_prior_b:0.5f}")
        print(f"Wasserstein distance between posterior a and prior b: {wd_posterior_a_prior_b:0.5f}")
    
        # Difference between the standard deviations (from true mean) of two samples
        weights_prior_params = args["weights_prior_params"]
        diff_std_prior_a_prior_b = compute_diff_std(samples_a_weights_prior, samples_b_weights_prior, weights_prior_params[0])
        diff_std_posterior_a_prior_b = compute_diff_std(samples_a_weights_posterior, samples_b_weights_prior, weights_prior_params[0])
        print(f"Difference standard deviations between between prior a and prior b: {diff_std_prior_a_prior_b:0.5f}")
        print(f"Difference standard deviations between posterior a and prior b: {diff_std_posterior_a_prior_b:0.5f}")

if __name__ == '__main__':
    main()
