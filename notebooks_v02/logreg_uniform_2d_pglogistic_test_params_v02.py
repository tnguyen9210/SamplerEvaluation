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
from pypolyagamma import PyPolyaGamma

from utils import *
from cores import generate_prior_and_posterior_samples, gibb_sample_polyagamma

def main():
    # Set parameters
    # data_x parameters
    args = {}
    args["seed"] = 1
    args["num_feats"] = 2
    args["num_data"] = 2
    args["num_data_half"] = args['num_data'] // 2

    # num_samples
    args["num_samples"] = 1000

    # weights prior distribution parameters
    args["weights_prior_params"] = [[0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]]]

    # init sigma used in numerical optimization for laplace approximation
    args["pg_burnin_steps"] = 200

    # Generate data_x
    # data_x = stats.uniform.rvs(0, 1, size=(num_data, num_feats), random_state=12345)
    data_x = np.array([[0, 1], [1, 0]])
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
        pg_dist = PyPolyaGamma(seed=0)
        args["pg_dist"] = pg_dist
        samples_a_weights_prior, samples_b_weights_prior, samples_a_weights_posterior = \
            generate_prior_and_posterior_samples(data_x, gibb_sample_polyagamma, args)

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
