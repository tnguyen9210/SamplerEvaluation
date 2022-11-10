

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
    
    # generate prior and posterior samples
    samples_a_weights_prior, samples_b_weights_prior, samples_a_weights_posterior = \
        generate_prior_and_posterior_samples(data_x, args)
    # print(samples_a_weights_prior.shape)
    # print(samples_a_weights_prior[:5])
    # print(samples_b_weights_prior[:5])
    # print(samples_a_weights_posterior[:5])

    # Visualize the generated prior and posterior samples
    fig, axes = plt.subplots(
        nrows=1, ncols=1, sharex=True, sharey=True, figsize=(10,10))
    # axes = axes.flatten()

    sns.kdeplot(samples_a_weights_prior[:,0], shade=False, color="blue",
                label="sample_a_prior", ax=axes)
    sns.kdeplot(samples_b_weights_prior[:,0], shade=False, color="green",
                label="sample_b_prior", ax=axes)
    sns.kdeplot(samples_a_weights_posterior[:,0], shade=False, color="orange",
                label="sample_a_posterior", ax=axes)
    plt.legend()
    plt.show()

def generate_data_x(args):
    # Generate data_x
    num_data = args["num_data"]

    data_x_marginal_dists = [
        stats.multivariate_normal(mu, sigma, seed=12345) \
        for mu, sigma in args["data_x_marginal_params"]]

    c0_x = data_x_marginal_dists[0].rvs(size=(num_data))[:,None]
    c1_x = data_x_marginal_dists[1].rvs(size=(num_data))[:,None]
    data_x = np.vstack((c0_x, c1_x))

    return data_x

def generate_prior_and_posterior_samples(data_x, args):
    
    # np.random.seed(0)

    # params
    num_feats = args["num_feats"]
    num_samples = args["num_samples"]
    laplace_init_sigma = args["laplace_init_sigma"]
    laplace_num_iters = args["laplace_num_iters"]
    
    # weights' prior distribution
    weights_prior_params = args["weights_prior_params"]

    weights_prior_dist_a = stats.multivariate_normal(
        weights_prior_params[0], weights_prior_params[1], seed=args['seed'])
    
    weights_prior_dist_b = stats.multivariate_normal(
        weights_prior_params[0], weights_prior_params[1], seed=args['seed']+10)

    
    samples_a_weights_prior = []
    samples_b_weights_prior = []
    samples_a_weights_posterior = []
    for i in range(num_samples):
        # sample two set of weights' priors 
        sample_a_weights_prior = weights_prior_dist_a.rvs(1)
        sample_b_weights_prior = weights_prior_dist_b.rvs(1)
        samples_a_weights_prior.append(sample_a_weights_prior)
        samples_b_weights_prior.append(sample_b_weights_prior)
    
        # generate sample y_i from theta_i in A
        sample_a_logit = 1.0 / (1 + np.exp(-np.dot(data_x, sample_a_weights_prior)))
        sample_a_y = stats.bernoulli.rvs(sample_a_logit)
        # print(sample_a_y.shape)
    
        # fit laplace approximation
        w_map, h_map = bayes_logistic.fit_bayes_logistic(
            y = sample_a_y.squeeze(-1),
            X = data_x, 
            wprior = sample_a_weights_prior,  # note: init prior for laplace approximation is same assample prior
            # wprior = np.zeros(num_feats),
            H = ((np.identity(num_feats)) * laplace_init_sigma),
            weights = None,
            solver = "Newton-CG",
            bounds = None,
            maxiter = laplace_num_iters
        )
        cov_map = np.linalg.inv(h_map)
    
        # sample weights' posterior
        sample_a_weights_posterior = stats.multivariate_normal.rvs(w_map, cov_map)
        samples_a_weights_posterior.append(sample_a_weights_posterior)    

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
