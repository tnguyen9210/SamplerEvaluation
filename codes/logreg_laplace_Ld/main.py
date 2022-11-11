

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
    # visualize individual feature vectors (1st, 2nd, 3rd, ...)
    nrows = 3
    fig, axes = plt.subplots(nrows=nrows, ncols=1, sharex=True, sharey=True, figsize=(10,10))
    axes = axes.flatten()

    for i in range(nrows):
        sns.kdeplot(samples_a_weights_prior[:,0], fill=False, color="blue", label="sample_a_prior", ax=axes[i])
        sns.kdeplot(samples_b_weights_prior[:,0], fill=False, color="green", label="sample_b_prior", ax=axes[i])
        sns.kdeplot(samples_a_weights_posterior[:,0], fill=False, color="orange", label="sample_a_posterior", ax=axes[i])
        axes[i].legend()
    plt.show()

    # visualize pairs of feature vectors (1st and 2nd)
    fig, axes = plt.subplots(
        nrows=3, ncols=1, sharex=True, sharey=True, figsize=(10,10))
    # axes = axes.flatten()

    sns.kdeplot(x=samples_a_weights_prior[:,0], y=samples_a_weights_prior[:,1],
                n_levels=20, cmap="inferno", fill=False, cbar=True, ax=axes[0])

    sns.kdeplot(x=samples_b_weights_prior[:,0], y=samples_b_weights_prior[:,1],
                n_levels=20, cmap="inferno", fill=False, cbar=True, ax=axes[1])

    sns.kdeplot(x=samples_a_weights_posterior[:,0], y=samples_a_weights_posterior[:,1],
                n_levels=20, cmap="inferno", fill=False, cbar=True, ax=axes[2])
    axes[0].set_title("sample_a_prior")
    axes[1].set_title("sample_b_prior")
    axes[2].set_title("sample_a_posterior")
    plt.show()

def generate_data_x(args):
    # Generate data_x
    num_data = args["num_data"]
    num_feats = args["num_feats"]

    c0_mu = np.ones(num_feats)*(-1)
    c1_mu = np.ones(num_feats)*5

    c0_cov = np.identity(num_feats)
    c1_cov = np.identity(num_feats)*3
    # c0_cov = generate_psd_matrix(num_feats)
    # c1_cov = generate_psd_matrix(num_feats)*3

    data_x_marginal_params = [
        [c0_mu, c0_cov],
        [c1_mu, c1_cov]]

    data_x_marginal_dists = [
        stats.multivariate_normal(mu, sigma, seed=12345) \
        for mu, sigma in data_x_marginal_params]  # fixed random seed

    c0_x = data_x_marginal_dists[0].rvs(size=(num_data))
    c1_x = data_x_marginal_dists[1].rvs(size=(num_data))
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
        sample_a_weights_prior = weights_prior_dist_a.rvs(1)[None,:]
        sample_b_weights_prior = weights_prior_dist_b.rvs(1)[None,:]
        samples_a_weights_prior.append(sample_a_weights_prior)
        samples_b_weights_prior.append(sample_b_weights_prior)

    
        # generate sample y_i from theta_i in A
        sample_a_logit = 1.0 / (1 + np.exp(
            -np.matmul(data_x, sample_a_weights_prior.T)))
        sample_a_y = stats.bernoulli.rvs(sample_a_logit)
        # print(sample_a_y.shape)
    
        # fit laplace approximation
        w_map, h_map = bayes_logistic.fit_bayes_logistic(
            y = sample_a_y.squeeze(-1),
            X = data_x, 
            wprior = sample_a_weights_prior.squeeze(0),  # note: init prior for laplace approximation is same assample prior
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
