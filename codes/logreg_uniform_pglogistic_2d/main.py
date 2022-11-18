

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
from pypolyagamma import PyPolyaGamma

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
    nrows = 2
    fig, axes = plt.subplots(nrows=nrows, ncols=1, sharex=True, sharey=True, figsize=(10,10))
    axes = axes.flatten()

    for i in range(nrows):
        sns.kdeplot(samples_a_weights_prior[:,0], fill=False, color="blue", label="sample_a_prior", ax=axes[i])
        sns.kdeplot(samples_b_weights_prior[:,0], fill=False, color="green", label="sample_b_prior", ax=axes[i])
        sns.kdeplot(samples_a_weights_posterior[:,0], fill=False, color="orange", label="sample_a_posterior", ax=axes[i])
        axes[i].legend()
    plt.savefig(f"./figures/{args['test_name']}_indi")
    plt.show()

    # Visualize the generated prior and posterior samples
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
    plt.savefig(f"./figures/{args['test_name']}_pair")
    plt.show()

def generate_data_x(args):
    # Generate data_x
    num_data = args["num_data"]
    num_feats = args["num_feats"]
    data_x_type = args["data_x_type"]

    if data_x_type == "twopoints":
        data_x = np.array([0, 1]*num_data//2)
    elif data_x_type == "uniform":
        data_x = np.random.randn(num_data, num_feats)
    elif data_x_type == "multigauss":
        data_x = np.random.rand(num_data, num_feats)
        mu = args["data_x_marginal_params"][0][0]
        sigma = args["data_x_marginal_params"][0][1]
        data_x = stats.multivariate_normal.rvs(mu, sigma, seed=12345)
    elif data_x_type == "mixgauss":
        data_x_marginal_dists = [
            stats.multivariate_normal(mu, sigma, seed=12345) \
            for mu, sigma in args["data_x_marginal_params"]]  # fixed random seed

        c0_x = data_x_marginal_dists[0].rvs(size=(num_data))
        c1_x = data_x_marginal_dists[1].rvs(size=(num_data))
        data_x = np.vstack((c0_x, c1_x))
    else:
        stop
        
    print(data_x.shape)

    return data_x

def generate_prior_and_posterior_samples(data_x, args):
    
    # np.random.seed(0)

    # params
    num_feats = args["num_feats"]
    num_samples = args["num_samples"]
    pglogistic_burnin_steps = args["pglogistic_burnin_steps"]
    
    # weights' prior distribution
    weights_prior_params = args["weights_prior_params"]

    weights_prior_dist_a = stats.multivariate_normal(
        weights_prior_params[0], weights_prior_params[1], seed=args['seed'])
    
    weights_prior_dist_b = stats.multivariate_normal(
        weights_prior_params[0], weights_prior_params[1], seed=args['seed']+10)

    samples_a_weights_prior = []
    samples_b_weights_prior = []
    samples_a_weights_posterior = []

    pg_dist = PyPolyaGamma(seed=0)
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
    
        # sample weights' posterior from polyagamma inference
        sample_a_weights_posterior = pg_inference(
            data_x, sample_a_y.squeeze(-1), weights_prior_params, pg_dist,
            burnin_steps=pglogistic_burnin_steps)
        samples_a_weights_posterior.append(sample_a_weights_posterior)

    samples_a_weights_prior = np.vstack(samples_a_weights_prior)
    samples_b_weights_prior = np.vstack(samples_b_weights_prior)
    samples_a_weights_posterior = np.vstack(samples_a_weights_posterior)
    # print(samples_a_weights_prior.shape)
    # print(samples_b_weights_prior)
    # print(samples_a_weights_posterior)
    return samples_a_weights_prior, samples_b_weights_prior, \
        samples_a_weights_posterior

def pg_inference(X, y, weights_prior_params, pg_dist, burnin_steps=200):
    #
    num_data, num_feats = X.shape

    # Gibbs sampling with PG augmentation for burnin_steps
    # init states
    beta_mu = np.array(weights_prior_params[0])
    beta_cov = np.array(weights_prior_params[1])
    # beta_cov = np.diag(np.ones(num_feats))
    beta_hat = np.random.multivariate_normal(beta_mu, beta_cov)
    k = y - 1/2

    # pg = PyPolyaGamma(seed=0)
    # perform Gibbs sampling
    for bid in range(burnin_steps+1):
        # ω ~ PG(b, c) = PG(1, x*β).
        omega_b = np.ones(num_data)
        omega_c = X @ beta_hat
        omega_diag = np.array(
            [pg_dist.pgdraw(b, c) for b, c in zip(omega_b, omega_c)])

        # β ~ N(m, V).
        V = np.linalg.inv(X.T @ np.diag(omega_diag) @ X + np.linalg.inv(beta_cov))
        m = np.dot(V, X.T @ k + np.linalg.inv(beta_cov) @ beta_mu)
        beta_hat = np.random.multivariate_normal(m, V)

    return beta_hat

if __name__ == '__main__':
    run_program()
