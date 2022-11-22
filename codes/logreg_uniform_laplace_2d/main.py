

import random as rnd
import numpy as np
# rnd.seed(0)
# np.random.seed(0)

import scipy.stats as stats
import bayes_logistic

# 
from config import parse_args
from utils import print_config, plot_prior_posterior_samples

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

    # measure distance
    
    
    # visualize the generated prior and posterior samples
    plot_prior_posterior_samples(
        samples_a_weights_prior,
        samples_b_weights_prior,
        samples_a_weights_posterior,
        num_feats=2, args=args)
   

def generate_data_x(args):
    # Generate data_x
    num_data = args["num_data"]
    num_feats = args["num_feats"]
    data_x_type = args["data_x_type"]

    if data_x_type == "twopoints":
        if num_data != 2:
            raise ValueError("num_data should equal 2")
        data_x = np.array([[0, 1], [1, 0]])
        
    elif data_x_type == "uniform":
        data_x = stats.uniform.rvs(
            0, 1, size=(num_data, num_feats), random_state=12345)
        
    elif data_x_type == "multigauss":
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
        
    print(data_x)
    print(data_x.shape)

    return data_x


def generate_prior_and_posterior_samples(data_x, args):
    
    # params
    num_feats = args["num_feats"]
    num_samples = args["num_samples"]
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
        # note:
        # init params (mu and Hessian) for laplace approximation of posterior
        # are set to be the same as params of prior 
        w_map, h_map = bayes_logistic.fit_bayes_logistic(
            y = sample_a_y.squeeze(-1),
            X = data_x, 
            wprior = np.array(weights_prior_params[0]), 
            H = np.linalg.inv(np.array(weights_prior_params[1])),
            # H = ((np.identity(num_feats)) * laplace_init_sigma),
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
    # print(samples_a_weights_prior[:2])
    # print(samples_b_weights_prior[:2])
    # print(samples_a_weights_posterior)
    return samples_a_weights_prior, samples_b_weights_prior, \
        samples_a_weights_posterior
    
if __name__ == '__main__':
    run_program()
