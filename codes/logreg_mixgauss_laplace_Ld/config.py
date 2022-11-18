

import argparse

import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('--seed', type=int, default=0)

    args = vars(parser.parse_args())

    args["num_feats"] = 10
    args["num_data"] = 100
    args["num_data_half"] = args['num_data'] // 2
    args["num_samples"] = 1000

    # weights prior distribution parameters
    weights_prior_mu = np.zeros(args["num_feats"])
    weights_prior_cov = np.identity(args["num_feats"])
    args["weights_prior_params"] = [weights_prior_mu, weights_prior_cov]
    
    # init sigma used in numerical optimization for laplace approximation
    args["laplace_init_sigma"] = 0.01
    args["laplace_num_iters"] = 1000

    return args
