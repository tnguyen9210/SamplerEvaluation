

import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('--seed', type=int, default=0)

    args = vars(parser.parse_args())

    args["num_feats"] = 1
    args["num_data"] = 100
    args["num_data_half"] = args['num_data'] // 2
    args["num_samples"] = 1000

    args["data_x_marginal_params"] = [
        [-1.0, 1.0],
        [5.0, 2.0]
    ]

    # weights prior distribution parameters
    args["weights_prior_params"] = [0.0, 1.0]

    # init sigma used in numerical optimization for laplace approximation
    args["laplace_init_sigma"] = 0.1
    args["laplace_num_iters"] = 1000

    return args
