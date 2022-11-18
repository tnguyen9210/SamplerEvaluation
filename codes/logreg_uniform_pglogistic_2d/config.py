

import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('--seed', type=int, default=0)

    args = vars(parser.parse_args())

    args["num_feats"] = 2
    args["num_data"] = 2
    args["num_data_half"] = args['num_data'] // 2
    args["num_samples"] = 1000

    args["data_x_type"] = "uniform"
    args["data_x_marginal_params"] = [
        [[1.0, 5.0], [(1.0, 0.0), (0.0, 1.0)]],
        [[-5.0, 1.0], [(3.0, 0.0), (0.0, 3.0)]]]

    # weights prior distribution parameters
    args["weights_prior_params"] = [[0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]]]

    args["pglogistic_burnin_steps"] = 100

    return args
