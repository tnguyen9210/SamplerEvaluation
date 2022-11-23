
import numpy as np 
import scipy as sp
import scipy.stats as stats

import bayes_logistic
from pypolyagamma import PyPolyaGamma

def sample_laplace_approx(X, y, args):

    weights_prior_params = args["weights_prior_params"]
    laplace_num_iters = args["laplace_num_iters"]
    
    # note:
    # init params (mu and Hessian) for laplace approximation of posterior
    # are set to be the same as params of prior
    
    w_map, h_map = bayes_logistic.fit_bayes_logistic(
        y = y.squeeze(-1),
        X = x, 
        wprior = np.array(weights_prior_params[0]), 
        H = np.linalg.inv(np.array(weights_prior_params[1])),
        weights = None,
        solver = "Newton-CG",
        bounds = None,
        maxiter = laplace_num_iters
    )
    cov_map = np.linalg.inv(h_map)
    sample_w = stats.multivariate_normal.rvs(w_map, cov_map)
    
    return sample_w


def generate_prior_and_posterior_samples(data_x, sampling_method, args):
    
    # params
    num_feats = args["num_feats"]
    num_samples = args["num_samples"]
    
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
        # and sample weights' posterior from the approximation
        sample_a_weights_posterior = sampling_method(
            data_x, sample_a_y, args)
        samples_a_weights_posterior.append(sample_a_weights_posterior)    

    samples_a_weights_prior = np.vstack(samples_a_weights_prior)
    samples_b_weights_prior = np.vstack(samples_b_weights_prior)
    samples_a_weights_posterior = np.vstack(samples_a_weights_posterior)
    # print(samples_a_weights_prior[:2])
    # print(samples_b_weights_prior[:2])
    # print(samples_a_weights_posterior)
    return samples_a_weights_prior, samples_b_weights_prior, \
        samples_a_weights_posterior


def gibb_sample_polyagamma(X, y, args):
    #
    weights_prior_params = args["weights_prior_params"]
    pg_dist = args["pg_dist"]
    burnin_steps = args["pg_burnin_steps"]
    
    num_data, num_feats = X.shape
    y = y.squeeze(-1)
    
    # Gibbs sampling with PG augmentation for burnin_steps
    # init states
    beta_mu = np.array(weights_prior_params[0])
    beta_cov = np.array(weights_prior_params[1])
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
