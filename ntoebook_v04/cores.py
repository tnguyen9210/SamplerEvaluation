
import numpy as np 
import scipy as sp
import scipy.stats as stats

import bayes_logistic
from pypolyagamma import PyPolyaGamma

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import pyro
import pyro.distributions as dist
import pyro.distributions.constraints as constraints
from pyro.nn import PyroModule, PyroSample


def generate_prior_and_posterior_samples(data_x, sampling_method, args):
    np.random.seed(0)
    
    # params
    num_feats = args["num_feats"]
    num_samples = args["num_samples"]
    weights_prior_params = args["weights_prior_params"]
    laplace_num_iters = args["laplace_num_iters"]
    
    # generate samples A of weights prior
    samples_a_weights_prior = stats.multivariate_normal.rvs(
        weights_prior_params[0], weights_prior_params[1], 
        size=(num_samples), random_state=args["seed"])

    # generate samples B of weights prior
    samples_b_weights_prior = stats.multivariate_normal.rvs(
        weights_prior_params[0], weights_prior_params[1], 
        size=(num_samples), random_state=args["seed"]+10)

    # generate samples C of weights posterior using samples A of weights prior
    samples_a_weights_posterior = []
    samples_a_weights_posterior_pdf = []
    samples_a_weights_posterior_params = []
    for sidx in range(num_samples):
        # for each sample w_i in A
        sample_a_weights_prior = samples_a_weights_prior[sidx].reshape(1, num_feats)

        # generate sample y_i from Ber(x, w_i)
        sample_a_logit = 1.0 / (1 + np.exp(-np.matmul(data_x, sample_a_weights_prior.T)))
        sample_a_y = stats.bernoulli.rvs(sample_a_logit)

        # fit laplace approximation for pair (x, y_i)
        w_map, h_map = bayes_logistic.fit_bayes_logistic(
            y = sample_a_y.squeeze(-1),
            X = data_x, 
            wprior = np.array(weights_prior_params[0]), # initialize wprior same as prior params
            H = np.linalg.inv(np.array(weights_prior_params[1])),
            weights = None,
            solver = "Newton-CG",
            bounds = None,
            maxiter = laplace_num_iters
        )
        cov_map = np.linalg.inv(h_map)

        # sample weights' posterior p(w|x,y_i)
        sample_a_weights_posterior = stats.multivariate_normal.rvs(w_map, cov_map)
        samples_a_weights_posterior.append(sample_a_weights_posterior)
        
        # sample_a_weights_posterior_pdf = stats.multivariate_normal.pdf(
        #     sample_a_weights_posterior, w_map, cov_map).reshape(1,1)
        # samples_a_weights_posterior_pdf.append(sample_a_weights_posterior_pdf)
        samples_a_weights_posterior_params.append([w_map, cov_map])

    samples_a_weights_prior = np.vstack(samples_a_weights_prior)
    samples_b_weights_prior = np.vstack(samples_b_weights_prior)
    samples_a_weights_posterior = np.vstack(samples_a_weights_posterior)
    # samples_a_weights_posterior_pdf = np.vstack(samples_a_weights_posterior_pdf)

    return samples_a_weights_prior, samples_b_weights_prior, samples_a_weights_posterior

def sample_laplace_approx(X, y, args):

    weights_prior_params = args["weights_prior_params"]
    laplace_num_iters = args["laplace_num_iters"]
    
    # note:
    # init params (mu and Hessian) for laplace approximation of posterior
    # are set to be the same as params of prior
    
    w_map, h_map = bayes_logistic.fit_bayes_logistic(
        y = y.squeeze(-1),
        X = X, 
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



def gibbs_sample_polyagamma(X, y, args):
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


def sample_svi_approx(X, y, args):
    
    logreg_model = args["logreg_model"]
    logreg_guide = args["logreg_guide"]
    svi_num_iters = args["svi_num_iters"]

    num_data, num_feats = X.shape
    y = torch.tensor(y, dtype=torch.float).squeeze(-1)
    
    # sample weights' posterior using SVI
    # diagonal_normal_guide = pyro.infer.autoguide.AutoDiagonalNormal(logistic_regression_model)
    optim = pyro.optim.Adam({"lr": 0.03})
    svi = pyro.infer.SVI(
        logreg_model, logreg_guide, optim, loss=pyro.infer.Trace_ELBO())
    
    pyro.clear_param_store()
    for it in range(svi_num_iters):
        # calculate the loss and take a gradient step
        loss = svi.step(X, y)
        # if i % 100 == 0:
        #     print("[iteration %04d] loss: %.4f" % (i, loss / len(data_x)))
        
    predictive_fn = pyro.infer.Predictive(
        logreg_model, guide=logreg_guide, num_samples=1)

    svi_samples = predictive_fn(X, y)
    sample_w = svi_samples["w"].reshape(1, num_feats)

    return sample_w

