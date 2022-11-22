# Import packages
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

color_list = ["Green", "Blue"]
color_map = mcolors.ListedColormap(["Green", "Blue"])

import seaborn as sns
sns.set()
sns.set_palette("tab10")

import random as rnd
import numpy as np
# rnd.seed(0)
# np.random.seed(0)

import scipy.stats as stats
import bayes_logistic
from pypolyagamma import PyPolyaGamma


# Set parameters
# num_data
num_data = 2
num_data_half = num_data // 2
num_feats = 2

# num_samples 
num_samples = 1000

# data_x marginal distribution parameters
data_x_marginal_params = [
    [[1.0, 5.0], [(1.0, 0.0), (0.0, 1.0)]],
    [[-5.0, 1.0], [(3.0, 0.0), (0.0, 3.0)]]]

# weights prior distribution parameters
weights_prior_params = [[0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]]]

# init sigma used in numerical optimization for laplace approximation
laplace_init_sigma = 0.01
laplace_num_iters = 1000

data_x = np.random.randn(num_data, num_feats)
print(data_x.shape)

### Generate prior and posterior samples

# weights' prior distribution
weights_prior_dist_a = stats.multivariate_normal(
    weights_prior_params[0], weights_prior_params[1], seed=1)
weights_prior_dist_b = stats.multivariate_normal(
    weights_prior_params[0], weights_prior_params[1], seed=11)


def pg_inference(X, y, burnin_steps=200):
    #
    num_data, num_feats = X.shape

    # Gibbs sampling with PG augmentation for burnin_steps
    # init states
    beta_mu = np.zeros(num_feats)
    beta_cov = np.diag(np.ones(num_feats))
    beta_hat = stats.multivariate_normal.rvs(beta_mu, beta_cov)
    # beta_hat = np.random.multivariate_normal(beta_mu, beta_cov)
    k = y - 1/2

    pg = PyPolyaGamma(seed=1)
    # perform Gibbs sampling
    for bid in range(burnin_steps+1):
        # print(f"\n-> {bid}")
        # ω ~ PG(1, x*β).
        omega_b = np.ones(num_data)
        # print(X.shape)
        # print(beta_hat.shape)
        omega_c = X @ beta_hat
        # omega_c = np.matmul(X, beta_hat)
        omega_diag = np.array(
            [PyPolyaGamma().pgdraw(b, c) for b, c in zip(omega_b, omega_c)])

        # β ~ N(m, V).
        V = np.linalg.inv(X.T @ np.diag(omega_diag) @ X + np.linalg.inv(beta_cov))
        # first_term = X.T @ k
        # second_term = np.linalg.inv(beta_cov) @ beta_mu
        # tmp = X.T @ k + np.linalg.inv(beta_cov) @ beta_mu
        # print()
        # print(first_term.shape)
        # print(second_term.shape)
        # print(tmp.shape)
        # stop
        m = np.dot(V, X.T @ k + np.linalg.inv(beta_cov) @ beta_mu)
        # beta_hat  = stats.multivariate_normal.rvs(m, V)
        # print(m.shape)
        # print(V.shape)
        beta_hat = np.random.multivariate_normal(m, V)

    return beta_hat

samples_a_weights_prior = []
samples_b_weights_prior = []
samples_a_weights_posterior = []
for i in range(num_samples):
    # sample two set of weights' priors 
    sample_a_weights_prior = weights_prior_dist_a.rvs(1)[None,:]
    sample_b_weights_prior = weights_prior_dist_b.rvs(1)[None,:]
    samples_a_weights_prior.append(sample_a_weights_prior)
    samples_b_weights_prior.append(sample_b_weights_prior)
    # print(sample_a_weights_prior.shape)
    # print(sample_b_weights_prior.shape)
    
    # generate sample y_i from theta_i in A
    sample_a_logit = 1.0 / (1 + np.exp(-np.matmul(data_x, sample_a_weights_prior.T)))
    sample_a_y = stats.bernoulli.rvs(sample_a_logit).squeeze(-1)
    # print(data_x.shape)
    # print(sample_a_y.shape)
    
    # sample weights' posterior
    # sample_a_weights_posterior = stats.multivariate_normal.rvs(w_map, cov_map)
    sample_a_weights_posterior = pg_inference(data_x, sample_a_y, burnin_steps=200)
    samples_a_weights_posterior.append(sample_a_weights_posterior)    

samples_a_weights_prior = np.vstack(samples_a_weights_prior)
samples_b_weights_prior = np.vstack(samples_b_weights_prior)
samples_a_weights_posterior = np.vstack(samples_a_weights_posterior)
print(samples_a_weights_prior.shape)
# print(samples_b_weights_prior)
# print(samples_a_weights_posterior)

# Visualize the generated prior and posterior samples 
nrows = 2
fig, axes = plt.subplots(nrows=nrows, ncols=1, sharex=True, sharey=True, figsize=(10,10))
axes = axes.flatten()

for i in range(nrows):
    sns.kdeplot(samples_a_weights_prior[:,0], fill=False, color="blue", label="sample_a_prior", ax=axes[i])
    sns.kdeplot(samples_b_weights_prior[:,0], fill=False, color="green", label="sample_b_prior", ax=axes[i])
    sns.kdeplot(samples_a_weights_posterior[:,0], fill=False, color="orange", label="sample_a_posterior", ax=axes[i])
    axes[i].legend()
plt.show()
