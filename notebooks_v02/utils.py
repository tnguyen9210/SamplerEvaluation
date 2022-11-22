
import math
import numpy as np

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


def compute_kl_div(p, q):
    p = p + 1e-8
    q = q + 1e-8
    return np.sum(p*np.log(p/q), axis=0)
    # return np.sum(np.where(p != 0, p*np.log(p/q), 0))

    
def compute_mmd_rbf(p, q, alpha=0.5):
    num_data, num_feats = p.shape
    
    pp = np.matmul(p, p.T)
    qq = np.matmul(q, q.T)
    pq = np.matmul(p, q.T)
    
    rp = np.repeat(np.diag(pp)[None,:], num_data, axis=0)
    rq = np.repeat(np.diag(qq)[None,:], num_data, axis=0)
    # print(rp[:4,:4])
    # print(rq[:4,:4])
    # print(rp.shape)
    # print(rq.shape)
    # stop
    
    dpp = rp.T + rp - 2*pp
    dqq = rq.T + rq - 2*qq
    dpq = rp.T + rq - 2*pq
    
    kpp = np.zeros(pp.shape)
    kqq = np.zeros(qq.shape)
    kpq = np.zeros(pq.shape)
    
    # bandwidths = [10, 15, 20, 50]
    bandwidths = [1]
    for bw in bandwidths:
        kpp += np.exp(-alpha*dpp/bw)
        kqq += np.exp(-alpha*dqq/bw)
        kpq += np.exp(-alpha*dpq/bw)

    mn = 1./(num_data*(num_data-1)*len(bandwidths))
    mm = 2./(num_data*num_data*len(bandwidths))
    score = mn*np.sum(kpp) - mm*np.sum(kpq) + mn*np.sum(kqq)

    return score


def compute_wasserstein_distance(p, q):
    distances = cdist(p, q)
    assignment = linear_sum_assignment(distances)
    return distances[assignment].sum() / len(p)


def compute_diff_std(p, q, true_mu):
    num_data, num_feats = p.shape
    
    p_std_from_mean = np.sqrt(np.mean(np.sum((p-true_mu)**2, axis=1)))
    q_std_from_mean = np.sqrt(np.mean(np.sum((q-true_mu)**2, axis=1)))

    return np.abs(p_std_from_mean - q_std_from_mean)

