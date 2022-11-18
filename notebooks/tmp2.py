import matplotlib.pyplot as plt
import numpy as np
from   numpy.linalg import inv
import numpy.random as npr
from   pypolyagamma import PyPolyaGamma


def sigmoid(x):
    """Numerically stable sigmoid function.
    """
    return np.where(x >= 0, 
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))

def multi_pgdraw(pg, B, C):
    """Utility function for calling `pgdraw` on every pair in vectors B, C.
    """
    for b, c in zip(B, C):
        print(b)
        print(c)
        stop
    return np.array([pg.pgdraw(b, c) for b, c in zip(B, C)])

def gen_bimodal_data(N, p):
    """Generate bimodal data for easy sanity checking.
    """
    y     = npr.random(N) < p
    X     = np.empty(N)
    X[y]  = npr.normal(0, 1, size=y.sum())
    X[~y] = npr.normal(4, 1.4, size=(~y).sum())
    return X, y.astype(int)


# Set priors and create data.
N_train = 1000
N_test  = 1000
b       = np.zeros(2)
B       = np.diag(np.ones(2))
X_train, y_train = gen_bimodal_data(N_train, p=0.3)
X_test, y_test   = gen_bimodal_data(N_test, p=0.3)
print(X_train.shape)
# Prepend 1 for the bias β_0.
X_train = np.vstack([np.ones(N_train), X_train])
X_test  = np.vstack([np.ones(N_test), X_test])
print(X_train.shape)
# print(X_train[:10])
# print(X_test[:10])

# Peform Gibb sampling for T iterations.
pg         = PyPolyaGamma()
T          = 100
Omega_diag = np.ones(N_train)
beta_hat   = npr.multivariate_normal(b, B)
k          = y_train - 1/2.

for _ in range(T):
    # ω ~ PG(1, x*β).
    # tmp = X_train.T @ beta_hat
    # print(tmp.dtype)
    # print(np.ones(N_train).dtype)
    # stop
    Omega_diag = multi_pgdraw(pg, np.ones(N_train), X_train.T @ beta_hat)
    # β ~ N(m, V).
    V         = inv(X_train @ np.diag(Omega_diag) @ X_train.T + inv(B))
    m         = np.dot(V, X_train @ k + inv(B) @ b)
    beta_hat  = npr.multivariate_normal(m, V)

y_pred = npr.binomial(1, sigmoid(X_test.T @ beta_hat))
bins = np.linspace(X_test.min()-3., X_test.max()+3, 100)
plt.hist(X_test.T[y_pred == 0][:, 1],    color='r', bins=bins)
plt.hist(X_test.T[~(y_pred == 0)][:, 1], color='b', bins=bins)
plt.show()
