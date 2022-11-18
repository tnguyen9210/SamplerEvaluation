
import numpy as np

num_feats = 2
A = np.random.rand(num_feats, num_feats)
B = np.matmul(A, A.transpose())
print(B)
