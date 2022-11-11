
import numpy as np

np.random.seed(0)

a = np.random.rand(3,2)
a_repeat = np.tile(a, (2,1))
print(a)
print(a.shape)
print(a_repeat)
print(a_repeat.shape)

