import numpy as np

A = np.array([[2, 0], [0, 3]])
a = np.array([2, 5])
print(a.dot(A.T), a.shape)
