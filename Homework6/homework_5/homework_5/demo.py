import numpy as np

np.random.seed(0)
p = np.array([0.1, 0.0, 0.7, 0.2])
index = np.random.choice(4, p = p.ravel())
print(p[1])

