import numpy as np
np.random.seed(42) 

px = np.array([[0.25, 0.25, 0.25,0.25]]).T #  distribucion de x M=4  (M,1)
py = np.array([0.25, 0.25, 0.25,0.25]).T # K= 4  distribucion de y   (K,1)
py_x = 1/2*np.array([[1, 1, 0, 0],
                [1, 1, 0, 0],
                [0, 0, 1, 1],
                [0, 0, 1, 1]])

M=4
K=4
N=4
pi=px
qk=py
ski=py_x

# inicializacion Constant slope GAS
r0 = np.random.rand(N, 1) + 1
r0 = r0 / np.sum(r0)
w0 = np.array([[0.4, 0.4, 0.1, 0.1],
                [0.4, 0.4, 0.1, 0.1],
                [0.1, 0.1, 0.4, 0.4],
                [0.1, 0.1, 0.4, 0.4]])
z0 = np.array([[0.1, 0.1, 0.025, 0.025],
                [0.1, 0.1, 0.025, 0.025],
                [0.025, 0.025, 0.1, 0.1],
                [0.025, 0.025, 0.1, 0.1]])