import numpy as np

px = np.array([[0.25, 0.25, 0.25,0.25]]).T #  distribucion de x M=4  (M,1)
py = np.array([0.25, 0.25, 0.25,0.25]).T # K= 4  distribucion de y   (K,1)
py_x = 1/2*np.array([[1, 1, 0, 0],
                [1, 1, 0, 0],
                [0, 0, 1, 1],
                [0, 0, 1, 1]])