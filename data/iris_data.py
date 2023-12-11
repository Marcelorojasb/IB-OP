from sklearn.datasets import load_iris
import numpy as np

np.random.seed(42)  

# Cargar el conjunto de datos Iris
iris = load_iris()

data = iris.data  # Obtener las características
target = iris.target  # Obtener las etiquetas

x = np.column_stack([data, target])

x1 = x[:, 0]
x1_a, x1_b = np.min(x1), np.max(x1)

x2 = x[:, 1]
x2_a, x2_b = np.min(x2), np.max(x2)

x3 = x[:, 2]
x3_a, x3_b = np.min(x3), np.max(x3)

x4 = x[:, 3]
x4_a, x4_b = np.min(x4), np.max(x4)

step = 1
n1 = np.ceil((8 - 4) / step)
n2 = np.ceil((4.5 - 2) / step)
n3 = np.ceil((7 - 1) / step)
n4 = np.ceil((2.5 - 0) / step)

len_x = int(n1 * n2 * n3 * n4)
len_y = 3
p_xy = np.zeros((len_x, len_y))

for i in range(len(x1)):
    t1 = np.floor((x[i, 0] - 4) / step) + 1
    t2 = np.floor((x[i, 1] - 2) / step) + 1
    t3 = np.floor((x[i, 2] - 1) / step) + 1
    t4 = np.floor((x[i, 3] - 0.1) / step) + 1
    nx = int(t1 + (t2 - 1) * n1 + (t3 - 1) * n1 * n2 + (t4 - 1) * n1 * n2 * n3)
    ny = int(x[i, 4])
    p_xy[nx, ny] += 1

p_xy = p_xy / len(x1)

cols_que_no_sumen_0 = np.sum(p_xy, axis=1) != 0
P_xy = p_xy[cols_que_no_sumen_0, :]  # quedarse con columnas que no sumen 0

qk = np.sum(P_xy, axis=0).reshape(-1, 1)
pi = np.sum(P_xy, axis=1).reshape(-1, 1)
ski=P_xy.T@np.diag(1./pi.flatten())
ski[ski < 10 ** (-60)] = 10 ** (-60)

M = len(pi)
N = 29
K = 3

py_x=ski
px=pi
py=qk

# Inicialización para GAS

r0 = np.random.rand(N, 1) + 1
r0 = r0 / np.sum(r0)

w0 = np.random.rand(M, N) + 1
w0 = np.diag(1./np.sum(w0.T,axis=0))@w0

z0=ski@w0@np.diag(r0.flatten())