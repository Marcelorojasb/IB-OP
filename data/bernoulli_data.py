import numpy as np

np.random.seed(42) 
M=2
N=2
K=2
u=0.3
pi=np.array([[0.5], [0.5]])
I=np.log(2)-(-u*np.log(u)-(1-u)*np.log(1-u))
e=0.15
ski = np.array([[e, 1 - e], [1 - e, e]])
v=(u-e)/(1-2*e)
qk=ski@pi

px=pi
py=qk
py_x=ski

# inicializacion bernoulli GAS
r0 = np.random.rand(N, 1) + 1
r0 = r0 / np.sum(r0)
w0 = np.array([[0.99, 0.01], [0.01, 0.99]])
z0 = np.array([[0.01, 0.49], [0.49, 0.01]])
