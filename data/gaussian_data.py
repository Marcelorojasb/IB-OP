import numpy as np

np.random.seed(42)  
M=100
N=100
K=100
snr=1
h = 20 / K
y = np.arange(-10, 10, h).reshape(-1, 1)
x = np.arange(-10, 10, h).reshape(-1, 1)

q = h * (1 / np.sqrt(2 * np.pi)) * np.exp(-y ** 2 / 2)
q[q < 10 ** (-60)] = 10 ** (-60)
s1=np.diag(q.flatten())@(1/np.sqrt(2*np.pi*(1/snr))*np.exp(-snr*(y@np.ones((1,M))-np.ones((K,1))@x.T)**2/2))*h
p = np.sum(s1, axis=0).reshape(-1, 1)
s = s1 @ np.diag((1. / p).flatten())
s[s < 10 ** (-60)] = 10 ** (-60)
p = p / np.sum(p,axis=0)
q = s @ p

px=p
py_x=s
py=q

pi=p
ski=s
qk=q

# inicialización para GAS 
r0 = np.random.rand(N, 1) + 1
r0 = r0 / np.sum(r0)
w0 = np.random.rand(M, N) + 1
w0 = w0@np.diag((1. / np.sum(w0, axis=0)))
z0=ski@w0@np.diag(r0.flatten())
