import numpy as np 
import matplotlib.pyplot as plt
from itmetrics import entropy, jEntropy, cEntropy, mutual_information


def plot_mutual_information_plane_gas(pi,qk,ski,N,r0,w0,z0,max_iter=400,num_ib=20):
    ix_t_list=[]
    iy_t_list=[]
    epsilon=1e-5
    I_max=mutual_information(np.einsum('ki,il->ki', ski, pi),pi,qk)
    for I in (np.linspace(epsilon,I_max,num_ib)):
        gas_result,i_yt=GAS(pi,qk,ski,I,N,r0,w0,z0,max_iter=max_iter)
        ix_t_list.append(gas_result)
        iy_t_list.append(i_yt)
    plt.scatter(ix_t_list, iy_t_list, color='blue', marker='o', label='Scatter Plot')
    plt.title('IB curve')
    plt.xlabel('I(X;T)')
    plt.ylabel('I(Y;T)')
    plt.grid(True, color='grey', linestyle='--', linewidth=0.5)
    plt.show()

def calculate_u_ij(w_matrix,r_matrix,pi):
    joint=np.einsum('ij,jk->ij',w_matrix,r_matrix)
    uij=joint/pi
    return(uij) # M,N

def calculate_I_yt(w_matrix,r_matrix,pi,ski,z_matrix,qk):
    """
    Calcula información mutua I(Y;T)
    """
    uij=calculate_u_ij(w_matrix,r_matrix,pi)
    sum_km=0.0
    N=z_matrix.shape[1]
    M=pi.shape[0]
    K=ski.shape[0]
    for j in range(N):
        for k in range(K):
            for i in range(M):
                sum_km+=uij[i,j]*ski[k,i]*pi[i,0]*np.log(z_matrix[k,j])
    I_y_t=sum_km+entropy(r_matrix)+entropy(qk)
    return(I_y_t)


def GAS(pi,qk,ski,I,N,r0,w0,z0,max_iter=400):
    """
    Algoritmo GAS
    """
    I_max=mutual_information(np.einsum('ki,il->ki', ski, pi),pi,qk)
    M = len(pi)  # alfabeto de X
    K = len(qk) # alfabeto de Y
    ##GAS 
    g = np.ones((N,1)) / N  # (N,1)
    f = np.ones((M,1)) / M  # (M,1)
    R = np.zeros((M, N))    # (M,N)
    r=r0
    w=w0
    z=z0
    zeta=1
    lambda_matrix=-zeta*np.ones((K,N))   # (K,N)
    G=0
    I_prime=I+np.sum(np.where(qk != 0, qk*np.log(qk), 0))      

    R1 = -ski.T @ lambda_matrix  # M,K @ K,N --> M,N
    R2 = ski.T @ np.log(z)
    for j in range(max_iter):

        R1 = -ski.T @ lambda_matrix # R1
        R2 = ski.T @ np.log(z)
        R = R1 + zeta * (R2 - np.ones((M, 1)) @ (np.max(R2,axis=0).reshape(1,-1)))  # M,N

        g+=-(np.log(np.sum(np.exp(R+f@np.ones((1,N))+np.ones((M,1))@g.T),axis=0))).reshape(-1,1) # actualiza g
        f+= np.log(pi)  - np.log(np.exp(R+f@np.ones((1,N))+np.ones((M,1))@g.T)@r) # actualiza f
        r[r < 1e-60] = 1e-60  # para evitar problemas con ceros
        c = I_prime - entropy(r)

        zeta=find_zeta_G(ski,R1,zeta,R2,M,f,N,g,r,w,c)
        if zeta<0: 
            break
        w= np.exp(R+f@np.ones((1,N))+np.ones((M,1))@(g.T)) # actualiza w
        lambda_matrix = -zeta * np.ones((K, N)) # actualiza lambda_matrix
        z = ski@w@np.diag(r.flatten())  # actualiza z
        z[z < 1e-60] = 1e-60 # para evitar problemas con ceros

        ww0 = w.copy() 
        ww0[ww0 < 1e-30] = 1 # para evitar problemas con ceros
        a = (np.sum((ski@w)*np.log(z), axis=0)).reshape(-1,1)-(np.sum(ww0*np.log(ww0),axis=0).reshape(-1,1))/zeta-((np.sum(w@np.diag((-g+zeta*(np.max(R2, axis=0).reshape(-1,1))-0.5).flatten()),axis=0).T)/zeta).reshape(-1,1)+(-g+zeta*np.max(R2, axis=0).reshape(-1,1)-0.5)/zeta-np.sum((ski@w)*lambda_matrix,axis=0).reshape(-1,1)/zeta-1 # para actualizar el r 
    
        r=np.exp(a-np.max(a, axis=0).reshape(1,-1)) # actualiza r
        r=r/np.sum(r, axis=0)

        ww= w@np.diag(r.flatten())
        ww[ww < 1e-30] = 1 # para evitar problemas con ceros

    value=np.sum(np.sum(ww*np.log(ww),axis=0),axis=0)+entropy(r)+entropy(pi) # calcula información mutua I(X;T)
    I_yt=calculate_I_yt(w,r,pi,ski,z,qk) # calcula información mutua I(Y;T)

    if value<0 or I_yt>I_max or I_yt<0:  # control de flujo para que tengan sentido los resultados
        return(np.nan,np.nan)
    return(value,I_yt)


def find_zeta_G(ski,R1,zeta,R2,M,f,N,g,r,w,c):
    """
    Función para encontrar ceros en función G(zeta)
    """
    i=0
    G=np.sum(np.sum((ski@(np.exp(R1+zeta*(R2-np.ones((M,1))@(np.max(R2, axis=0).reshape(1,-1)))+f@np.ones((1,N))+np.ones((M,1))@(g.T)))@np.diag(r.flatten()))*np.log(ski@w@np.diag(r.flatten())),axis=0),axis=0)-c
    while i < 20 and abs(G) > 1e-16:
        i += 1
        zeta=zeta-(G)/(np.sum(np.sum((ski @ (np.exp(R1 + zeta * (R2 - np.ones((M, 1))@(np.max(R2, axis=0).reshape(1,-1)))+f@np.ones((1,N))+np.ones((M,1))@(g.T))*(R2 - np.ones((M, 1))@(np.max(R2, axis=0).reshape(1,-1))))@np.diag(r.flatten()))*(np.log(ski@w@np.diag(r.flatten()))),axis=0),axis=0))
        G=np.sum(np.sum((ski@(np.exp(R1+zeta*(R2-np.ones((M,1))@(np.max(R2, axis=0).reshape(1,-1)))+f@np.ones((1,N))+np.ones((M,1))@(g.T)))@np.diag(r.flatten()))*np.log(ski@w@np.diag(r.flatten())),axis=0),axis=0)-c
    return(zeta)