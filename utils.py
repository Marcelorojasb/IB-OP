import numpy as np
from ib_algorithms.gas import GAS
import matplotlib.pyplot as plt
from itmetrics import entropy, jEntropy, cEntropy, mutual_information
from ib_algorithms.blahut_arimoto import ba_algorithm

def construct_probability_vector(dim,seed=123):
    np.random.seed(seed)
    P=np.random.rand(dim,1)
    P=P/np.sum(P)
    return(P)

def contruct_probability_conditional(dims,seed=123): #first dim|second dim
    np.random.seed(seed)
    p_conditional=np.random.rand(dims[0],dims[1])
    p_conditional/= np.sum(p_conditional, axis=0, keepdims=True)
    return(p_conditional)



def plot_mutual_information_plane_gas_ba(pi,qk,ski,N,r0,w0,z0,max_iter_gas=400,max_iter_ba=200,num_ib=20,num_betas=10):
    """
    Hace un gráfico de la curva IB para los algoritmos GAS y BA
    """
    ## GAS
    ix_t_list=[]
    iy_t_list=[]
    epsilon=1e-5
    I_max=mutual_information(np.einsum('ki,il->ki', ski, pi),pi,qk)
    for I in np.linspace(epsilon,I_max,num_ib):
        gas_result,i_yt=GAS(pi,qk,ski,I,N,r0,w0,z0,max_iter=max_iter_gas)
        ix_t_list.append(gas_result)
        iy_t_list.append(i_yt)
    plt.scatter(ix_t_list, iy_t_list, color='red', marker='o', facecolors='none', label='GAS Algorithm')

    ##BA

    px=pi
    py_x=ski
    py=qk
    ix_t_list=[]
    iy_t_list=[]
    epsilon=1e-5

    for beta in np.exp(np.linspace(epsilon,20,num_betas)): #  Se busca en un espacio exponencial por posibles betas
        pt_x,pt,py_t=ba_algorithm(py_x,px,dim_N=N,beta=beta,max_iter=max_iter_ba) 
        p_tx = np.einsum('ij,ki->ik', px, pt_x)  # P(X,Y) M,1 , N,M --> N,M
        p_yt = np.einsum('ij,ki->ki', pt, py_t)  # P(T,Y) N,1 , K,N --> K,N
        ix_t_list.append(mutual_information(p_tx,pt,px))
        iy_t_list.append(mutual_information(p_yt,pt,py))

    for beta in (np.linspace(epsilon,100,num_betas)): #  Se busca en un espacio lineal por posibles betas
        pt_x,pt,py_t=ba_algorithm(py_x,px,dim_N=N,beta=beta,max_iter=max_iter_ba)
        p_tx = np.einsum('ij,ki->ik', px, pt_x)  # P(X,Y) M,1 , N,M --> N,M
        p_yt = np.einsum('ij,ki->ki', pt, py_t)  # P(T,Y) N,1 , K,N --> K,N
        ix_t_list.append(mutual_information(p_tx,pt,px))
        iy_t_list.append(mutual_information(p_yt,pt,py))

    for beta in np.log(np.linspace(1+epsilon,100,num_betas)): #  Se busca en un espacio logaritmico por posibles betas
        pt_x,pt,py_t=ba_algorithm(py_x,px,dim_N=N,beta=beta,max_iter=max_iter_ba)
        p_tx = np.einsum('ij,ki->ik', px, pt_x)  # P(X,Y) M,1 , N,M --> N,M
        p_yt = np.einsum('ij,ki->ki', pt, py_t)  # P(T,Y) N,1 , K,N --> K,N
        ix_t_list.append(mutual_information(p_tx,pt,px))
        iy_t_list.append(mutual_information(p_yt,pt,py))
        
    plt.scatter(ix_t_list, iy_t_list, color='blue', marker='+', label='BA Algotihm')
    plt.title('IB curve')
    plt.xlabel('I(X;T)')
    plt.ylabel('I(Y;T)')
    plt.grid(True, color='grey', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.show()


