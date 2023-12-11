import numpy as np
from itmetrics import kl_divergence, mutual_information
import matplotlib.pyplot as plt


def contruct_probability_conditional(dims,seed=123): #first dim|second dim
    np.random.seed(seed)
    p_conditional=np.random.rand(dims[0],dims[1])
    p_conditional/= np.sum(p_conditional, axis=0, keepdims=True)
    return(p_conditional)

def calculate_column_pt_x(pt, py_t, py_x_col, beta):
    kl_matrix=kl_divergence_matrix(py_x_col,py_t)
    pt_x_column = pt * np.exp(-beta * kl_matrix)
    pt_x_column /= np.sum(pt_x_column,axis=0,keepdims=True)
    return pt_x_column

def kl_divergence_matrix(py_x_col,py_t):
    N=py_t.shape[1]
    kl_matrix=np.zeros((N,1))
    for j in range(N):
        kl_matrix[j,0]=kl_divergence(py_x_col,py_t[:,j])
    return(kl_matrix)


def calculate_pt_x(pt,py_t,py_x,beta):
    N=pt.shape[0]
    M=py_x.shape[1]
    pt_x=np.zeros((N,M))
    for i in range(M):
        pt_x[:,i]=calculate_column_pt_x(pt,py_t,py_x[:,i],beta)[:,0] # N,1
    return(pt_x) #N,M


def calculate_py_t(pt, py_x, pt_x, px):
    result_1=pt_x@np.diag(px.flatten())
    result=py_x@(result_1.T)
    py_t=result/pt.reshape(1,-1)  
    return py_t

def BA_iteration(pt,py_t,py_x,px,beta):
    pt_x=calculate_pt_x(pt,py_t,py_x,beta)  # N,M
    pt=(pt_x@px).reshape(-1,1) # N,M @ M,1 = N,1
    py_t=calculate_py_t(pt,py_x,pt_x,px) # K,N
    return(pt_x,pt,py_t)

def ba_algorithm(py_x,px,dim_N,beta,max_iter=1000):

    # Iniciar P0(T|X)
    M=px.shape[0]
    pt_x=contruct_probability_conditional(dims=(dim_N,M)) # P(T|X) N,M
    pt=np.einsum('ij,ki->kj', px, pt_x) # P(T) M,1 , N,M --> N,1
    py_t=calculate_py_t(pt,py_x,pt_x,px)

    for i in range(max_iter):
        pt_x,pt,py_t=BA_iteration(pt,py_t,py_x,px,beta)
    return(pt_x,pt,py_t)


def plot_mutual_information_plane_ba(py_x,px,py,dim_N=2,num_betas=40,max_iter=1000):
    ix_t_list=[]
    iy_t_list=[]
    epsilon=1e-5
    for beta in np.exp(np.linspace(epsilon,20,num_betas)): # Se busca en el exp space por posibles betas
        pt_x,pt,py_t=ba_algorithm(py_x,px,dim_N=dim_N,beta=beta,max_iter=max_iter)
        p_tx = np.einsum('ij,ki->ik', px, pt_x)  # P(X,Y) M,1 , N,M --> N,M
        p_yt = np.einsum('ij,ki->ki', pt, py_t)  # P(T,Y) N,1 , K,N --> K,N
        ix_t_list.append(mutual_information(p_tx,pt,px))
        iy_t_list.append(mutual_information(p_yt,pt,py))
        
    for beta in (np.linspace(epsilon,100,num_betas)): #  Se busca en un espacio lineal por posibles betas
        pt_x,pt,py_t=ba_algorithm(py_x,px,dim_N=dim_N,beta=beta,max_iter=max_iter)
        p_tx = np.einsum('ij,ki->ik', px, pt_x)  # P(X,Y) M,1 , N,M --> N,M
        p_yt = np.einsum('ij,ki->ki', pt, py_t)  # P(T,Y) N,1 , K,N --> K,N
        ix_t_list.append(mutual_information(p_tx,pt,px))
        iy_t_list.append(mutual_information(p_yt,pt,py))

    for beta in np.log(np.linspace(1+epsilon,100,num_betas)): # Se busca en un espacio logaritmico por posibles betas
        pt_x,pt,py_t=ba_algorithm(py_x,px,dim_N=dim_N,beta=beta,max_iter=max_iter)
        p_tx = np.einsum('ij,ki->ik', px, pt_x)  # P(X,Y) M,1 , N,M --> N,M
        p_yt = np.einsum('ij,ki->ki', pt, py_t)  # P(T,Y) N,1 , K,N --> K,N
        ix_t_list.append(mutual_information(p_tx,pt,px))
        iy_t_list.append(mutual_information(p_yt,pt,py))
    plt.scatter(ix_t_list, iy_t_list, color='blue', marker='o', label='Scatter Plot')
    plt.title('IB curve')
    plt.xlabel('I(X;T)')
    plt.ylabel('I(Y;T)')
    plt.grid(True, color='grey', linestyle='--', linewidth=0.5)
    plt.show()