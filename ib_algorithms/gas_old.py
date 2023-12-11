import numpy as np
from scipy import optimize
from itmetrics import entropy, jEntropy, cEntropy, mutual_information
import matplotlib.pyplot as plt

# primer intento de implementacion algoritmo GAS 

def LAMBDA_MATRIX(ski,lambda_matrix,zeta,z_matrix):
    """
    Calcula matrix Lambda mayuscula dim=(M,N)
    """
    M=ski.shape[1]
    K=ski.shape[0]
    N=lambda_matrix.shape[1]
    L=np.zeros((M,N))
    for i in range(M):
        for j in range(N):
            sum=np.sum(ski[:,i]*(lambda_matrix[:,j]-zeta*np.log(z_matrix[:,j])))  # suma sobre matriz de K,1
            L[i,j]=np.exp(-sum)
    return(L)

def a_from_phi(phi_matrix): #(M,1)
    """
    Obtiene el vector a a partir de phi
    """
    ai=-np.log(phi_matrix)-1/2
    return(ai)  #(M,1)

def b_from_psi(psi_matrix): #(N,1)
    """
    Obtiene el vector b a partir de psi
    """
    bj=-np.log(psi_matrix)-1/2
    return(bj)  #(N,1)
    

def update_rj_aux(zeta,w_matrix,ski,z_matrix,phi_matrix,psi_matrix,lambda_matrix):
    """
    Hace el update al vector rj segun como se indica en el paper:
    W. Y. H. W. H. W. W. Z. B. B. L. Chen, S. Wu and Y. Sun, “Information bottleneck revisited: Posterior probability perspective with optimal transport
    """
    ai=a_from_phi(phi_matrix)
    bj=b_from_psi(psi_matrix)
    M=ski.shape[1]
    K=ski.shape[0]
    N=psi_matrix.shape[0]
    sum_M=np.zeros((N,1))
    for i in range(M):   # comprobar que funcione, si no irse a la segura con el for
        sum_M+=(w_matrix[i,:].reshape(-1,1))*(np.log(w_matrix[i,:]).reshape(-1, 1))+(ai[i,:].reshape(-1,1))*(w_matrix[i,:].reshape(-1, 1))+bj[:,:]*(w_matrix[i,:].reshape(-1, 1))-bj[:,:]  # revisar los que aparezca el j
        sum_k=0
        for k in range(K):
            sum_k+=(ski[k,i]*(w_matrix[i,:].reshape(-1,1)))*(-zeta*(np.log(z_matrix[k,:]).reshape(-1,1))+(lambda_matrix[k,:].reshape(-1,1)))
        sum_M+=sum_k
    result=(-1/zeta)*sum_M-1    # (N,1)
    rj_aux=np.exp(result)     # (N,1)
    return(rj_aux)


def G_find_zeros(zeta,r_matrix,phi_matrix,ski,lambda_matrix,z_matrix,psi_matrix,qk,I):
    """
    Función G(zeta), para buscar raices positivas 
    """
    M=ski.shape[1]
    K=ski.shape[0]
    N=lambda_matrix.shape[1]
    I_prime=I+np.sum(qk*np.log(qk))
    G=-np.sum(r_matrix*np.log(r_matrix))-I_prime
    sum_n_k=0
    for j in range(N):
        for k in range(K):
            sum_M=0
            for i in range(M):
                sum_k_prime=0
                for k_prime in range(K):
                    sum_k_prime+=ski[k_prime,i]*(lambda_matrix[k_prime,j]-zeta*np.log(z_matrix[k_prime,j]))
                sum_M+=ski[k,i]*phi_matrix[i,0]*np.exp(-sum_k_prime)    # aca sale que hay overflow
            sum_n_k+=sum_M*psi_matrix[j,0]*r_matrix[j,0]*np.log(z_matrix[k,j])
    G+=sum_n_k
    return(G)


def find_positive_roots(g):
    """
    Busca raices positivas en una función con el método de newton
    """
    epsilon=1e-5
    for i in np.exp(np.linspace(epsilon,20,1000)):  # busca en este rango las raices 
        sol=optimize.newton(g,i,disp=False)
        if sol>0:
            return(sol)
    return(None)

def calculate_min_GAS(phi_matrix,psi_matrix,L,r_matrix):  # calcula el minimo del algoritmo GAS 
    """
    Calcula el return del algoritmo GAS, corresponde al funcional que se esta minimizando.
    """
    M=phi_matrix.shape[0]
    N=psi_matrix.shape[0]
    sum=0
    for i in range(M):
        for j in range(N):
            sum+=phi_matrix[i,0]*L[i,j]*psi_matrix[j,0]*r_matrix[j,0]*np.log(phi_matrix[i,0]*L[i,j]*psi_matrix[j,0])
    return(sum)


def GAS(pi,qk,ski,I,N,max_iter):
    """
    Implementación del algoritmo GAS, retorna el valor mínimo del funcional a minimizar.
    """
    M = len(pi)  # alfabeto de X
    K = len(qk) # alfabeto de Y

    phi_matrix=np.ones((M,1))  # matriz de 1´s de M,1
    psi_matrix=np.ones((N,1))  # matriz de 1´s de N,1
    zeta,eta=1,1
    lambda_matrix=-1*np.ones((K,N))# matriz de -1´s K,N

    r_matrix= np.ones((N,1))/N # matriz de 1´s/N N,1 dim
    w_matrix= np.ones((M, N))/M # matriz de 1´s/M M,N dim
    z_matrix= np.ones((K,N))/(K*N)#  matriz de 1´s/KN K,N dim
    L=LAMBDA_MATRIX(ski,lambda_matrix,zeta,z_matrix) # corresponde a matriz Lambda mayuscula

    for i in range(max_iter):
        psi_matrix=np.divide(np.ones((N,1)),L.T@phi_matrix) # matriz psi
        phi_matrix=np.divide(pi,(L)@(psi_matrix*r_matrix)) # matriz phi

        g = lambda x: G_find_zeros(x,r_matrix,phi_matrix,ski,lambda_matrix,z_matrix,psi_matrix,qk,I) # funcion g(zeta)
        zeta=find_positive_roots(g) # busca raices

        L=LAMBDA_MATRIX(ski,lambda_matrix,zeta,z_matrix) # actualiza Lambda mayuscula

        w_matrix=np.einsum('ij,jl,il->ij',L,psi_matrix,phi_matrix) # actualiza w

        ##
        #w_matrix/=np.sum(w_matrix,axis=0) # actualiza w
        ##

        lambda_matrix=-zeta*np.ones((K,N)) # actualiza Lambda 
        z_matrix=np.einsum('ij,jl,ki->kj',w_matrix,r_matrix,ski) # actualiza z
        
        ##
       # z_matrix/=np.sum(z_matrix) # actualiza z
        ##
        
        rj_aux=update_rj_aux(zeta,w_matrix,ski,z_matrix,phi_matrix,psi_matrix,lambda_matrix) # actualiza r barra
        r_matrix=rj_aux/np.sum(rj_aux) # actualiza r
    ## DEBUG
    errores=manejo_de_errores(phi_matrix,psi_matrix,zeta,lambda_matrix,L,r_matrix,w_matrix,z_matrix,qk,I,pi,ski)
    print(errores)
    ## DEBUG    
    return(calculate_min_GAS(phi_matrix,psi_matrix,L,r_matrix),calculate_I_yt(w_matrix,r_matrix,pi,ski,z_matrix,qk),w_matrix)  # calcula salida de algoritmo

    
def LAMBDA_MATRIX_no_overflow(ski,lambda_matrix,zeta,z_matrix):
    """
    Calcula matrix Lambda mayuscula dim=(M,N)
    """
    M=ski.shape[1]
    K=ski.shape[0]
    N=lambda_matrix.shape[1]
    L=np.zeros((M,N))
    zjmax_matrix=zj_max(ski,z_matrix) # (N,1)
    for i in range(M):
        for j in range(N):
            sum=np.sum(ski[:,i]*(lambda_matrix[:,j]-zeta*np.log(z_matrix[:,j])))  # suma sobre matriz de K,1
            sum-=zeta*zjmax_matrix[j,0]
            L[i,j]=np.exp(-sum)
    return(L)


def zj_max(ski,z_matrix):
    z_max=np.einsum('ki,kj->ij', ski, np.log(z_matrix)) # K,M * K,N = M,N
    return(np.max(z_max,axis=0).reshape(-1,1)) # M,N-> N,1

def GAS_no_overflow(pi,qk,ski,I,N,max_iter):
    """
    Implementación del algoritmo GAS, retorna el valor mínimo del funcional a minimizar.
    """
    M = len(pi)  # alfabeto de X
    K = len(qk) # alfabeto de Y

    phi_matrix=np.ones((M,1))  # matriz de 1´s de M,1
    psi_matrix=np.ones((N,1))  # matriz de 1´s de N,1
    zeta,eta=1,1
    lambda_matrix=-1*np.ones((K,N))# matriz de -1´s K,N

    r_matrix= np.ones((N,1))/N # matriz de 1´s/N N,1 dim
    w_matrix= np.ones((M, N))/M # matriz de 1´s/M M,N dim
    z_matrix= np.ones((K,N))/(K*N)#  matriz de 1´s/KN K,N dim
    L=LAMBDA_MATRIX_no_overflow(ski,lambda_matrix,zeta,z_matrix) # corresponde a matriz Lambda mayuscula

    for i in range(max_iter):
        psi_matrix=np.divide(np.ones((N,1)),L.T@phi_matrix) # matriz psi
        phi_matrix=np.divide(pi,(L)@(psi_matrix*r_matrix)) # matriz phi

        psi_matrix_subs=psi_matrix*np.exp(-zeta*zj_max(ski,z_matrix))
        g = lambda x: G_find_zeros_2(x,r_matrix,phi_matrix,ski,lambda_matrix,z_matrix,psi_matrix_subs,qk,I) # funcion g(zeta)
        zeta=find_positive_roots(g) # busca raices

        L=LAMBDA_MATRIX_no_overflow(ski,lambda_matrix,zeta,z_matrix) # actualiza Lambda mayuscula

        w_matrix=np.einsum('ij,jl,il->ij',L,psi_matrix,phi_matrix) # actualiza w
        lambda_matrix=-zeta*np.ones((K,N)) # actualiza Lambda 
        z_matrix=np.einsum('ij,jl,ki->kj',w_matrix,r_matrix,ski) # actualiza z
        psi_matrix_subs=psi_matrix*np.exp(-zeta*zj_max(ski,z_matrix))

        rj_aux=update_rj_aux(zeta,w_matrix,ski,z_matrix,phi_matrix,psi_matrix_subs,lambda_matrix) # actualiza r barra
        r_matrix=rj_aux/np.sum(rj_aux) # actualiza r
    return(calculate_min_GAS(phi_matrix,psi_matrix,L,r_matrix))  # calcula salida de algoritmo

#TEST
def update_rj_aux_2(zeta,w_matrix,ski,z_matrix,phi_matrix,psi_matrix,lambda_matrix):
    """
    Hace el update al vector rj segun como se indica en el paper:
    W. Y. H. W. H. W. W. Z. B. B. L. Chen, S. Wu and Y. Sun, “Information bottleneck revisited: Posterior probability perspective with optimal transport
    """
    ai=a_from_phi(phi_matrix)
    bj=b_from_psi(psi_matrix)
    M=ski.shape[1]
    K=ski.shape[0]
    N=psi_matrix.shape[0]
    rj_aux=np.zeros((N,1))
    for j in range(N):
        sum_M=np.sum(w_matrix[:,j]*np.log(w_matrix[:,j])+ai[:,:]*w_matrix[:,j]+bj[j,:]*w_matrix[:,j]-bj[j,:])
        for i in range(M):
            sum_K=0.0
            for k in range(K):
                sum_K+=-zeta*ski[k,i]*w_matrix[i,j]*np.log(z_matrix[k,j])+ski[k,i]*w_matrix[i,j]*lambda_matrix[k,j]
            sum_M+=sum_K
        rj_aux[j,:]=np.exp(-(1/zeta)*sum_M-1)
    return(rj_aux)

def G_find_zeros_2(zeta,r_matrix,phi_matrix,ski,lambda_matrix,z_matrix,psi_matrix,qk,I):
    """
    Función G(zeta), para buscar raices positivas 
    """
    M=ski.shape[1]
    K=ski.shape[0]
    N=lambda_matrix.shape[1]
    I_prime=I+np.sum(qk*np.log(qk))
    G=0.0
    G=-np.sum(r_matrix*np.log(r_matrix))-I_prime
    sum_NK=0.0
    for j in range(N):
        for k in range(K):
            sum_M=0.0
            for i in range(M):
                sum_k_prime=0.0
                for k_prime in range(K):
                    sum_k_prime+=ski[k_prime,i]*(lambda_matrix[k_prime,j]-zeta*np.log(z_matrix[k_prime,j]))
                sum_k_prime=np.exp(-sum_k_prime)
                sum_M+=phi_matrix[i,:]*ski[k,i]*sum_k_prime
            sum_NK+=sum_M*psi_matrix[j,:]*r_matrix[j,:]*np.log(z_matrix[k,j])
    return(sum_NK+G)

def plot(g,zeta):
    xpts = np.linspace(0, 10, 1000)
    plt.plot(xpts, g(xpts))
    plt.annotate(zeta,(zeta,0))
    plt.annotate(g(zeta),(zeta,-4))
    plt.show()
    pass

# Construct probs

def contruct_probability_conditional(dims,seed=123): #first dim|second dim
    np.random.seed(seed)
    p_conditional=np.random.rand(dims[0],dims[1])
    p_conditional/= np.sum(p_conditional, axis=0, keepdims=True)
    return(p_conditional)





# plot

def plot_mutual_information_plane_gas(pi,qk,ski,N):
    ix_t_list=[]
    iy_t_list=[]
    epsilon=1e-5
    for I in np.linspace(epsilon,0.2,10):
        gas_result,i_yt,w=GAS(pi,qk,ski,I,N,max_iter=100)
        information_xt=entropy(pi)+gas_result   # gas result es la enrtopia?
        print(w)
        ix_t_list.append(information_xt)
        iy_t_list.append(i_yt)
    plt.scatter(ix_t_list, iy_t_list, color='blue', marker='o', label='Scatter Plot')
    plt.title('IB curve')
    plt.xlabel('I(X;T)')
    plt.ylabel('I(Y;T)')
    plt.grid(True, color='grey', linestyle='--', linewidth=0.5)
    plt.show()

def plot_mutual_information_plane_gas_no_of(pi,qk,ski,N,max_iter=100):
    ix_t_list=[]
    iy_t_list=[]
    epsilon=1e-5
    for I in np.linspace(epsilon,0.66,6):
        gas_result=GAS_no_overflow(pi,qk,ski,I,N,max_iter=max_iter)
        information_xt=entropy(pi)+gas_result   # gas result es la enrtopia?
        ix_t_list.append(information_xt)
        iy_t_list.append(I)
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


def manejo_de_errores(phi_matrix,psi_matrix,zeta,lambda_matrix,L,r_matrix,w_matrix,z_matrix,qk,I,pi,ski):
    error_list=[]
    #phi
    if np.any(phi_matrix < 0):
        error_list.append("phi")
    #psi
    if np.any(psi_matrix < 0):
        error_list.append("psi")
    #zeta
    if zeta<0:
        error_list.append("zeta")
    #L
    if np.any(L < 0):
        error_list.append("L")
    # r_matrix
    if np.any(r_matrix < 0):
        error_list.append("r_0")
    if np.sum(r_matrix)!=1:
        error_list.append("r_1")
    # w_matrix
    if np.any(w_matrix < 0):
        error_list.append("w_0")
    if np.all(np.sum(w_matrix,axis=0)!=1):
        error_list.append("w_1")
        print(w_matrix)
    # z_matrix
    if np.any(z_matrix < 0):
        error_list.append("z_0")
    if np.sum(z_matrix)!=1:
        print(z_matrix)
        error_list.append("z_1")
    if np.any(np.sum(z_matrix,axis=1)!=qk):
        print(z_matrix)
        error_list.append("z_2")
    if np.any(np.sum(z_matrix,axis=0)!=r_matrix):
        print(z_matrix,r_matrix)
        error_list.append("z_3")

    delta_i_y_t=calculate_I_yt(w_matrix,r_matrix,pi,ski,z_matrix,qk)-I
    if delta_i_y_t<0:
        error_list.append("i_y_t")
    print(delta_i_y_t)

    return(error_list)