import numpy as np
from scipy import optimize

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
            sum=0
            for k in range(K):
                sum+=ski[k,i]*(lambda_matrix[k,j]-zeta*np.log(z_matrix[k,j]))
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
    for i in np.exp(np.linspace(1,20,1000)):  # busca en este rango las raices 
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


def entropy(p):
    """
    Calcula la entropía de una distribución de probabilidad
    """
    entropy=np.sum(-p*np.log(p))
    return(entropy)

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
        lambda_matrix=-zeta*np.ones((K,N)) # actualiza Lambda 
        z_matrix=np.einsum('ij,jl,ki->kj',w_matrix,r_matrix,ski) # actualiza z
        rj_aux=update_rj_aux(zeta,w_matrix,ski,z_matrix,phi_matrix,psi_matrix,lambda_matrix) # actualiza r barra
        r_matrix=rj_aux/np.sum(rj_aux) # actualiza r
    return(calculate_min_GAS(phi_matrix,psi_matrix,L,r_matrix))  # calcula salida de algoritmo

    
def LAMBDA_MATRIX_no_overflow(ski,lambda_matrix,zeta,z_matrix):
    M=ski.shape[1]
    K=ski.shape[0]
    N=lambda_matrix.shape[1]
    L=np.zeros((M,N))
    zjmax_matrix=zj_max(ski,z_matrix) # (N,1)
    for i in range(M):
        for j in range(N):
            sum=0
            for k in range(K):
                sum+=-ski[k,i]*(lambda_matrix[k,j])+ski[k,i]*np.log(z_matrix[k,j])*zeta
            sum+=-zeta*zjmax_matrix[j,0]
            L[i,j]=np.exp(sum)
    return(L)

def zj_max(ski,z_matrix):  # returns (N,1)
    M=ski.shape[1]
    K=ski.shape[0]
    N=z_matrix.shape[1]
    zj_max=np.zeros((N,1))
    for j in range(N):
        sum_aux=0
        for k in range(K):
            sum_aux+=ski[0,k]*np.log(z_matrix[k,j])   # primer valor es el max
        max_value=sum_aux
        for i in range(M):
            sum=0
            for k in range(K):
                sum+=ski[k,i]*np.log(z_matrix[k,j])  # verificar aca dimensiones
            if sum>=max_value:
                max_value=sum
        zj_max[j,:]=max_value #ver que esto funcione
    return(zj_max)


def GAS_no_overflow(pi,qk,ski,I,N,max_iter):
    M = len(pi)  # alfabeto de X 
    K = len(qk) # alfabeto de Y
    phi_matrix=np.ones((M,1))  # matriz de 1´s de M,1
    psi_matrix=np.ones((N,1))  # matriz de 1´s de N,1
    zeta,eta=1,1
    lambda_matrix=-1*np.ones((K,N))# matriz de -1´s K,N

    r_matrix= np.ones((N,1))/N # matriz de 1´s/N N,1 dim
    w_matrix= np.ones((M, N))/M # matriz de 1´s/M M,N dim
    z_matrix= np.ones((K,N))/(K*N)#  matriz de 1´s/KN K,N dim
    L=LAMBDA_MATRIX_no_overflow(ski,lambda_matrix,zeta,z_matrix)
    for i in range(max_iter):
        psi_matrix=np.divide(np.ones((N,1)),L.T@phi_matrix)  
        phi_matrix=np.divide(pi,(L)@(psi_matrix*r_matrix))
        psi_matrix_no_of=psi_matrix*np.exp(-zeta*zj_max(ski,z_matrix))  #  N,1 y ver como se actualizan, si solo en algunos lugares o en todos # OVERFLOW
        g = lambda x: G_find_zeros(x,r_matrix,phi_matrix,ski,lambda_matrix,z_matrix,psi_matrix_no_of,qk,I) # hay que cambiar la entrada del psi_matrix
        zeta=find_positive_roots(g)
        L=LAMBDA_MATRIX_no_overflow(ski,lambda_matrix,zeta,z_matrix)  # cambiar aca

        w_matrix=np.einsum('ij,jl,il->ij',L,psi_matrix,phi_matrix)   # corroborar que funcione
        lambda_matrix=-zeta*np.ones((K,N))
        z_matrix=np.einsum('ij,jl,ki->kj',w_matrix,r_matrix,ski)   # corroborar que funcione
        
        psi_matrix_no_of=psi_matrix*np.exp(-zeta*zj_max(ski,z_matrix))
        rj_aux=update_rj_aux(zeta,w_matrix,ski,z_matrix,phi_matrix,psi_matrix_no_of,lambda_matrix) # corroborar que funcione
        r_matrix=rj_aux/np.sum(rj_aux) # revisar esto los none types y los overflow
    return(calculate_min_GAS(phi_matrix,psi_matrix,L,r_matrix))