import numpy as np

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

