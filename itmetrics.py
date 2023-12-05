import numpy as np

def entropy(p):
    """
    Calcula la entropía de una distribución de probabilidad
    """
    entropy=np.sum(np.where(p != 0, -p * np.log(p), 0))
    return(entropy)

def jEntropy(p_yx):
    """
    H(Y;X)
    Reference: https://en.wikipedia.org/wiki/Joint_entropy
    """
    return entropy(p_yx.flatten())

def cEntropy(p_yx,px): #P(X,Y) P(X)-->P(Y|X)
    """
    H(Y|X)
    Reference: https://en.wikipedia.org/wiki/Conditional_entropy
    """
    return jEntropy(p_yx) - entropy(px)  # H(Y|X)

def mutual_information(p_yx,px,py):
    """
    I(X;Y)
    Reference: https://en.wikipedia.org/wiki/Mutual_information
    """
    return entropy(px) - cEntropy(p_yx,py)  # H(X) - H(Y|X)

def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))