import pandas as pd
import numpy as np

# UCI Iris dataset URL
iris_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# Define column names
column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]

# Load the Iris dataset into a Pandas DataFrame
iris_df = pd.read_csv(iris_url, header=None, names=column_names)
label_map = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
iris_df['class'] = iris_df['class'].replace(label_map)


def empirical_dist_iris(bins = (2,2,2,2,3)):
    H, edges=np.histogramdd((iris_df["sepal_length"],iris_df["petal_length"],iris_df["sepal_width"],iris_df["petal_width"],iris_df["class"]),bins = bins,density=False)
    
    prob=np.reshape(H,(-1,H.shape[-1]))
    prob/=np.sum(prob)
    p_yx=prob.T # P(Y,X) K,M
    px=np.sum(p_yx,axis=0).reshape(-1,1)
    py=np.sum(p_yx,axis=1).reshape(1,-1)
    py_x=p_yx/np.where(np.sum(p_yx,axis=0)!=0,np.sum(p_yx,axis=0),1)
    return(p_yx,py_x,px,py.reshape(-1,1))

p_yx,py_x,px,py=empirical_dist_iris()
qk=py
pi=px
ski=py_x

