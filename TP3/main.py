from matplotlib import cm
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def chargement():

    dt = pd.read_csv(r"data_ffnn_3classes.txt", sep='\s+')

    X1 = dt.iloc[:,0].to_numpy()
    X2 = dt.iloc[:,1].to_numpy()
    Y_Training = dt.iloc[:,2].to_numpy()

    X_Training = np.array([np.ones([len(X1)]),X1,X2],dtype=float).T
    X_Training = X_Training/np.max(X_Training)
    Y_Training = Y_Training/np.max(Y_Training)

    return X_Training, Y_Training

def inverse(mat):
    retmat=np.zeros((len(mat),len(mat[0])))
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            retmat[i][j] =  1 / mat[i][j]
    return retmat

if __name__ == "__main__" :

    #premiere couche
    V=np.random.rand(3,3)
    X_Training, Y_Training=chargement()
    XX = np.array(X_Training.dot(V))
    F = ((1+np.exp(-XX)))
    inv_F = inverse(F)

    #deuxieme couche
    FF_tmp=np.c_[np.ones(inv_F.shape[0]),inv_F]
    W=np.random.rand(4,3)
    FF=FF_tmp.dot(W)
    G=((1+np.exp(-FF)))
    invG=inverse(G)



