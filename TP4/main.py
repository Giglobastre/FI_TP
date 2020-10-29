# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 10:26:01 2020

@author: hugo7
"""
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import inv
from mpl_toolkits.mplot3d import Axes3D
from math import *

def chargement_sequence():
    cpt=0
    seq=[]
    for i in range(10):
        nb=random.randint(0,1)
        seq.append(nb)
    return seq

def cost_function (y,ypredicted):
    m=len(y)
    tot = 0.0
    for i in range (m):
        tot += (ypredicted[i] - y[i])**2
    return tot /(2*m)

def chargement_input():
    y=[]
    ytest=[]
    for i in range (40):
        seq=chargement_sequence()
        if i<30:
            y.append(sum(seq))
            if i==0:
                X=seq
            else:
                X=np.c_[X,seq]
        else:
            ytest.append(sum(seq))
            if i==30:
                Xtest=seq
            else:
                Xtest=np.c_[Xtest,seq]
            
    return X.T,y,Xtest.T,ytest

def forward(X,vf,vx):
    y=np.zeros(len(X))
    f=np.zeros([30,10])
    #f[:,0]=1
    for n in range (len(X)): 
        for i in range (10):
           if i==0:
               previous=1
           
           f[n][i]=vf*previous+vx*X[n][i]
           #print(vf,previous,vx,X[n][i])
           #print(vf*previous+vx*X[n][i])
           previous=f[n][i]
        y[n]=previous
    
    
    return y.T,f

def backpropagation(Ypredict,Y,X,vf,vx,f):
# =============================================================================
#     Calcul de vfet vx
# ============================================================================
    sumT = 0
    sumX = 0
    summ = 0
    sum2 = 0
    for t in range (1,10):
       
        for i in range (1,len(X)):
            summ = summ + (Ypredict[i] - Y[i]) * f[i][t - 1]     
            sum2 = sum2 + (Ypredict[i] - Y[i]) * X[i][t]

        sumT = sumT + summ * pow(vf,10 - t)
        sumX = sumX + sum2 * pow(vf,10 - t)
    
        summ = 0
        sum2 = 0

    vf = vf - 0.000001 * sumT
    vx = vx - 0.000001 * sumX
    return vf,vx
def resilient_propagation(Ypredict,Y,X,vf,vx,f):
# =============================================================================
#     Calcul de grad vf/vx
# ============================================================================
    sumT = 0
    sumX = 0
    summ = 0
    sum2 = 0
    yp=1.2
    yn=0.5
    deltaf=vf
    deltax=vx
    for t in range (1,10):
       
        for i in range (1,len(X)):
            summ = summ + (Ypredict[i] - Y[i]) * f[i][t - 1]     
            sum2 = sum2 + (Ypredict[i] - Y[i]) * X[i][t]

        sumT = sumT + summ * pow(vf,10 - t)
        sumX = sumX + sum2 * pow(vf,10 - t)
    
        summ = 0
        sum2 = 0
# =============================================================================
#         resilient vf
# =============================================================================
    signf=1
    if sumT <0:
        signf=-1
    if sumT >0:
        signf=1
        
    if ((deltaf<0 and signf==-1 ) or (deltaf>0 and signf==1)):
        vf = vf -signf*(0.001*yp)
        
    if ((deltaf>0 and signf==-1 ) or (deltaf<0 and signf==1)):
        vf = vf -signf*(0.001*yn)
        
# =============================================================================
#      resilient vx   
# =============================================================================
    signx=1
    if sumX <0:
        signx=-1
    if sumX >0:
        signx=1
    
    if ((deltax<0 and signx==-1 ) or (deltax>0 and signx==1)):
        vx = vx -signx*(0.001*yp)
      
        
    if ((deltax>0 and signx==-1 ) or (deltax<0 and signx==1)):
        vx = vx -signx*(0.001*yn)
        
    
    return vf,vx 

def clip_propagation(Ypredict,Y,X,vf,vx,f):
# =============================================================================
#     Calcul de grad vf/vx
# ============================================================================
    sumT = 0
    sumX = 0
    summ = 0
    sum2 = 0
    for t in range (1,10):
       
        for i in range (1,len(X)):
            summ = summ + (Ypredict[i] - Y[i]) * f[i][t - 1]     
            sum2 = sum2 + (Ypredict[i] - Y[i]) * X[i][t]

        sumT = sumT + summ * pow(vf,10 - t)
        sumX = sumX + sum2 * pow(vf,10 - t)
    
        summ = 0
        sum2 = 0
    vf = vf - 0.000001 * sumT
    vx = vx - 0.000001 * sumX
    if sumT>4:
        vf=(4*sumT)/(sqrt(sumT**2))
    if sumX>4:
        vx=(4*sumX)/(sqrt(sumX**2))
        
    return vf,vx


def backboucle(Y,X,vf,vx)  :
    cost_H = []
    for ita in range(10000):
        ypredict, f = forward(X,vf,vx)
        vf, vx = backpropagation(ypredict,Y,X,vf,vx,f)
        cost = cost_function(Y, ypredict)
        
        cost_H.append(cost)
    return ypredict, cost_H,vf ,vx

def resilient_boucle(Y,X,vf,vx)  :
    cost_H = []
    for ita in range(10000):
        ypredict, f = forward(X,vf,vx)
        vf, vx = resilient_propagation(ypredict,Y,X,vf,vx,f)
        cost = cost_function(Y, ypredict)
        
        cost_H.append(cost)
    return ypredict, cost_H, vf, vx

def clip_boucle(Y,X,vf,vx)  :
    cost_H = []
    for ita in range(10000):
        ypredict, f = forward(X,vf,vx)
        vf, vx = clip_propagation(ypredict,Y,X,vf,vx,f)
        cost = cost_function(Y, ypredict)
        
        cost_H.append(cost)
    return ypredict, cost_H,vf ,vx

 
if __name__=="__main__":
    Xtrain,Ytrain,Xtest,Ytest = chargement_input()
    vf=0.4
    vx=0.5
    
# =============================================================================
#     Sur les trois lignes ci dessous, enlever le # de la méthode voulue. 
#     Mettez un # devant la methode que vous ne voulez plus utiliser.
# =============================================================================
    #ypredict, cost, vf, vx = clip_boucle(Ytrain, Xtrain, vf, vx) #gradient clippping
    ypredict, cost, vf, vx = backboucle(Ytrain, Xtrain, vf, vx) #backpropagtion
    #ypredict, cost, vf, vx = resilient_boucle(Ytrain, Xtrain, vf, vx) #resilient 
    
    
    print("--------YTrain----------------")
    print(Ytrain)
    print("--------Ypredicted-----------")
    print(np.around(ypredict,decimals=0,out=None))#valeurs arrondies
    #print(ypredict)#non arrondi

    ypredict_test, f = forward(Xtest,vf,vx)
    print("--------Ytest----------------")
    print(Ytest)
    print("--------Ypredict_test----------------")
    print(np.around(ypredict_test,decimals=0,out=None)) #valeurs arrondies
    #print(ypredict_test)#non arrondi

    plt.plot(cost)
    plt.show()
   #ypredict,f=forward(Xtrain,vf,vx)
