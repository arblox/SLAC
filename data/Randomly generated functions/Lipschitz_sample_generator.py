#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 21:42:54 2018

@author: ablaas
"""

# 0.1 Load packages 
import os
#os.chdir("~Documents/2 Oxford/Uni/DPhil/Codes/Uncertainty bounds/Solar regression")
import numpy as np
import pandas as pd
import time
import pandas
import gpflow
from matplotlib import pyplot as plt
import keras
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.utils import plot_model
from keras.layers.core import Lambda
from keras import backend as K
from scipy.optimize import minimize
from sklearn import preprocessing
plt.style.use('ggplot')


np.random.seed(2000)
a = -1.0
b = 1.0
stepsize = 0.0002

N = 1000000 # number of distinct data points to be learned
T = 1000# #number of functions to be drawn
l_vec_NN = np.zeros(T)
l_vec_LACKI = np.zeros(T)
l_vec_paper = np.zeros(T)

columnames=['X']
for cc in range(1,T+1):
    columnames.append("Y{0}".format(cc))  

rownames = ['LACKI Lipschitz constant','New NN Lipschitz constant','Old NN Lipschitz constant']
for cc in range(1,int((b-a)/stepsize)+1):
    rownames.append("Value{0}".format(cc))


l1 = int(np.sqrt(3.0*N)+2.0*np.sqrt(N/3.0))
l2 = int(np.sqrt(N/3.0))

def plotBNN(m):
    xx = np.arange(a, b, stepsize)
    M_pred = np.zeros(len(xx))      
    M_pred = np.squeeze(m.predict(xx))
    plt.figure(figsize=(9, 3))
    plt.plot(xx, M_pred, 'k-', mew=2)
    plt.xlim(a, b)
#    plt.ylim(-1, 1)
    plt.title('Inference with BNN', fontsize='x-large')
    L_low = 0.0
    for ll in range(0,len(xx)-1):
        candidate = np.absolute(M_pred[ll]-M_pred[ll+1])/max(0.00001,np.linalg.norm(xx[ll]-xx[ll+1]))
        if candidate > L_low:
            L_low = candidate
    print(L_low)
#    plt.savefig('BNN Inference.png')

modelBNN = Sequential()
modelBNN.add(Dense(units=l1, activation='sigmoid', input_dim=1))
#modelBNN.add(Lambda(lambda x: K.dropout(x,level=dropout)))
modelBNN.add(Dense(units=l2, activation='sigmoid'))
#modelBNN.add(Lambda(lambda x: K.dropout(x,level=dropout)))
modelBNN.add(Dense(units=1))
sgd = SGD(lr=base_lr, decay=gamma, momentum=momentum, nesterov=False)
modelBNN.compile(loss='mean_squared_error', optimizer='adam')
modelBNN.summary()

xx = np.arange(a, b, stepsize)
dataframe = np.zeros((len(xx)+3,T+1))
dataframe[3:,0] = xx
M_pred = np.zeros(len(xx))      
plt.figure(figsize=(9, 3))
plt.xlim(a, b)
plt.title('Sample functions drawn from random 2-layer NNs', fontsize='x-large')


for kk in range(0,T):
    k = modelBNN.get_weights()
    k[0] = np.random.normal(0.0,1.0,(1, l1))
    k[1] = np.random.normal(0.0,1.0,(l1, ))
    k[2] = np.random.normal(0.0,10.0,(l1, l2))
    k[3] = np.random.normal(0.0,1.0,(l2, ))
    k[4] = np.random.normal(0.0,10.0,(l2, 1))
    k[5] = np.random.normal(0.0,1.0,(1, ))
    modelBNN.set_weights(k)
    
    M_pred = np.squeeze(modelBNN.predict(xx))
    M_pred = (M_pred - np.mean(M_pred))/np.std(M_pred)
    dataframe[3:,kk+1]=M_pred
    plt.plot(xx, M_pred, 'k-', mew=2)
    
    L_low = 0.0
    for ll in range(0,len(xx)-1):
        candidate = np.absolute(M_pred[ll]-M_pred[ll+1])/max(0.00001,np.linalg.norm(xx[ll]-xx[ll+1]))
        if candidate > L_low:
            L_low = candidate
    L = np.matmul(np.abs(k[0]),np.abs(k[2]))
    l_vec_NN[kk] = np.matmul(L,np.abs(k[4]))#np.linalg.norm(k[0],1)*np.linalg.norm(k[2],1)*np.linalg.norm(k[4],1)
    l_vec_LACKI[kk] = L_low
    l_vec_paper[kk] = np.linalg.norm(k[0])*np.linalg.norm(k[2])*np.linalg.norm(k[4])

    dataframe[0:3,kk+1]=[l_vec_LACKI[kk],l_vec_NN[kk],l_vec_paper[kk]]

dataframe = pd.DataFrame(dataframe, columns=columnames)

print(np.amax(l_vec_LACKI))
print(np.amax(l_vec_NN))
print(np.amax(l_vec_paper))
plt.savefig('Sampled functions.png')

import pickle

with open('dataset.pickle', 'wb') as handle:
    pickle.dump(dataframe, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('dataset.pickle', 'rb') as handle:
    b = pickle.load(handle)

print(dataframe == b)


#print(np.floor(L)+1.0)
#plotBNN(modelBNN)

#tunen und rausfinden warum immer an den rändern des plots flach


#%%

#s = 1000000
#normals = np.random.normal(0,10,s)
#transformed = 1.0/(1.0+np.exp(-normals))
#plt.hist(transformed, np.linspace(0,1,100))
  

    