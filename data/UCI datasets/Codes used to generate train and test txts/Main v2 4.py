#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 17:25:47 2018

@author: ablaas


v2: now with sparse method for LACKI and cv for POKI to make it more stable
"""

#%% 0. Import libraries + define necessary functions + weight 





# 0.1 Load packages 
import os
#os.chdir("~Documents/2 Oxford/Uni/DPhil/Codes/Uncertainty bounds/Solar regression")
import numpy as np
import time
import pandas
import gpflow
#from Lipschitz_inference_v2 import LipModel
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
#from LipLikelihood import Lip_log_likelihood
#from LipLikelihood import Gauss_log_likelihood
plt.style.use('ggplot')
#%matplotlib inline

# 0.2 Define global variables
gamma = 0.0001*0.25  #since we do not take lr update to the power of 0.25, we need to divide by 4 when using to replicate Yarin's results
power = 0.25
base_lr = 5e-2
momentum = 0.9
weight_decay = 5e-6  
dropout = 0.1
samples = 200
BNN = True
plotting = False     #only works for 1D input
Lip_method = 'POKI'
Lip_losstype = 'l1'
split = 0.8
maxevals = 2000
optimizer = 'Shubert'
cv = 5

BATCH_SIZE = 128
EPOCHS = 100

# 0.3 Define functions
def plotBNN(m,X,Y,X_test, Y_test):
    xx = np.arange(min(X), max(X_test), 0.05)
    M_pred = np.zeros((samples,len(xx)))
    for ii in range(0,samples):    
        M_pred[ii,:] = np.squeeze(m.predict(xx))
    y_mean = np.mean(M_pred, axis=0)
    y_var = np.var(M_pred, axis=0)
    y_std = y_var**(0.5)
    plt.figure(figsize=(9, 3))
    plt.plot(X, Y, 'k-', mew=2)
    plt.plot(X_test, Y_test, 'g-', mew=2)
    plt.plot(xx, y_mean, 'b', lw=2)
    plt.fill_between(xx, y_mean-1.96*y_std, y_mean+1.96*y_std, color='blue', alpha=0.2)
    plt.xlim(min(X), max(X_test))
    plt.ylim(-20, 20)
    plt.title('Inference with BNN', fontsize='x-large')
    plt.savefig('BNN Inference.png')
    
def eval_BNN(m,X_test, Y_test):
    xx = X_test
    M_pred = np.zeros((samples,len(xx[:,0])))
    for ii in range(0,samples):    
        M_pred[ii,:] = np.squeeze(m.predict(xx))
    y_mean = np.mean(M_pred, axis=0)
    y_var = np.var(M_pred, axis=0)
    percentage = 0.0
    for ii in range(0,len(xx[:,0])):    
        if ((Y_test[ii] <= (y_mean[ii] + 2.0*np.sqrt(y_var[ii]))) and (Y_test[ii] >= (y_mean[ii] - 2.0*np.sqrt(y_var[ii])))):
            percentage += 1.0
    percentage /= len(xx[:,0])
    v_ll = Gauss_log_likelihood(Y_test,(y_mean - 2.0*np.sqrt(y_var)), (y_mean + 2.0*np.sqrt(y_var)))
    ll = np.sum(v_ll)
    v_SE = (Y_test - y_mean)**2
    RMSE = np.mean(v_SE)**(0.5)
    return RMSE,ll, percentage
    
def plotGP(m,X,Y,X_test, Y_test):
    xx = np.linspace(min(X), max(X_test), 500)[:,None]
    mean, var = m.predict_y(xx)
    plt.figure(figsize=(9, 3))
    plt.plot(X, Y, 'k-', mew=2)
    plt.plot(X_test, Y_test, 'g-', mew=2)
    plt.plot(xx, mean, 'b', lw=2)
    plt.fill_between(xx[:,0], mean[:,0] - 2*np.sqrt(var[:,0]), mean[:,0] + 2*np.sqrt(var[:,0]), color='blue', alpha=0.2)
    plt.xlim(min(X), max(X_test))
    plt.ylim(-20, 20)
    plt.title('Inference with GP', fontsize='x-large')
    plt.savefig('GP inference.png')
      
def eval_GP(m,X_test, Y_test):
    xx = X_test
    mean, var = m.predict_y(xx)
    percentage = 0.0
    for ii in range(0,len(xx[:,0])):    
        if ((Y_test[ii] <= (mean[ii] + 2.0*np.sqrt(var[ii]))) and (Y_test[ii] >= (mean[ii] - 2.0*np.sqrt(var[ii])))):
            percentage += 1.0
    percentage /= len(xx[:,0])
    v_ll = Gauss_log_likelihood(np.expand_dims(Y_test,axis=1),(mean - 2.0*np.sqrt(var)), (mean + 2.0*np.sqrt(var)))
    ll = np.sum(v_ll)
    v_SE = (Y_test - np.squeeze(mean))**2
    RMSE = np.mean(v_SE)**(0.5)
    return RMSE,ll, percentage
    
def plotLip(m,X,Y,X_test, Y_test):
    xx = np.append(X, X_test)
    M_pred = np.zeros((3,len(xx)))
    for ii in range(0,len(xx)):    
        M_pred[0,ii],M_pred[1,ii],M_pred[2,ii]  = m.predict(xx[ii])
    plt.figure(figsize=(9, 3))
    plt.plot(X, Y, 'k-', mew=2)
    plt.plot(X_test, Y_test, 'g-', mew=2)
    plt.plot(xx, M_pred[0,:], 'b', lw=2)
    plt.fill_between(xx, M_pred[2,:], M_pred[1,:], color='blue', alpha=0.2)
    plt.xlim(min(X), max(X_test)) 
    plt.ylim(-20, 20)
    plt.title('Inference with Lipschitz Interpolation', fontsize='x-large')
    plt.savefig('Lipschitz inference.png')
    
def eval_Lip(m,X_test, Y_test):
    xx = X_test
    M_pred = np.zeros((3,len(xx[:,0])))
    percentage = 0.0
    for ii in range(0,len(xx[:,0])):    
        M_pred[0,ii],M_pred[1,ii],M_pred[2,ii]  = m.predict(xx[ii,:])
        if ((Y_test[ii] <= M_pred[1,ii]) and (Y_test[ii] >= M_pred[2,ii])):
            percentage += 1.0
    percentage /= len(xx[:,0])
    v_ll = Lip_log_likelihood(Y_test, M_pred[2,:], M_pred[1,:])
    ll = np.sum(v_ll)
    v_SE = (Y_test - M_pred[0,:])**2
    RMSE = np.mean(v_SE)**(0.5)
    return RMSE ,ll, percentage
    
#%% 1. Load and preprocess data

#1.1 Load
filepath = os.path.expanduser(os.path.expanduser("~/Documents/2 Oxford/Uni/DPhil/Codes/Uncertainty bounds/Kin8nm")) 
data_file = os.path.join(filepath, "kin8nm.txt")


data = np.loadtxt(data_file)
print(np.shape(data))



#data_scaled = preprocessing.scale(data) #if we want to standardize using moments of entire data instead of only training data


#1.2 Split into training and test

np.random.shuffle(data)
split_pos = int(np.floor(len(data[:,0])*split))
train_data ,test_data = data[:split_pos, :], data[split_pos:,:]

std_train = np.std(train_data, 0)
std_train[ std_train == 0 ] = 1
mean_train = np.mean(train_data, 0)

#1.3 Center and normalise

train_data = (train_data - np.full(train_data.shape, mean_train)) / \
            np.full(train_data.shape, std_train)

test_data = (test_data - np.full(test_data.shape, mean_train)) / \
            np.full(test_data.shape, std_train)
            
data_scaled = np.append(train_data, test_data,axis=0)

X_train = train_data[:,0:-1]
Y_train = train_data[:,-1]

X_test = test_data[:,0:-1]
Y_test = test_data[:,-1]

#1.4 Save normalized data and parameters used for normalization

np.savetxt('train.txt', train_data)
np.savetxt('test.txt', test_data)

with open('normalizers.txt', 'w') as f:
    print('mean vector of training data', mean_train, "\n" 'std vector of training data',std_train, file=f) 


#1.5 Find LACKI Lipschitz lower bound 

L_low = 0.0
for ll in range(0,len(data_scaled[:,0])-1):
    for mm in range(1,len(data_scaled[:,0])-ll):
        candidate = np.absolute(data_scaled[ll,-1]-data_scaled[ll+mm,-1])/np.linalg.norm(data_scaled[ll,0:-1]-data_scaled[ll+mm,0:-1])
        if (candidate > L_low): #& (candidate < 66.6):
            L_low = candidate 
print(L_low)

#%% 2. Design models


#2.1 Build GP

kGP = gpflow.kernels.Matern52(1, lengthscales=0.3)
modelGP = gpflow.models.GPR(X_train, Y_train[:,np.newaxis], kern=kGP)
modelGP.likelihood.variance = 0.01
modelGP.compile()


#2.2 Build Yarin's BNN

if BNN:
    modelBNN = Sequential()
    modelBNN.add(Dense(units=1024, activation='relu', input_dim=np.shape(X_train)[1], kernel_regularizer=regularizers.l2(weight_decay), bias_regularizer=regularizers.l2(weight_decay)))
    modelBNN.add(Lambda(lambda x: K.dropout(x,level=dropout)))
    modelBNN.add(Dense(units=1024, activation='relu', kernel_regularizer=regularizers.l2(weight_decay), bias_regularizer=regularizers.l2(weight_decay)))
    modelBNN.add(Lambda(lambda x: K.dropout(x,level=dropout)))
    modelBNN.add(Dense(units=1024, activation='relu', kernel_regularizer=regularizers.l2(weight_decay), bias_regularizer=regularizers.l2(weight_decay)))
    modelBNN.add(Lambda(lambda x: K.dropout(x,level=dropout)))
    modelBNN.add(Dense(units=1))
    sgd = SGD(lr=base_lr, decay=gamma, momentum=momentum, nesterov=False)
    modelBNN.compile(loss='mean_squared_error', optimizer='adam')
    modelBNN.summary()
    


#2.3 Build Lipschitz Model
modelLip = LipModel((X_train),np.expand_dims(Y_train,axis=1))


#%% 3. Train models

time_vec = np.zeros(shape=(3,))

# 3.1 Train GP
t = time.time()
gpflow.train.ScipyOptimizer().minimize(modelGP)
time_vec[0] = time.time() - t
# 3.2 Train BNN
if BNN:
    t = time.time()
    historyBNN = modelBNN.fit(X_train, Y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1, validation_split = 0.15)
    tl,vl = historyBNN.history['loss'], historyBNN.history['val_loss'] 
    time_vec[1] = time.time() - t

    # -> Save trained BNN
    model_json = modelBNN.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
        # serialize weights to HDF5
        modelBNN.save_weights("modelBNN.h5")
        print("Saved model to disk")



# 3.3 Train LACKI, POKI and LDNN
t = time.time()
modelLip.train(method='LACKI',losstype=Lip_losstype, optimizer = 'Shubert', maxevals = maxevals, cv = 10)  #Lip_method
print(modelLip.L, modelLip.iter) 
time_vec[2] = time.time() - t
 
#print(time_vec) 
#modelLip.L = 30.0   
# -> Save trained Lipschitz Model
with open('modelLip.txt', 'w') as f:
    print('Optimized Lipschitz constant', modelLip.L, ' - method used:',Lip_method, 'with losstype', Lip_losstype, ' and maxevals:', maxevals, 'using optimizer:',optimizer,'. LACKI for entire training data =',L_low, file=f) 

#%%
     
#2. Run on other data.
#3. Understand instability of L Poki
#4. Implement log-likelihood -> wie soll das überhaupt aussehen für lip interpolation?
#5. Implement ARD lipschitz
#6. Make POKI optimization faster by setting lower bound for L and writing while loop doubling max L each time hat L is too close to max L (Jan's Algo im prinzip)
 

#%% 4. Evaluate models 






#4.1 Evaluate by plot
if plotting:
    plotGP(modelGP,X_train,Y_train, X_test,Y_test)
    if BNN:
        plotBNN(modelBNN,X_train,Y_train, X_test, Y_test)
    plotLip(modelLip,X_train,Y_train,X_test, Y_test)


#4.2 Evaluate by time taken for training of model
etimes_in_sec = {
    'GP': time_vec[0],
    'BNN': time_vec[1],
    'Lip': time_vec[2]
    }

#print(times_in_sec)

#4.3 Evaluate by RMSE on test data

eval_results = {
    'GP': eval_GP(modelGP,X_test,Y_test),
    'BNN': eval_BNN(modelBNN,X_test,Y_test),
    'Lip': eval_Lip(modelLip,X_test,Y_test)
    }


with open('results.txt', 'w') as f:
    print('Time used to train model in seconds', etimes_in_sec, "\n" '(RMSEs, % test points inside uncertainty bounds):',eval_results, file=f) 

with open('traindata_used.txt', 'w') as f:
    print('TRAINDATA', "\n", train_data ,'TESTDATA',"\n" ,test_data, file=f) 



#test = np.loadtxt('traindata_used.txt')

#print('RMSE GP:', RMSE_GP(modelGP,X_test,Y_test))
#print('RMSE BNN:', RMSE_BNN(modelBNN,X_test,Y_test))
#print('RMSE Lip:', RMSE_Lip(modelLip,X_test,Y_test))
#






#%%  

#def loss_vec(L, x_cond, y_cond, x_eval, y_eval):
#        
#    y_eval = np.squeeze(y_eval)
#    assert np.shape(y_eval) == (len(y_eval),)
#    if L < 0:
#        L = 0
#    pred_temp = np.squeeze(np.zeros(shape=(len(x_eval[:,0]), 1)))
#    for ii in range(0, len(x_eval[:,0])):
#        pred_temp[ii] = inference(x_eval[ii,:], L, x_cond, y_cond)
#    loss_vector = np.absolute(y_eval - pred_temp)   
#    return loss_vector  
#
#
#def upper_bound(xx,l_n,X,Y):
#    'gives upper_bound for function value at x for Lipschitz constant L and sample Data D strictly as defined in paper'
#    u_vec = Y + l_n * np.expand_dims(np.linalg.norm((X - xx),norm, axis=1),axis=1)
#    u = np.amin(u_vec)
#    return u
#
#        
#def lower_bound(xx, l_n,X,Y): 
#    'gives lower_bound for function value at x for Lipschitz constant L and sample Data D strictly as defined in paper'
#    l_vec = Y - l_n * np.expand_dims(np.linalg.norm((X - xx),norm, axis=1),axis=1)
#    l = np.amax(l_vec)
#    return l 
#
#    
#    
#def inference(xx, l_n, X, Y):
#    'Gives prediction at point x'
#    y = alpha*upper_bound(xx,l_n,X,Y) + (1.0-alpha)*lower_bound(xx,l_n,X,Y)
#    return y 
#
#
#       
#def loss_fun(L):
#    # loss function as a function of the vector containing absolute errors for each evaluation point
#    if losstype =='l1':
#        loss = np.average(loss_vec(L, x_cond, y_cond, x_eval, y_eval))
#    elif losstype=='l2':
#        loss = np.average(loss_vec(L, x_cond, y_cond, x_eval, y_eval)**2)
#    elif losstype=='maxi':
#        loss = np.amax(loss_vec(L, x_cond, y_cond, x_eval, y_eval))
#    else: print('losstype not defined, could not train model')
#    return loss 
#
#losstype = Lip_losstype
#alpha = 0.5  
#norm = None 
#
#
#diam_low = 0.0
#for ll in range(0,len(np.expand_dims(Y_train,axis=1))):
#    for mm in range(1,len(np.expand_dims(Y_train,axis=1))-ll):
#        candidate = np.linalg.norm(X_train[ll,:]-X_train[ll+mm,:],norm)
#        if (candidate > diam_low): 
#            diam_low = candidate 
#
#print(diam_low)
#
#data = np.c_[X_train, np.expand_dims(Y_train,axis=1)]
#np.random.shuffle(data)
#split_pos = int(np.floor(len(np.expand_dims(Y_train,axis=1))*split))
#data_cond ,data_eval = data[:split_pos, :], data[split_pos:,:]
#x_cond = data_cond[:,0:-1]
#y_cond = np.expand_dims(np.array(data_cond[:,-1]),axis=1)
#x_eval = data_eval[:,0:-1]
#y_eval = np.expand_dims(np.array(data_eval[:,-1]),axis=1)
#  
#
#       
##            print(data)
##            print('x_cond:',x_cond, 'y_cond:', y_cond)# 'x_eval:' ,x_eval, 'y_eval:', y_eval)
#
#
#xx = np.arange(0.0, 5.0, 0.05)
#M_pred = np.zeros((len(xx)))
#for ii in range(0,len(xx)):    
#    M_pred[ii] = loss_fun(xx[ii])
#plt.figure(figsize=(9, 4))
#plt.plot(xx, M_pred, 'k-', mew=2)






