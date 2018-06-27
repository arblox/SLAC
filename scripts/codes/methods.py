#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
First created on Sun Mar 11 2018
Final version on Sun Jun 24 2018

@authors: Arno Blaas and Adam D. Cobb
"""

#TO DO: comment and clean both methods and utils


try:
    import pylab as pb
except:
    pass
import numpy as np
from operator import itemgetter
# import matplotlib.pyplot as plt

import gpflow

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
from scipy.misc import logsumexp

from codes.utils import fct_minimise_Shubert, fct_minimise_Malherbe
from codes.utils import Gauss_log_likelihood, Lip_log_likelihood


#%%

class LipModel:
    """
    Class containing all relevant functionalities for Lipschitz interpolation model
    """
    
    def __init__(self, X: np.array, Y: np.array, alpha=0.5, L=0.0, norm = None):
        """
        Builds basic Lipschitz interpolation model
        :X: training data X 
        :Y: training data y, needs to be passed as array of shape (n,1);
        :param alpha: weight of upper bound vs. lower bound
        :param L: Best Lipschitz constant of target function, if known
        :param norm: norm to be used in input space
        """
        if np.shape(X)[0] == len(Y):
            assert np.shape(X)[1] >= 1
        else:
            assert np.shape(X)[1] == len(Y)
            X = np.transpose(X)
            
        self.X = X
        
        assert np.shape(Y)[1] == 1
        self.Y = Y
        
        self.support_idx = range(0,np.shape(X)[0])
        
        self.diam_X = []
        self.alpha = alpha

        self.iter = 0
        if L == 0.0:
            print('Warning: Lipschitz constant currently 0. Please train model before inference')
        self.L = L
        self.trained_params = [0]
        self.norm = norm
        return
    

    
    def loss_vec(self,L, x_cond, y_cond, x_eval, y_eval):
        """
        Calculates vector containing absolute differences between prediction and target for each point in evaluation set
        :L: value of hyperparameter L used to predict on evaluation set
        :x_cond: part of training data X used as conditioning set
        :y_cond: part of training data y used as conditioning set
        :x_eval: part of training data X used as evaluation set to be predicted on
        :y_eval: part of training data y used as evaluation set, i.e. true target values at x_eval
        """
        y_eval = np.squeeze(y_eval)
        assert np.shape(y_eval) == (len(y_eval),)
        if L < 0:
            L = 0
        pred_temp = np.squeeze(np.zeros(shape=(len(x_eval[:,0]), 1)))
        for ii in range(0, len(x_eval[:,0])):
            pred_temp[ii] = self.inference(x_eval[ii,:], L, x_cond, y_cond)
        loss_vector = np.absolute(y_eval - pred_temp)   
        return loss_vector  

    def upper_bound(self,xx,l_n,X,Y): 
        '''
        Computes upper_bound u(xx) for target function value at xx for Lipschitz constant hyperparameter l_n and sample data X and Y
        \mathfrak{u}_{l_n}(xx;X,Y) &:= \min_{s_i \in X} (f_i + l_n \mathfrak{d}(s_i,xx))
        :xx: single test point
        :l_n: value of hyperparameter used for prediction
        :X: input of sample data X used for prediction
        :Y: output of sample data Y used for prediction
        '''
        u_vec = Y + l_n * np.expand_dims(np.linalg.norm((X - xx),self.norm, axis=1),axis=1)
        u = np.amin(u_vec)
        return u

        
    def lower_bound(self,xx, l_n,X,Y): 
        '''
        Computes lower bound l(xx) for target function value at xx for Lipschitz constant hyperparameter l_n and sample data X and Y
        \mathfrak{l}_{l_n}(xx;X,Y) &:= \max_{s_i \in X} (f_i - l_n \mathfrak{d}(s_i,xx))
        :xx: single test point
        :l_n: value of hyperparameter used for prediction
        :X: input of sample data X used for prediction
        :Y: output of sample data Y used for prediction
        '''
        l_vec = Y - l_n * np.expand_dims(np.linalg.norm((X - xx),self.norm, axis=1),axis=1)
        l = np.amax(l_vec)
        return l 
    
    def inference(self,xx, l_n, X, Y):
        '''
        Computes prediction of target function value at xx for Lipschitz constant hyperparameter l_n and sample data X and Y
        \hat{f}_{l_n}(xx;X,Y) = \alpha \mathfrak{l}_{l_n}(xx;X,Y) + (1-\alpha)\mathfrak{u}_{l_n}(xx;X,Y)
        :xx: single test point
        :l_n: value of hyperparameter used for prediction
        :X: input of sample data X used for prediction
        :Y: output of sample data Y used for prediction
        '''
        y = self.alpha*self.upper_bound(xx,l_n,X,Y) + (1.0-self.alpha)*self.lower_bound(xx,l_n,X,Y)
        return y 
    
    
    def train(self, method='LACKI', losstype='l2', split=0.5, optimizer='Malherbe', ARD = False, maxevals = 1000, cv = 0, number = 1):
        """
        Trains Lipschitz interpolation model by learning hyperparameter l_n from data
        :param method: determines which estimation is used. (sparse) LACKI or POKI
        :param losstype: only relevant for POKI: which loss function is used? maxi, l1 or l2
        :param split: only relevant for POKI: determines split into conditioning and evaluation data
        :param optimizer: Choice of optimizer only relevant for POKI, Malherbe, Shubert and standard Scipy available
        :param ARD: if true, ARD metric is used, i.e. delta(x,x') = max_(i = 1,...,d) theta_i * |x_i - x'_i|  -- not implemented yet
        :param maxevals: only relevant for POKI, maximum steps in optimisation in POKI
        :param cv: if value for cv is passed, this is how often data is reshuffled for either POKI or (sparse) LACKI, which is automatically applied in case cv is passed
        :number: number of optimizer initializations / restarts of Scipy, has no effect if Malherbe or Shubert optimizer are chosen
        """
        self.trained_params = [1]
        if ARD:
            assert optimizer == 'Malherbe'
            print('ARD not implemented yet')
            pass
        
        if ARD == False:
            d = 1
        
        
        if method == 'POKI':
            assert losstype is not None
            diam_low = 0.0
            for ll in range(0,len(self.Y)):
                for mm in range(1,len(self.Y)-ll):
                    candidate = np.linalg.norm(self.X[ll,:]-self.X[ll+mm,:],self.norm)
                    if (candidate > diam_low): 
                        diam_low = candidate 
        
            self.diam_X = diam_low 
            
            data = np.c_[self.X, self.Y]
            l_new_vec = np.zeros(shape = (cv+1,number))
            l_new_vec[:,:] = np.nan
            for ii in range(0,cv+1):
                np.random.shuffle(data)
                split_pos = int(np.floor(len(self.Y)*split))
                data_cond ,data_eval = data[:split_pos, :], data[split_pos:,:]
                x_cond = data_cond[:,0:-1]
                y_cond = np.expand_dims(np.array(data_cond[:,-1]),axis=1)
                x_eval = data_eval[:,0:-1]
                y_eval = np.expand_dims(np.array(data_eval[:,-1]),axis=1)


                def loss_fun(L):
                    # loss function as a function of the vector containing absolute errors for each evaluation point
                    if losstype =='l1':
                        loss = np.average(self.loss_vec(L, x_cond, y_cond, x_eval, y_eval))
                    elif losstype=='l2':
                        loss = np.average(self.loss_vec(L, x_cond, y_cond, x_eval, y_eval)**2)
                    elif losstype=='maxi':
                        loss = np.amax(self.loss_vec(L, x_cond, y_cond, x_eval, y_eval))
                    else: print('losstype not defined, could not train model')
                    return loss 
                if optimizer == 'Malherbe':
                    curr_argmin,curr_fmin,n,timer = fct_minimise_Malherbe(loss_fun,d,0.0,500.0,L=self.diam_X, maxevals=maxevals)
                    l_new_vec[ii,0] = curr_argmin
                elif optimizer == 'Shubert':
                    upper_bound = 50.0
                    cont = True
                    while cont:
                        curr_argmin,curr_fmin,n = fct_minimise_Shubert(loss_fun,0.0,upper_bound, L=self.diam_X,errthresh =0.05, maxevals=maxevals)
                        if curr_argmin < 0.9*upper_bound:
                            cont = False
                        else:
                            upper_bound = 2.0*upper_bound
                    l_new_vec[ii,0] = curr_argmin
                    self.iter = n
                elif optimizer == 'Scipy':
                    for pp in range(0,number):
                        res = minimize(loss_fun, np.random.uniform(0.0,100.0), method='nelder-mead', options={'xtol': 1e-8, 'disp': False})
                        l_new = res.x
                        l_new_vec[ii,pp] = l_new
                else:
                    print('Error: Chosen optimizer not implemented / check spelling')
                
            self.L = np.mean(l_new_vec[:,0])
            temp = np.mean(l_new_vec, axis = 0)
            self.trained_params = temp[~np.isnan(temp)]
        
        
        elif method == 'LACKI':
            if cv < 1.0:   #corresponds to LAC, exhaustive method of evaluating all combinatorically possible slopes
                L_low = 0.0
                
                for ll in range(0,len(self.Y)-1):
                    for mm in range(1,len(self.Y)-ll):
                        if np.linalg.norm(self.X[ll,:]-self.X[ll+mm,:]) > 0.0:  #ignore doublettes in input space
                            candidate = np.absolute(self.Y[ll]-self.Y[ll+mm])/np.linalg.norm(self.X[ll,:]-self.X[ll+mm,:])
                            if candidate > L_low:
                                L_low = candidate    
                self.L=L_low
            else:  #corresponds to SLAC, i.e. the sparse approximation of LAC
                L_low = 0.0
                data = np.c_[self.X, self.Y]
                if np.shape(self.X)[1] == 1:
                    cv = 1
                for ii in range(0,cv):
                    if np.shape(self.X)[1] == 1:
                        data = data[data[:,0].argsort()]  #for 1D we get SLAC=LAC by sorting data in X
                    else:
                        np.random.shuffle(data)
                    x_cond = data[:,0:-1]
                    y_cond = np.expand_dims(np.array(data[:,-1]),axis=1)
                    for ll in range(0,len(self.Y)-1):
                        if np.linalg.norm(x_cond[ll,:]-x_cond[ll+1,:]) > 0.0:  #ignore doublettes in input space
                            candidate = np.absolute(y_cond[ll]-y_cond[ll+1])/np.linalg.norm(x_cond[ll,:]-x_cond[ll+1,:])
                            if candidate > L_low:
                                L_low = candidate    
                self.L=L_low
                     
        else: 
            print("Method not defined yet, could not train model")
   
    
    def correct_support(self):
        '''
        Corrects support of interpolation method for predictions from \mathcal{D}_n) to \tilde{\mathcal{D}}_n,
        i.e. removes those points in X that for current value of hyperparameter L yield in ill-defined areas of model
        '''
        self.support_idx = []
        for uu in range(0,np.shape(self.X)[0]):
            up = self.upper_bound(self.X[uu,:],self.L,self.X,self.Y)
            low = self.lower_bound(self.X[uu,:],self.L,self.X,self.Y)
            if ((self.Y[uu] >= low) and (self.Y[uu] <= up)):
                self.support_idx.append(uu)
    
    def predict(self,xx):
        '''
        Predicts value of target function at test point xx and returns prediction as well as upper and lower bound at xx
        :xx: test point xx
        '''
        up = self.upper_bound(xx,self.L, self.X[self.support_idx,:], self.Y[self.support_idx])
        low = self.lower_bound(xx,self.L, self.X[self.support_idx,:], self.Y[self.support_idx])
        dist = up-low
        up = low + 1.0*dist  #keep bounds at 100% bounds
        low = low + 0.0*dist #keep bounds at 100% bounds
        mean = self.inference(xx, self.L, self.X[self.support_idx,:], self.Y[self.support_idx])
        return mean, up, low 
    
    
    def evaluate(self,X_test, Y_test):
        '''
        Evaluates Lipschitz interpolation model on test set, returning RMSE, pseudo-loglikelihood,
        precentage of test points in bounds and average distance of bounds. If model was trained multiple times
        in SLAC fashion then also standard deviation of those metrics are returned
        :X_test: input values of test set
        :Y_test: output values of test set
        '''
        if len(self.trained_params) == 1:
            self.trained_params[0] = self.L
        xx = X_test
        M_pred = np.zeros((3*len(self.trained_params),len(xx[:,0])))
        percentage = np.zeros(len(self.trained_params))
        A = np.zeros(len(self.trained_params))
        ll = np.zeros(len(self.trained_params))
        RMSE = np.zeros(len(self.trained_params))
        for jj in range(0,len(self.trained_params)):
            self.L = self.trained_params[jj]
            for ii in range(0,len(xx[:,0])):
                if self.trained_params[0] == 0:
                    M_pred[3*jj+0,ii],M_pred[3*jj+2,ii],M_pred[3*jj+1,ii]  = self.predict(xx[ii,:]) #if L = 0, swaps bounds so that you can calculate what appens for u = max f_i, l = min f_i
                else:
                    M_pred[3*jj+0,ii],M_pred[3*jj+1,ii],M_pred[3*jj+2,ii]  = self.predict(xx[ii,:])
                A[jj] += np.abs(M_pred[3*jj+2,ii] - M_pred[3*jj+1,ii])
                if ((Y_test[ii] <= M_pred[3*jj+1,ii]) and (Y_test[ii] >= M_pred[3*jj+2,ii])):
                    percentage[jj] += 1.0
            percentage[jj] /= len(xx[:,0])
            A[jj] /= len(xx[:,0])
            v_ll = Lip_log_likelihood(Y_test, np.expand_dims(M_pred[3*jj+2,:],axis=1), np.expand_dims(M_pred[3*jj+1,:],axis=1))
            ll[jj] = np.sum(v_ll)
            v_SE = (Y_test - np.expand_dims(M_pred[3*jj+0,:],axis=1))**2
            RMSE[jj] = np.mean(v_SE)**(0.5)
        i = np.argmin(RMSE)
        return RMSE[i], np.std(RMSE), ll[i], np.std(ll), percentage[i],np.std(percentage), A[i], np.std(A), 0.0 ,0.0, 0.0 ,0.0  #placeholders to have equal output for GP and BNN eval
    
    
    def plot(self,X_test, Y_test):
        '''
        If input space X is 1D, plots Lipschitz interpolation model on test set
        :X_test: input values of test set
        :Y_test: output values of test set
        '''
        if np.shape(self.X)[1] >= 2:
            print('Dimension of input too high - cannot plot')
            return
        xx = np.sort(np.append(self.X, X_test))
        M_pred = np.zeros((3,len(xx)))
        for ii in range(0,len(xx)):    
            M_pred[0,ii],M_pred[1,ii],M_pred[2,ii]  = self.predict(xx[ii])
        plt.figure(figsize=(9, 3))
        plt.plot(self.X, self.Y, 'k*', mew=2)
        plt.plot(X_test, Y_test, 'g*', mew=2)
        plt.plot(xx, M_pred[0,:], 'b', lw=2)
        plt.fill_between(xx, M_pred[2,:], M_pred[1,:], color='blue', alpha=0.2)
        plt.xlim(min(self.X), max(X_test)) 
        plt.ylim(-5, 5)
        plt.title('Inference with Lipschitz Interpolation', fontsize='x-large')
#        plt.savefig('Lipschitz inference.png')



#%%

class GPModel:
    """
    Builds wrapper for GP model
    """
    
    def __init__(self, X: np.array, Y: np.array, kernel = 'SE', ARD = False, n_Z = 250):
        """
        :X: training data X 
        :Y: training data Y, needs to be passed as array of shape (n,1)
        :param kernel: covariance function to be used. for our purposes only SE, Matern32 and Matern52 implemented
        :param ARD: if true, ARD version of kernel is used
        :param n_Z: number of pseudo inputs for a sparse GP 
        """
        if np.shape(X)[0] == len(Y):
            assert np.shape(X)[1] >= 1
        else:
            assert np.shape(X)[1] == len(Y)
            X = np.transpose(X)
            
        self.X = X
        assert np.shape(Y)[1] == 1
        self.Y = Y
                
        self.ARD = ARD
        self.trained_params = []
        self.log_likelihood = []
        self.train_crashes = 0.0
        
        if kernel == 'SE':
            self.kernel = gpflow.kernels.RBF(input_dim=np.shape(self.X)[1], ARD=self.ARD)
        elif kernel == 'Matern32':
            self.kernel = gpflow.kernels.Matern32(input_dim=np.shape(self.X)[1], ARD=self.ARD)
        elif kernel == 'Matern52':
            self.kernel = gpflow.kernels.Matern52(input_dim=np.shape(self.X)[1], ARD=self.ARD)
        else:
            print('Error: Kernel not implemented, choose other kernel')
        
        # Sparse GP:
        if np.shape(X)[0] > 2000:
            self.sparse = True
            d = np.shape(X)[1]
            z = n_Z  
            p = np.zeros(shape=(z,d))
            for ii in range(0,d):
                p[:,ii] = np.random.uniform(np.min(X[:,ii]),np.max(X[ii]),z)
            self.Z = p
            self.model = gpflow.models.SGPR(self.X, self.Y, kern=self.kernel, Z=self.Z)
        else:
            self.Z = 0
            self.sparse = False
            self.model = gpflow.models.GPR(self.X, self.Y, kern=self.kernel)
        self.model.likelihood.variance = 0.01
        self.model.compile()


    def learn_params(self):
        gpflow.train.ScipyOptimizer().minimize(self.model)

    def train(self,number = 10, largest = 5):
        '''
        Does randomized restarts of training and returns matrix with parameters of all trained models
        :param number: number of randomized restarts
        :param largest: defines upper end of interval for uniform sampling of initial values
        !! Careful, is hardcoded to implemented kernels SE, Matern32, Matern52 !!!
        '''
        if self.ARD:
            d = np.shape(self.X)[1]
        else:
            d = 1
        size = d + 1 + 1 +1 #lengthscales + kernel variance + kernel likelihood + log-likelihood
        models = np.zeros(shape=(number,size))
        for jj in range(0,number):
            if number ==1:
                self.model.likelihood.variance = 0.5
                self.model.kern.variance = 1.0
                self.model.kern.lengthscales = np.squeeze(np.ones(d))
            elif number >1:
                self.model.likelihood.variance = np.random.uniform(0,largest)
                self.model.kern.variance = np.random.uniform(0,largest)
                if d == 1:
                    self.model.kern.lengthscales = np.random.uniform(0,largest,d)
                else:
                    self.model.kern.lengthscales = np.squeeze(np.random.uniform(0,largest,d))
            try:    
                self.learn_params()
                if self.sparse:
                    models[jj,0] = self.model.read_trainables()['SGPR/likelihood/variance']
                    models[jj,1] = self.model.read_trainables()['SGPR/kern/variance']
                    models[jj,2:-1] = self.model.read_trainables()['SGPR/kern/lengthscales']    
                else:
                    models[jj,0] = self.model.read_trainables()['GPR/likelihood/variance']
                    models[jj,1] = self.model.read_trainables()['GPR/kern/variance']
                    models[jj,2:-1] = self.model.read_trainables()['GPR/kern/lengthscales']
                models[jj,-1] = self.model.compute_log_likelihood()
                if np.isnan(self.model.compute_log_likelihood()):
                    models[jj,:] = np.zeros(size)
                    self.train_crashes += 1.0
            except:
                models[jj,:] = np.zeros(size)
                self.train_crashes += 1.0
        models = models[~np.all(models == 0, axis=1)]
        self.trained_params = models[:,0:-1]
        self.log_likelihood = models[:,-1]

            
    def predict(self, X_test):
        mean, var = self.model.predict_y(X_test)
        return mean, var
    
    def evaluate(self,X_test, Y_test, extensive = False):
        '''
        Evaluates GP model on test set, returning RMSE and pseudo-loglikelihood, as well as the 
        precentage of test points in (2 and 3 sigma) bounds and average distance of these bounds. 
        Also returns standard deviation of those metrics between the random initialisations of training
        :X_test: input values of test set
        :Y_test: output values of test set
        :param extensive: if True, returns the results for 4 sigma bounds as well
        '''
        xx = X_test
        percentage_2 = np.zeros(np.shape(self.trained_params)[0])
        A_2 = np.zeros(np.shape(self.trained_params)[0])
        percentage_3 = np.zeros(np.shape(self.trained_params)[0])
        A_3 = np.zeros(np.shape(self.trained_params)[0])
        percentage_4 = np.zeros(np.shape(self.trained_params)[0])
        A_4 = np.zeros(np.shape(self.trained_params)[0])
        RMSE = np.zeros(np.shape(self.trained_params)[0])
        ll = np.zeros(np.shape(self.trained_params)[0])
        for jj in range(0,np.shape(self.trained_params)[0]):
            self.model.likelihood.variance = self.trained_params[jj,0]
            self.model.kern.variance = self.trained_params[jj,1]
            self.model.kern.lengthscales = np.squeeze(self.trained_params[jj,2:])
            mean, var = self.predict(xx)            
            for ii in range(0,len(xx[:,0])):
                A_2[jj] += (2.0*1.96)*np.sqrt(var[ii])
                A_3[jj] += 6.0*np.sqrt(var[ii])
                A_4[jj] += 8.0*np.sqrt(var[ii])
                if ((Y_test[ii] <= (mean[ii] + 1.96*np.sqrt(var[ii]))) and (Y_test[ii] >= (mean[ii] - 1.96*np.sqrt(var[ii])))):
                    percentage_2[jj] += 1.0
                if ((Y_test[ii] <= (mean[ii] + 3.0*np.sqrt(var[ii]))) and (Y_test[ii] >= (mean[ii] - 3.0*np.sqrt(var[ii])))):
                    percentage_3[jj] += 1.0
                if ((Y_test[ii] <= (mean[ii] + 4.0*np.sqrt(var[ii]))) and (Y_test[ii] >= (mean[ii] - 4.0*np.sqrt(var[ii])))):
                    percentage_4[jj] += 1.0
            percentage_2[jj] /= len(xx[:,0])
            A_2[jj] /= len(xx[:,0])
            percentage_3[jj] /= len(xx[:,0])
            A_3[jj] /= len(xx[:,0])
            percentage_4[jj] /= len(xx[:,0])
            A_4[jj] /= len(xx[:,0])
#            print(A[jj])
#            A[jj] = float(A)
            v_ll = Gauss_log_likelihood(Y_test,(mean - 2.0*np.sqrt(var)), (mean + 2.0*np.sqrt(var)))
            ll[jj] = np.sum(v_ll)
            v_SE = (Y_test - (mean))**2
            RMSE[jj] = np.mean(v_SE)**(0.5)
        i = np.argmin(RMSE)  
        if extensive:
            return RMSE[i],np.std(RMSE),ll[i],np.std(ll), percentage_2[i],np.std(percentage_2), A_2[i],np.std(A_2),percentage_3[i],np.std(percentage_3), A_3[i],np.std(A_3),percentage_4[i],np.std(percentage_4), A_4[i],np.std(A_4)
        else:
            return RMSE[i],np.std(RMSE),ll[i],np.std(ll), percentage_2[i],np.std(percentage_2), A_2[i],np.std(A_2),percentage_3[i],np.std(percentage_3), A_3[i],np.std(A_3)
    
    
    def plot(self,X_test, Y_test):
        if np.shape(self.X)[1] >= 2:
            print('Dimension of input too high - cannot plot')
            return
        xx = np.linspace(min(self.X), max(max(X_test),max(self.X)), 500)[:,None]
        mean, var = self.predict(xx)
        plt.figure(figsize=(9, 3))
        plt.plot(self.X, self.Y, 'k*', mew=2)
        plt.plot(X_test, Y_test, 'g*', mew=2)
        plt.plot(xx, mean, 'b', lw=2)
        plt.fill_between(xx[:,0], mean[:,0] - 2*np.sqrt(var[:,0]), mean[:,0] + 2*np.sqrt(var[:,0]), color='blue', alpha=0.2)
        plt.xlim(min(self.X), max(max(X_test),max(self.X)))
        plt.ylim(-5, 5)
        plt.title('Inference with GP', fontsize='x-large')
#        plt.savefig('GP inference.png')
        
        
#%%        
        
class BNNModel:
    """
    Builds basic BNN model around training data
    """
    
    def __init__(self, X: np.array, Y: np.array, architecture: list, dropout = 0.1, T = 10, tau = 1.0, lengthscale = 1., base_lr = 5e-2, gamma = 0.0001*0.25):
        """
        :X: training data X -> so far only implemented for 1D data, needs to be of shape (n,1) or (1,n)
        :Y: training data y, needs to be passed as array of shape (n,1);
        :param architecture: list of perceptrons per layer, as long as network deep
        :param dropout: probability of perceptron being dropped out
        :param T: number of samples from posterior of weights during test time
        :param tau: precision of prior
        :param lengthscale: lengthscale
        :param base_lr: initial learning rate for SGD optimizer
        :param gamma: parameter for decay of initial learning rate according to default SGD learning schedule
        """
        if np.shape(X)[0] == len(Y):
            assert np.shape(X)[1] >= 1
        else:
            assert np.shape(X)[1] == len(Y)
            X = np.transpose(X)
            
        self.X = X
        assert np.shape(Y)[1] == 1
        self.Y = Y
        
        self.dropout = dropout
        self.T = T
        self.tau = tau
        self.lengthscale = lengthscale 
        # Eq. 3.17 Gal thesis:
        self.weight_decay = ((1-self.dropout)*self.lengthscale**2)/(self.X.shape[0]*self.tau) # Don't need to dived by two as we are using squared error
        self.architecture = architecture
        
        self.model = Sequential()
        self.model.add(Dense(units=architecture[0], activation='relu', input_dim=np.shape(self.X)[1], kernel_regularizer=regularizers.l2(self.weight_decay), bias_regularizer=regularizers.l2(self.weight_decay)))
        self.model.add(Lambda(lambda x: K.dropout(x,level=self.dropout)))
        for jj in range(1,(len(architecture))):
            self.model.add(Dense(units=architecture[jj], activation='relu', input_dim=np.shape(self.X)[1], kernel_regularizer=regularizers.l2(self.weight_decay), bias_regularizer=regularizers.l2(self.weight_decay)))
            self.model.add(Lambda(lambda x: K.dropout(x,level=self.dropout)))
        self.model.add(Dense(units=1))
#        sgd = SGD(lr=base_lr, decay=gamma, momentum=0.9, nesterov=True)
        self.model.compile(loss='mean_squared_error', optimizer='adam')

    
    def train(self, epochs = 100, batch_size = 128, validation_data = ()):
        """
        Trains model
        :param epochs: defines how many times each training point is revisited during training time
        :param batch_size: defines how big batch size used is
        """
        # Might want to save model check points?!
        Early_Stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
        historyBNN = self.model.fit(self.X, self.Y, epochs=epochs, batch_size=batch_size, verbose=0, validation_data = validation_data, callbacks=[Early_Stop])
#        tl,vl = historyBNN.history['loss'], historyBNN.history['val_loss'] 
        
    def predict(self, X_test):
        y_pred = self.model.predict(X_test)
        return y_pred

    def evaluate(self, X_test, Y_test):
        '''
        Evaluates BNN model on test set, returning RMSE and pseudo-loglikelihood, as well as the 
        precentage of test points in (2,3 and 4 sigma) bounds and average distance of these bounds. 
        Also returns standard deviation of those metrics between the random initialisations of training
        :X_test: input values of test set
        :Y_test: output values of test set
        '''
        xx = X_test
        M_pred = np.zeros((self.T,len(xx[:,0])))
        for ii in range(0,self.T):    
            M_pred[ii,:] = np.squeeze(self.predict(xx))
        y_mean = np.expand_dims(np.mean(M_pred, axis=0),axis=1)
        y_var = np.expand_dims(np.var(M_pred, axis=0),axis=1) + self.tau**-1
        if not all((y_mean + 2.0*np.sqrt(y_var))>(y_mean - 2.0*np.sqrt(y_var))):#b has to be upper bound, a has to be lower bound
            print('WARNING: SOMETHING WRONG')
            print('M_pred= ', M_pred, M_pred.shape )
            print('architecture_', self.architecture, 'tau:', self.tau, 'dropout:', self.dropout, 'weight_decay:' , self.weight_decay , 'lengthscale:', self.lengthscale )
            RMSE = 10000
            ll = 0.0
            percentage_2 = 0.0
            A_2 = 0.0
            percentage_3 = 0.0
            A_3 = 0.0
            percentage_4 = 0.0
            A_4 = 0.0
            return
        percentage_2 = 0.0
        A_2 = 0.0
        percentage_3 = 0.0
        A_3 = 0.0
        percentage_4 = 0.0
        A_4 = 0.0
        for ii in range(0,len(xx[:,0])):
            A_2 += (2.0*1.96)*np.sqrt(y_var[ii])
            A_3 += 6.0*np.sqrt(y_var[ii])
            A_4 += 8.0*np.sqrt(y_var[ii])
            if ((Y_test[ii] <= (y_mean[ii] + 1.96*np.sqrt(y_var[ii]))) and (Y_test[ii] >= (y_mean[ii] - 1.96*np.sqrt(y_var[ii])))):
                percentage_2 += 1.0
            if ((Y_test[ii] <= (y_mean[ii] + 3.0*np.sqrt(y_var[ii]))) and (Y_test[ii] >= (y_mean[ii] - 3.0*np.sqrt(y_var[ii])))):
                percentage_3 += 1.0
            if ((Y_test[ii] <= (y_mean[ii] + 4.0*np.sqrt(y_var[ii]))) and (Y_test[ii] >= (y_mean[ii] - 4.0*np.sqrt(y_var[ii])))):
                percentage_4 += 1.0
        percentage_2 /= len(xx[:,0])
        A_2 /= len(xx[:,0])
        A_2 = float(A_2)
        percentage_3 /= len(xx[:,0])
        A_3 /= len(xx[:,0])
        A_3 = float(A_3)
        percentage_4 /= len(xx[:,0])
        A_4 /= len(xx[:,0])
        A_4 = float(A_4)
        # We compute the test log-likelihood
        ll = (logsumexp(-0.5 * self.tau * (Y_test - M_pred.T)**2., 0) - np.log(self.T) 
            - 0.5*np.log(2*np.pi) + 0.5*np.log(self.tau))
        ll = np.mean(ll)
        v_SE = (Y_test - y_mean)**2
        RMSE = np.mean(v_SE)**(0.5)
        return RMSE,0.0,ll,0.0, percentage_2,0.0, A_2,0.0, percentage_3,percentage_4, A_3,A_4
    


    def plot(self,X_test, Y_test):
        if np.shape(self.X)[1] >= 2:
            print('Dimension of input too high - cannot plot')
            return
        xx = np.arange(min(self.X), max(X_test), 0.05)
        M_pred = np.zeros((self.T,len(xx)))
        for ii in range(0,self.T):    
            M_pred[ii,:] = np.squeeze(self.predict(xx))
        y_mean = np.mean(M_pred, axis=0)
        y_var = np.var(M_pred, axis=0)
        y_std = y_var**(0.5)
        plt.figure(figsize=(9, 3))
        plt.plot(self.X, self.Y, 'k*', mew=2)
        plt.plot(X_test, Y_test, 'g*', mew=2)
        plt.plot(xx, y_mean, 'b', lw=2)
        plt.fill_between(xx, y_mean-1.96*y_std, y_mean+1.96*y_std, color='blue', alpha=0.2)
        plt.xlim(min(self.X), max(X_test))
        plt.ylim(-5, 5)
        plt.title('Inference with BNN', fontsize='x-large')
#        plt.savefig('BNN Inference.png')
        
    def save(self,name = 'Default_Name'):
        self.model.save(name)
        return




