#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
First created on Sun Apr  8 2018
Final version on Sun Jun 24 2018
@authors: Arno Blaas and Adam D. Cobb
"""

#%% 0. Import libraries and fix seed
 
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="3"

os.system('export CUDA_HOME=/opt/cuda-8.0')
os.system('export LD_LIBRARY_PATH=/opt/cuda-8.0/lib64:/opt/cuda-8.0/extras/CUPTI/lib64')

import numpy as np
import time
from matplotlib import pyplot as plt
plt.style.use('ggplot')
from codes.methods import LipModel, GPModel, BNNModel
#%matplotlib inline
import GPyOpt

#fix seed for reproducability
seed = 2000
np.random.seed(seed)
    
#%% 1. Load data

dataset = 'yacht'  #adjust depending on which dataset code is run


load_path = os.path.join('../data/UCI datasets', dataset)
train_data = np.loadtxt(os.path.join(load_path,'train.txt'))   
test_data = np.loadtxt(os.path.join(load_path,'test.txt'))

X_train = train_data[:,0:-1]
Y_train = np.expand_dims(train_data[:,-1], axis=1)

X_test = test_data[:,0:-1]
Y_test = np.expand_dims(test_data[:,-1], axis=1)

#%% 2. Initialize models

#2.0 Define parameters for different baseline models
GP_ARD = True
inducing_points = 250 #number of inducing points used for Sparse GP if data set is too large (n > 2000)

SLAC_runs = 10 #number of SLAC bounds we calculate to get view on std and robustness of sparse method

samples = 20  #number of samples of posterior weight distribution in BNNs
n_iters = 10
epochs = 50
dropout = 0.1
N_try = 200
batch_size = 128
architectures = [[1024,1024],
                 [1024,1024,1024],
                 [1024,1024,1024,1024]]


#2.1 Build GPs  
GP1 = GPModel(X_train,Y_train,kernel = 'SE', ARD=GP_ARD, n_Z = inducing_points)
GP2 = GPModel(X_train,Y_train,kernel = 'Matern32', ARD=GP_ARD , n_Z = inducing_points)
GP3 = GPModel(X_train,Y_train,kernel = 'Matern52', ARD=GP_ARD , n_Z = inducing_points)


#2.2 Build Lipschitz Models
Lip1_list = []  
for ij in range(0,SLAC_runs):
    Lip1_list.append(LipModel(X_train, Y_train))   #Sparse LACKI
Lip2 = LipModel(X_train, Y_train)  #LACKI
Lip3 = LipModel(X_train, Y_train)  #POKI


#2.3 Build BNNs - first define parameters, then calculate optimal tau and lengthscale and finally build

#For some reason first compilation of BNN always crashes after GPflow was loaded so try compiling a fake BNN to trigger and ignore the crash
BNN_try = BNNModel(X_train, Y_train, architectures[0],dropout = dropout, T = samples, tau = 0.2, lengthscale = 0.2)
try:
    BNN_try.train(epochs = epochs, batch_size = batch_size)
except:
    pass

#Do Bayesian optimisation with GPyOpt to get optimal precision tau and lengthscale l 
def BNN_objective_init(X_train, Y_train, layers ,dropout, T, X_val, Y_val,batch_size,epochs):
 
    def objective_function(solution):
        bnn = BNNModel(X_train, Y_train, layers , dropout, T , tau = solution[0][0], lengthscale = solution[0][1])
        bnn.train(epochs = epochs, batch_size = batch_size, validation_data = (X_val,Y_val))
        _, _, ll, _, _, _, _, _ , _, _, _, _  = bnn.evaluate(X_val,Y_val)
        print(ll)
        del bnn
        return -np.array(ll).reshape((1,1))
    return objective_function 

tau_list = []
lengthscale_list = []

for layers in architectures:
    objective_function = BNN_objective_init(X_train[:N_try], Y_train[:N_try] ,layers ,dropout, samples, X_train[N_try:], Y_train[N_try:],batch_size,epochs)
     
    bounds = [{'name': 'tau', 'type': 'continuous',  'domain': (0.1, 10.0), 'dimensionality': 1},
    {'name': 'lengthscale', 'type': 'continuous',  'domain': (0.001, 0.5), 'dimensionality': 1}]
    opt_bnn = GPyOpt.methods.BayesianOptimization(objective_function, domain = bounds, batch_size=1, n_cores=1)
    opt_bnn.run_optimization(max_iter=n_iters)
    opt_score = max(opt_bnn.Y_best)
    opt_score_iter = list(opt_bnn.Y_best).index(opt_score)
    opt_solution = opt_bnn.X[opt_score_iter]
    tau_list.append(opt_solution[0])
    lengthscale_list.append(opt_solution[1])
    print('done with:',layers)

BNN1_list = [BNNModel(X_train, Y_train, architectures[0],dropout = dropout, T = samples, tau = tau_list[0], lengthscale = lengthscale_list[0])]
BNN2_list = [BNNModel(X_train, Y_train, architectures[1],dropout = dropout, T = samples, tau = tau_list[1], lengthscale = lengthscale_list[1])]
BNN3_list = [BNNModel(X_train, Y_train, architectures[2],dropout = dropout, T = samples, tau = tau_list[2], lengthscale = lengthscale_list[2])]


#%% 3. Train models

time_vec = np.zeros(shape=(9,2))

#2.0 Define training parameters for different methods
POKI_Scipy_initialisations = 2
Lip_losstype = 'l1'
maxevals = 500
optimizer = 'Shubert'
cv_LACKI = 100 #how many cross validations for sparse LACKI? I.e. how often do we reshuffle ordering?
cv_POKI = 4
batch_size = 128
epochs = 10000
GP_initialisations = 10 #number of random initialisations of hyperparameters of GP in training 
GP_initialisation_sample_bound = 3   #upper bound for uniform initial hyperparameter values of GP



# # 3.1 Train GPs
t = time.time()
GP1.train(number = GP_initialisations, largest = GP_initialisation_sample_bound)
time_vec[0,0] = time.time() - t
print('Finished SE training')
print(GP1.trained_params)
t = time.time()
GP2.train(number = GP_initialisations, largest = GP_initialisation_sample_bound)
time_vec[1,0] = time.time() - t
print('Finished Matern32 training')
t = time.time()
GP3.train(number = GP_initialisations, largest = GP_initialisation_sample_bound)
time_vec[2,0] = time.time() - t
print('Finished Matern52 training')


# 3.2 Train SLAC, LAC and POKI (optional)
t0=[];
for ij in range(0,SLAC_runs):
    t = time.time()
    Lip1_list[ij].train(method='LACKI',cv = cv_LACKI)
    Lip1_list[ij].correct_support()
    t0.append(time.time() - t)
time_vec[3,0] = np.mean(t0)
time_vec[3,1] = np.std(t0)

t = time.time()
Lip2.train(method='LACKI')
time_vec[4,0] = time.time() - t
#print(Lip2.L, Lip2.trained_params)  
t = time.time()
Lip3.train(method='POKI',losstype=Lip_losstype, optimizer = optimizer, maxevals = maxevals, cv = cv_POKI, number = POKI_Scipy_initialisations)  
time_vec[5,0] = time.time() - t
#print(Lip3.L, Lip3.trained_params)   



# 3.3 Train BNNs
N_BNN_models = len(BNN1_list)
t1=0; t2=0; t3=0
try:
    BNN1_list[0].train(epochs = epochs, batch_size = batch_size)
except:
    pass
for n in range(N_BNN_models):
    t = time.time()
    BNN1_list[n].train(epochs = epochs, batch_size = batch_size)
    t1 += time.time() - t
    t = time.time()
    BNN2_list[n].train(epochs = epochs, batch_size = batch_size)
    t2 += time.time() - t
    t = time.time()
    BNN3_list[n].train(epochs = epochs, batch_size = batch_size)
    t3 += time.time() - t
    print('BNN run: ',n)
time_vec[6,0] = np.mean(t1)
time_vec[7,0] = np.mean(t2) 
time_vec[8,0] = np.mean(t3)
time_vec[6,1] = np.std(t1)
time_vec[7,1] = np.std(t2) 
time_vec[8,1] = np.std(t3)


#%% 4. Evaluate models 


#4.1 Evaluate by time taken for training of model
train_times_in_sec = {
    'GP SE': time_vec[0,:],
    'GP Matern32': time_vec[1,:],
    'GP Matern52': time_vec[2,:],
    'Lip Sparse LACKI': time_vec[3,:],
    'Lip LACKI': time_vec[4,:],
    'Lip POKI': time_vec[5,:],
    'BNN low capacity': time_vec[6,:],
    'BNN medium capacity': time_vec[7,:],
    'BNN high capacity': time_vec[8,:]
    }


#4.2 Evaluate by RMSE, LL,  % in bounds on test data, as well as Average distance between bounds

# order of results per model: RMSE, std(RMSE),... 
  #... ll, std(ll), percentage, std(percentage), Avg distance , std(Avg distance)

Lip_results = np.zeros(shape=(6,1+SLAC_runs))
for ij in range(0,SLAC_runs):
    SLAC_results = Lip1_list[ij].evaluate(X_test,Y_test)
    Lip_results[1,ij+1] = SLAC_results[4]
    Lip_results[2,ij+1] = SLAC_results[6]
    Lip_results[3,ij+1] = SLAC_results[0]
    Lip_results[4,ij+1] = SLAC_results[2]
    Lip_results[5,ij+1] = Lip1_list[ij].L


eval_results = {
    'GP SE': GP1.evaluate(X_test,Y_test),
    'GP Matern 32': GP2.evaluate(X_test,Y_test),
    'GP Matern52': GP3.evaluate(X_test,Y_test),
    'Lip Sparse LACKI mean': np.mean(Lip_results[:,1:], axis=1),
    'Lip Sparse LACKI std': np.std(Lip_results[:,1:], axis=1),
    'Lip LACKI': Lip2.evaluate(X_test,Y_test),
    'Lip POKI': Lip3.evaluate(X_test,Y_test),
    'BNN low capacity': BNN1_list[0].evaluate(X_test,Y_test),
    'BNN medium capacity': BNN2_list[0].evaluate(X_test,Y_test),
    'BNN high capacity': BNN3_list[0].evaluate(X_test,Y_test),
    }


#4.3 Evaluate by plot if 1-dimensional input space
if np.shape(X_train)[1] < 2:
    iL = np.argmin(Lip_results[3,:])
    GP1.plot(X_test,Y_test)
    GP2.plot(X_test, Y_test)
    GP3.plot(X_test, Y_test)
    Lip1_list[iL].plot(X_test, Y_test)
    Lip2.plot(X_test, Y_test)
    Lip3.plot(X_test, Y_test)
    BNN1_list[0].plot(X_test, Y_test)
    BNN2_list[0].plot(X_test, Y_test)
    BNN3_list[0].plot(X_test, Y_test)

#%% 5. Save used parameters, trained models, and results

#5.0 change directory to target destination

import datetime
now = datetime.datetime.now()
date_string = now.strftime("%Y-%m-%d-%Hh%M")   #use formatted datestring to name directory where results are saved

write_path = os.path.join('../experiments/UCI datasets', dataset, date_string + 'SLAC')
os.makedirs(write_path)
os.chdir(write_path)

#5.1 Save parameters and inputs

used_params = {
        'seed': seed,
        'GP_ARD': GP_ARD,
        'inducing_points': inducing_points,
        'GP_initialisations': GP_initialisations, 
        'GP_initialisation_sample_bound': GP_initialisation_sample_bound,         
        'dropout': dropout,
        'Samples': samples,
        'batch_size': batch_size,
        'epochs': epochs, 
        'Lip_losstype': Lip_losstype,
        'maxevals': maxevals,
        'optimizer': optimizer,
        'POKI_Scipy_initialisations': POKI_Scipy_initialisations,
        'cv_sparse_LACKI': cv_LACKI,
        'SLAC_runs': SLAC_runs,
        'cv_POKI': cv_POKI
        }
np.save('used_params.npy', used_params)
 

#5.2 Save trained models

#____5.2.1 Save trained GPs

GP1_trained = {
        'trained parameters': GP1.trained_params,  #likelihood variance, kernel variance, lengthscale(s) 
        'marginal likelihoods': GP1.log_likelihood,  #loglikelihood of trained model(s)
        'crashes during training': GP1.train_crashes
        }
GP2_trained = {
        'trained parameters': GP2.trained_params, #likelihood variance, kernel variance, lengthscale(s)
        'marginal likelihoods': GP2.log_likelihood, #loglikelihood of trained model(s)
        'crashes during training': GP2.train_crashes
        }
GP3_trained = {
        'trained parameters': GP3.trained_params, #likelihood variance, kernel variance, lengthscale(s)
        'marginal likelihoods': GP3.log_likelihood, #loglikelihood of trained model(s)
        'crashes during training': GP3.train_crashes
        }

np.save('GP_SE_trained.npy', GP1_trained)
np.save('GP_Matern32_trained.npy', GP2_trained) 
np.save('GP_Matern52_trained.npy', GP3_trained) 

# Code for loading:
#read_dictionary = np.load('GP_SE_trained.npy').item()
#print(read_dictionary['trained parameters']) # displays Lip1.L


#____5.2.2 Save trained Lipschitz models

SLAC_L = []
for kl in range(0,SLAC_runs):
    SLAC_L.append(Lip1_list[kl].L)

L_trained = {
        'Lip Sparse LACKI': SLAC_L,
        'Lip LACKI': Lip2.L,
        'Lip POKI': Lip3.L
        }
np.save('Lipschitz_constants_trained.npy', L_trained) 

# Code for loading:
#read_dictionary = np.load('Lipschitz_constants_trained.npy').item()
#print(read_dictionary['Lip Sparse LACKI']) # displays Lip1.L


#____5.2.3 Save trained BNNs
BNN_params = {
    'dropout': dropout, 'tau used (from low to high capacity)':tau_list,'lengthscale used (from low to high capacity)': lengthscale_list
    }
np.save('BNN_optimized_params.npy', BNN_params) 

BNN1_list[0].model.save_weights('low_capacity_BNN_weights.h5')
BNN2_list[0].model.save_weights('medium_capacity_BNN_weights.h5')
BNN3_list[0].model.save_weights('high_capacity_BNN_weights.h5')  
print("Saved models to disk")

# Code for loading:
#del BNN1.model  # deletes the existing model
#from keras.models import load_model 
#model = load_model('my_model.h5')


#5.3 Save results

np.save('training_times.npy', train_times_in_sec) 
np.save('results.npy', eval_results) 

with open('results.txt', 'w') as f:
    print('Time used to train model in seconds', "\n" , 
          file=f)
    for key in train_times_in_sec:       
          print( key, '%.3f' % train_times_in_sec[key][0], file=f) 
    print( "\n" ,"\n" '(RMSEs,std,LL,std, % test points inside 95% uncertainty bounds, std, Avg distance between 95% bounds, std, % test points inside 100% uncertainty bounds, std, Avg distance between 100% bounds, std ):', file=f) 
    for key in eval_results:       
          print(key, ' '.join(format(f, '.3f') for f in eval_results[key]), file=f) 

#switch back to root folder of Main_p1.py in case of new run
os.chdir("../")
os.chdir("../")
os.chdir("../")


os.chdir("../scripts")
