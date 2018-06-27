#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
First created on Tue Mar 20 2018
Final version on Sun Jun 24 2018

@authors: Arno Blaas and Adam D. Cobb
"""

import time
import numpy as np





def fct_minimise_Malherbe(fct,d,a,b,L=1.,delta = 0.01, errthresh =0.0002,maxevals=100):
    #Lipschhitz optimisation a la Malherbe but minimising instead of maximising
    #fct: function handle
    #d: dimension
    #[a,b]: domain interval for each dimension
    #L: Lipschitz constant of fct
    ti = time.time()
    #0. Calculate upper bound variables for stopping criterion according to Theorem 15
    diam = np.absolute(b-a)*np.sqrt(d)
    c_k = 0.01
    k = 1.1
    C_k = ((c_k)*(diam*0.5)**(k-1.0)/(8*L))**d
    
    #1. Initialization:
    
    x = np.random.uniform(a,b,d)
    f = fct(x)
  
    #2. Loop:
    t = 1
    if d > 1:
        while t < maxevals:
            x_temp  = np.random.uniform(a,b,d)
            if (np.amax(f - L*np.linalg.norm((x - x_temp),axis = 1)) <= np.amin(f)):                  
                x = np.transpose(np.c_[np.transpose(x), x_temp])
                f = np.append(f,fct(x_temp))
                t +=1
            if (L * diam * 2.0**k * 0.5 * (1 + C_k*((t*(2**(d*(k-1))-1))/(np.log(t/delta) + 2*(2*np.sqrt(d)**d))))**(-k/(d*(k-1)))) < errthresh:
                break
        x_min = x[np.argmin(f),:]
        ti = time.time() - ti
        
    elif d == 1:
        while t < maxevals:
            x_temp  = np.random.uniform(a,b,d)
            if (np.amax(f - L*np.linalg.norm((x - x_temp))) <= np.amin(f)):                  
                x = np.transpose(np.c_[np.transpose(x), x_temp])
                f = np.append(f,fct(x_temp))
                t +=1
            if (L * diam * 2.0**k * 0.5 * (1 + C_k*((t*(2**(d*(k-1))-1))/(np.log(t/delta) + 2*(2*np.sqrt(d)**d))))**(-k/(d*(k-1)))) < errthresh:
                break
                        
            t +=1
        x_min = x[np.argmin(f)]
        ti = time.time() - ti
        
    return x_min ,np.amin(f), t, ti

def fct_minimise_Shubert(fct,a,b,L=1.,errthresh =0.0002,maxevals=100,grid_t=[],grid_fctvals=[]):
    #Lipschhitz optimisation a la Shubert
    #fct: function handle
    #I: domain interval
    #L: Lipschitz constant
    s = np.linspace(a,b,1000)
    #fig=plt.figure()
    #plt.plot(s,fct(s))
    if grid_t ==[]:
        grid_t = np.array([a,b])
        grid_fctvals = np.array([ fct(grid_t[0]), fct(grid_t[1])])
            
    n = grid_t.size
    argmins_floor = np.zeros(n);
    minvals_floor = np.zeros(n);

    abp = shiftadd(grid_t,grid_t)
    abm = shiftdiff(grid_t,grid_t)
    fabp = shiftadd(grid_fctvals,grid_fctvals)
    fabm = shiftdiff(grid_fctvals,grid_fctvals)
    argmins_floor = .5*(abp + fabm / L  ) # calculates array with x-values of locations of bottoms of "valleys" of lower bound
    minvals_floor = .5*(fabp + L *abm )  # calculates array with y-values of bottoms of "valleys" of lower bound
                    
                    
                    #  for i in range(n-1):
                    #      argmins_floor[i] = (grid_t[i] +grid_t[i+1])/2 - (grid_fctvals[i+1]-grid_fctvals[i])/(2*L);
                    #      minvals_floor[i] = (grid_fctvals[i+1]+grid_fctvals[i])/2 - L * (grid_t[i+1] -grid_t[i]);
                    #
    curr_fmin = np.min(grid_fctvals)
    ind = np.argmin(minvals_floor)
    xi = argmins_floor[ind]
    fxi = fct(xi)
    bxi = np.amin(minvals_floor)
    err = np.absolute(curr_fmin - bxi)
    n=0
    while (n <= maxevals and err > errthresh):
        n+=1
                                    #  plt.plot(s,fct(s))
                                    #  predf=KI_predict_1d(s,grid_t,grid_fctvals,L)
                                    #print(grid_t)
                                    # plt.plot(s,predf[2])
                                    
                                    
                                    
                                    #  plt.plot(grid_t,grid_fctvals,'+')
                                    
                                    #   plt.show()
                                    #print('Minvals_floor:',minvals_floor)
                                    #print('New point to be evaluated:',xi)
                                    #print("at index:",ind)
                                    #refine grid:
        ind_crit = np.argmax(grid_t > xi) #xi should never be smaller than the smallest nor larger than the largest element in grid_t
                                    #splice!(grid_t,ind_crit,xi)
        grid_t= np.insert(grid_t,ind_crit,xi)
                                        #splice!(grid_t,ind_crit,[xi;grid_t[ind_crit]])
        grid_fctvals = np.insert(grid_fctvals,ind_crit,fxi)
                                        #(rem: the new value in the grid now is at pos crit_ind in the grid)
                                        #refine floor:
        i = ind_crit-1
        newxileft = (grid_t[ind_crit-1] +grid_t[ind_crit])/2 - (grid_fctvals[ind_crit]-grid_fctvals[ind_crit-1])/(2.*L);
        newxiright = (grid_t[ind_crit] +grid_t[ind_crit+1])/2 - (grid_fctvals[ind_crit+1]-grid_fctvals[ind_crit])/(2.*L);
                                            #splice!(argmins_floor,i,[newxileft;newxiright])
                                            #print("argims floor (pre):",argmins_floor)
        argmins_floor = np.insert(argmins_floor,i,newxileft)
        argmins_floor[i+1] = newxiright;
        newfloorxileft = .5*(grid_fctvals[ind_crit]+grid_fctvals[ind_crit-1] - L * (grid_t[ind_crit] -grid_t[ind_crit-1]));
        newfloorxiright = .5*(grid_fctvals[ind_crit+1]+grid_fctvals[ind_crit] - L * (grid_t[ind_crit+1] -grid_t[ind_crit]));
        minvals_floor = np.insert(minvals_floor,i,newfloorxileft)
        minvals_floor[i+1] = newfloorxiright
                                                #print("argims floor (post):",argmins_floor)
                                                #splice!(minvals_floor,i, [newfloorxileft;newfloorxiright])
        curr_fmin = np.min(grid_fctvals)
        ind = np.argmin(minvals_floor)
        xi = argmins_floor[ind]
        fxi = fct(xi)
        bxi = np.amin(minvals_floor)
        err = abs(curr_fmin - bxi)
    i = np.argmin(grid_fctvals)
    curr_fmin = grid_fctvals[i]
    curr_argmin = grid_t[i]
    return curr_argmin,curr_fmin,n

def vecmetric_maxnorm(x,y):
    return np.max(np.abs(x-y),1)
def vecmetric_maxnorm_1d(x,y):
    return np.abs(x-y)

def shiftdiff(x,y):
    #for vecs x,y compute the x[i]-y[i+1] and store in new vector z,x,y must have same length
    #z has 1 lower dim than x,y
    z = x - np.roll(y,-1)
    return z[0:z.size-1]

def shiftadd(x,y):
    #for vecs x,y compute the x[i]-y[i+1] and store in new vector z,x,y must have same length
    #z has 1 lower dim than x,y
    z = x + np.roll(y,-1)
    return z[0:z.size-1]

def KI_predict_1d(x,sample_s,sample_f,L,epsilon=0.,fct_metric_inp=vecmetric_maxnorm_1d):
    #x is assumed to be a matrix of col vector inputs
    #pred will be a vector of prediction values
    n=x.size
    ns = sample_f.size;
    ceilpred = np.ones(n);
    floorpred = np.ones(n);
    pred =np.ones(n);
    err =np.Inf;
            #go through test input by test input:
    for i in range(n):
                #now ith row of x stacked next to itself for each
                #tex:
        X =x[i]*np.ones(ns)
                #take abs-differences:
        m_vec=L * fct_metric_inp(X,sample_s) #all the distances of inp in one row vec
        floorpred[i] = np.max(sample_f -epsilon  - m_vec);
        ceilpred[i]  = np.min(sample_f +epsilon+ m_vec);
        pred[i] =(ceilpred[i]+floorpred[i])/2;
     
    err = (ceilpred-floorpred)/2;        
    return pred,err,floorpred,ceilpred

def KI_predict(x,sample_s,sample_f,L,epsilon=0.,fct_metric_inp=vecmetric_maxnorm):
    #x is assumed to be a matrix of col vector inputs
    #pred will be a vector of prediction values
    n = np.shape(x)[0]; #number of test inputs
    ns = sample_f.size;
    ceilpred = np.ones(n);
    floorpred = np.ones(n);
    pred =np.ones(n);
    err =np.Inf;
            #go through test input by test input:
    for i in range(n):
                #now ith row of x stacked next to itself for each
                #tex:
        X =np.tile(x[i,:],(ns,1))
                #take abs-differences:
        m_colvec=L * fct_metric_inp(X,sample_s) #all the distances of inp in one row vec
        floorpred[i] = np.max(sample_f -epsilon  - m_colvec);
        ceilpred[i]  = np.min(sample_f +epsilon+ m_colvec);
        pred[i] =(ceilpred[i]+floorpred[i])/2;
        
        
            
    err = (ceilpred-floorpred)/2;
            
    return pred,err,floorpred,ceilpred



def Lip_likelihood(y: np.array,a : np.array,b : np.array) -> np.array:
    
    '''
    calculates vector of likelihoods of values y under assumption of a distribution for
    which 95% of the probability mass fall uniformly between a and b and the 
    rest of the probability mass is in gaussian tails starting at <a and >b,
    with mean 0.5*(a+b) and sigma = 0.25*(b-a) 
    '''
    
    assert all(b>a) #b has to be upper bound, a has to be lower bound
    assert np.shape(b) == np.shape(a)
    assert np.shape(b) == np.shape(y)
    n = len(y)
    p = np.zeros(n)
    for ii in range(0,n):
        if ((y[ii] >= a[ii]) & (y[ii] <= b[ii])):
            p[ii] = 0.9544997/(b[ii]-a[ii])
        else:
            p[ii] = (2*np.pi*(0.25*(b[ii]-a[ii]))**2)**(-0.5)*np.exp(-0.5*(y[ii]-0.5*(a[ii]+b[ii]))**2/(0.25*(b[ii]-a[ii]))**2)
    return p

def Gauss_likelihood(y: np.array,a : np.array,b : np.array) -> np.array:
    
    '''
    calculates vector of likelihoods of values y under assumption of
    gaussian distribution with with mean 0.5*(a+b) and sigma = 0.25*(b-a) 
    '''
    
    assert all(b>a) #b has to be upper bound, a has to be lower bound
    assert np.shape(b) == np.shape(a)
    assert np.shape(b) == np.shape(y)
    n = len(y)
    p = np.zeros(n)
    for ii in range(0,n):
        p[ii] = (2*np.pi*(0.25*(b[ii]-a[ii]))**2)**(-0.5)*np.exp(-0.5*(y[ii]-0.5*(a[ii]+b[ii]))**2/(0.25*(b[ii]-a[ii]))**2)
    return p

def Gauss_log_likelihood(y: np.array,a : np.array,b : np.array) -> np.array:
    
    '''
    calculates vector of likelihoods of values y under assumption of
    gaussian distribution with with mean 0.5*(a+b) and sigma = 0.25*(b-a) - this means for a = mean - 2.0*std and b = mean + 2.0*std 
    '''
    
    if not all(b>a):#b has to be upper bound, a has to be lower bound
        print('WARNING: ADDED SMALL NOISE TO BOUNDS')
        b += 0.001
        a -= 0.001
    assert np.shape(b) == np.shape(a)
    assert np.shape(b) == np.shape(y)
#    print(np.shape(y))
    n = len(y)
    p = np.zeros(n)
    for ii in range(0,n):
        p[ii] = -0.5*np.log(2*np.pi) - np.log(0.25*(b[ii]-a[ii])) - 0.5*(y[ii]-0.5*(a[ii]+b[ii]))**2/(0.25*(b[ii]-a[ii]))**2
    return p



def Lip_log_likelihood(y: np.array,a : np.array,b : np.array) -> np.array:
    
    '''
    calculates vector of likelihoods of values y under assumption of a distribution for
    which 95% of the probability mass fall uniformly between a and b and the 
    rest of the probability mass is in gaussian tails starting at <a and >b,
    with mean 0.5*(a+b) and sigma = 0.25*(b-a) 
    '''
    
#    assert all(b>a) #b has to be upper bound, a has to be lower bound
    assert np.shape(b) == np.shape(a)
    assert np.shape(b) == np.shape(y)
#    print(np.shape(y))
    n = len(y)
    p = np.zeros(n)
    for ii in range(0,n):
        if ((y[ii] >= min(a[ii],b[ii])) & (y[ii] <= max(a[ii],b[ii]))):
            p[ii] = np.log(0.9544997/(max(a[ii],b[ii])-min(a[ii],b[ii])))
        else:
            p[ii] = -np.log(0.25*(max(a[ii],b[ii])-min(a[ii],b[ii]))) -0.5*np.log(2*np.pi) -0.5*(y[ii]-0.5*(a[ii]+b[ii]))**2/(0.25*(max(a[ii],b[ii])-min(a[ii],b[ii])))**2
    return p