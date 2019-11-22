#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 19:52:07 2019

@author: Jiahao
"""

import pandas as pd
import os
import numpy as np

class learner_hpc(object):
    '''
    A on-line learner that uses results of experts tested on test data
    '''
    
    def __init__(self,source_path=None,loss_file=None,prediction_file=None,sigma=None):
        '''
        source path: The directory that contains results from hpc
        '''
        if source_path:
            self.df_loss, self.df_prediction,self.model_names = self.build_matrix(source_path)
        else:
            self.df_loss = loss_file[str(sigma)]
            self.df_prediction = prediction_file[str(sigma)]
            self.model_names = self.df_loss.columns
            
    
    def build_matrix(self,source_path):
        '''
        Build a loss and a prediction matrix
        based on all files in provided directory
        '''
        df_loss = pd.DataFrame()
        df_prediction = pd.DataFrame()
        model_names = []
        
        for f in os.listdir(source_path):
            if '.csv' in f:
                model_name = f[:-4]
                model_names.append(model_name)
                
                df_temp = pd.read_csv(os.path.join(source_path,f))
                
                loss = df_temp.loc[:,'loss']
                df_loss[model_name] = loss
                
                prediction = df_temp.loc[:,'prediction']
                df_prediction[model_name] = prediction
        
        return df_loss,df_prediction, model_names 
    
    def compute_weight_change(self):
        '''
        Compute weight change of experts within an online-learner
        
        Custimized for each online-learning algorithm
        '''
        
        pass
    
    def compute_algo_prediction(self,W):
        '''
        Compute algorithm prediction over time 
        
        Output: T * 1 algo predictions
        
        '''
        # W is 1 time step later than expert prediction
        W = W[:-1,:]
        
        # time steps and columns
        t,c = W.shape
        
        # the t * c expert prediction matrix
        P = np.array(self.df_prediction)
    
        # t *1 algo prediction list
        algo_prediction = np.zeros(t)
        
        # for each column
        for i in range(c):
            algo_prediction += W[:,i] * P[:,i]
        
        return np.array(algo_prediction)
        
    
    def compute_algo_loss(self,P,y_test):
        '''
        Compute algorithm loss over time 
        
        Output: T * 1 algo losses
        '''
            
        assert len(P) == len(y_test), "prediction and true y are not aligned"
        
        # algorithm T*1 loss
        algo_losses = (P - y_test) * (P - y_test)
        
        return algo_losses

    def compute_algo_regret(self,L):
        # cumulative losses for each expert 
        cumulative_losses = list(self.df_loss.sum())
        
        # find the expert with smallest cumulative loss 
        # best expert in hingdsight 
        val, idx = self.find_min(cumulative_losses)
        
        return sum(L) - val, idx
        
    def find_leading_expert(self,W):
        '''
        Find the leading expert over time 
        
        Return the idx of experts
        '''
        leading_expert = np.zeros(len(W),dtype=int)
        
        for i,w in enumerate(W):
            leading_expert[i] = self.find_max(w)[1]
            
        return leading_expert
    
    def get_model_names(self):
        return self.model_names
        
    def redist_loss(self,raw_losses):
        '''
        Redistribute loss accured this round
        Input: raw losses accured this round
        Output: redistributed losses
        '''
        
        # number of experts 
        n = len(raw_losses)
        
        # losses remained for each expert
        losses_remain = (1 - self.redis) * raw_losses
        
        # the weight to redistribute
        losses_redis = raw_losses - losses_remain
        
        # calculate losses after redistribution adjustment
        losses_adjusted = losses_remain + float(1/(n-1))*sum(losses_redis) - float(1/(n-1)) * losses_redis
        return losses_adjusted
        
        
        


    def normalize_weight(self,W):
        '''
        Normalize the weight to sum to 1
        
        input: W np-array
        '''
        s = sum(W)
        return W/s
    
    def find_min(self,l):
        '''
        Return the minimum value and its index for this list 
        
        '''
        return min((val, idx) for (idx, val) in enumerate(l))
    
    def find_max(self,l):
        return max((val, idx) for (idx, val) in enumerate(l))
        

class EWA_hpc(learner_hpc):
    
    def __init__(self,learning_rate,source_path=None,loss_file=None,prediction_file=None,sigma=None,redis=0):
        super().__init__(source_path,loss_file,prediction_file,sigma)
        self.learning_rate = learning_rate
        self.redis = redis
    
    def compute_weight_change(self):
        
        losses = np.array(self.df_loss) 
        
        
        n = len(self.model_names)

        # weight matrix W 
        W = []
        # initial weight
        w = np.array([1/n]*n)
        W.append(w)
        
        
        for i,loss in enumerate(losses):
            if self.redis > 0:
                l = self.redist_loss(loss)
            else:
                l = loss
            
            # update current weight
            w = w * np.exp(-self.learning_rate * l)
            # renormalize weight
            w = self.normalize_weight(w)
            
            # add to W matrix
            W.append(w)
            
        return np.array(W)

class FTL_hpc(learner_hpc):
    
    def __init__(self,source_path=None,loss_file=None,prediction_file=None,sigma=None,redis=0):
        super().__init__(source_path,loss_file,prediction_file,sigma)
        self.redis = redis
    
    def compute_weight_change(self):
        
        # row time, column experts
        losses = np.array(self.df_loss)
        
        # number of experts
        n = len(self.model_names)

        
        # weight matrix W 
        W = []
        
        # initial weight, randomly pick one
        w = np.zeros(n,dtype=int)
        w[np.random.randint(0,n)] = 1
        W.append(w)
        
        # cumulative loss
        cumulative_loss  = np.zeros(n)
        
        for i,loss in enumerate(losses):
            
            # redist loss
            if self.redis > 0:
                l = self.redist_loss(loss)
            else:
                l = loss
            
            # add to cumulative loss
            cumulative_loss  = cumulative_loss + l
            # best expert this time
            val, idx = self.find_min(cumulative_loss)
            
            w = np.zeros(n)
            w[idx] = 1
            W.append(w)
        
        return np.array(W)
    
class RWM_hpc(learner_hpc,):
    
    def __init__(self,learning_rate,source_path=None,loss_file=None,prediction_file=None,sigma=None,redis=0):
        super().__init__(source_path,loss_file,prediction_file,sigma)
        self.learning_rate = learning_rate
        self.redis = redis
    
    def compute_weight_change(self):
        
        losses = np.array(self.df_loss) 
        
        n = len(self.model_names)
        # latest weight
        w = np.array([1/n]*n)
        # weight matrix W 
        W = []
        
        
        for i,loss in enumerate(losses):
            if self.redis > 0:
                l = self.redist_loss(loss)
            else:
                l = loss
            
            median_loss = np.median(l)
            
            adjust = np.array([self.redis if v > median_loss else 1 for v in l])
            
            # update current weight
            w = w * adjust
            # renormalize weight
            w = self.normalize_weight(w)
            
            # add to W matrix
            W.append(w)
            
        return np.array(W)
        
                
                
        
            
        
        
        
        
        
        
        
        
        
        