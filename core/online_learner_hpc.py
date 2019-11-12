#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 19:52:07 2019

@author: Jiahao
"""

import pandas as pd
import os
import math
import numpy as np

class learner_hpc(object):
    '''
    A on-line learner that uses results of experts tested on test data
    '''
    
    def __init__(self,source_path):
        '''
        source path: The directory that contains results from hpc
        '''
        self.df_loss, self.df_prediction,self.model_names = self.build_matrix(source_path)
        
    
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
        pass
    
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
        


class EWA_hpc(learner_hpc):
    
    def __init__(self,source_path,learning_rate,redis=0):
        super().__init__(source_path)
        self.learning_rate = learning_rate
        self.redis = redis
    
    def compute_weight_change(self):
        
        losses = np.array(self.df_loss) 
        
        
        n = len(self.model_names)
        # latest weight
        w = [1/n]*n
        # weight matrix W 
        W = [w]
        
        # for each row (losses at that time step)
        for i,row in self.df_loss.iterrows():
            
            # losses at this round
            losses = np.array(row)
            # if redist
            if self.redis > 0:
                losses = self.redist_loss(losses)
            
            # update weight this round
            for j,weight in enumerate(w):
                w[j] = weight * math.exp(-self.learning_rate*losses[j])
            
            # normalize 
            w = self.normalize_weight(w)
            
            # add to W matrix
            W.append(w)
        return W

class FTL_hpc(learner_hpc):
    
    def __init__(self,source_path,redis=0):
        super().__init__(source_path)
        self.redis = redis
    
    def compute_weight_change(self):
        
        # row time, column experts
        losses = np.array(self.df_loss)
        
        # number of experts
        n = len(self.model_names)
        # number of time steps
        T = len(losses)
        
        # weight matrix W 
        W = []
        
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
            val, idx = min((val, idx) for (idx, val) in enumerate(cumulative_loss))
            
            w = np.zeros(n)
            w[idx] = 1
            W.append(w)
        
        return np.array(W)
        
                
                
        
            
        
        
        
        
        
        
        
        
        
        