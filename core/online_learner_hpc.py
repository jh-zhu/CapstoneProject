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
    
    def redist_loss(self,raw_loss):
        '''
        Redistribute loss accured this round
        Input: raw losses accured this round
        Output: redistributed losses
        '''
        
        n = len(raw_loss)
        losses = [0] * n
        for i,r_l in enumerate(raw_loss):
            redist_loss = r_l * self.redis # loss to redistribute
            remain_loss = r_l - redist_loss # loss remained to that expert 
            redist_loss_each = redist_loss / n
            for j in range(n):
                if j==i:
                    losses[j] += remain_loss
                    losses[j] += redist_loss_each
                else:
                    losses[j] += redist_loss_each
        
        return losses

    def normalize_weight(self,W):
        '''
        Normalize the weight to sum to 1
        '''
        s = sum(W)
        for i,w in enumerate(W):
            W[i] = w/s
        return W
        


class EWA_hpc(learner_hpc):
    
    def __init__(self,source_path,learning_rate,redis=0):
        super().__init__(source_path)
        self.learning_rate = learning_rate
        self.redis = redis
    
    def compute_weight_change(self):
        n = len(self.model_names)
        # latest weight
        w = [1/n]*n
        # weight matrix W 
        W = [w]
        
        # for each row (losses at that time step)
        for i,row in self.df_loss.iterrows():
            
            # losses at this round
            losses = list(row)
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
        n = len(self.model_names)
        # latest weight, randomly pick 1 expert at the beginning
        w = [0]*n
        w[np.random.randint(0,n,1)[0]] = 1
        # weight matrix W 
        W = [w]
        # cumulative loss
        cumulative_loss  = [0] *n
        
        # for each row (losses at that time step)
        for i,row in self.df_loss.iterrows():
            
            # losses at this round
            losses = list(row)
            # if redist
            if self.redis > 0:
                losses = self.redist_loss(losses)
            
            # add losses to cumulative losses
            for j,c_loss in enumerate(cumulative_loss):
                cumulative_loss[j] = c_loss + losses[j]
                
            # update weight this round
            w = [0]*n
            val, idx = min((val, idx) for (idx, val) in enumerate(cumulative_loss))
            w[idx] = 1
            
            # add to W matrix
            W.append(w)
        return W
                
                
        
            
        
        
        
        
        
        
        
        
        
        