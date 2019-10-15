#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 13:44:36 2019

@author: Jiahao
"""
import matplotlib.pyplot as plt


class testOL(object):
    
    def __init__(self,online_learner,test_data,plot_length = 0):
        
        self.online_learner = online_learner
        self.test_data = test_data
        self.n_experts = self.online_learner.n
        
        if plot_length > 0:
            self.plot_length = plot_length
        else:
            self.plot_length = len(self.test_data)
    
    
    def weight_plot(self,title,xlabel,ylabel):
        '''
        plot weight change of experts over time 
        '''
        
        experts = self.online_learner.models
        names = []
        for expert in experts:
            names.append(expert.get_name())
        
        weights = [[] for i in range(self.n_experts)]
        for data in self.test_data:
            self.online_learner.update_point([data],[data])
            W = self.online_learner.get_weight()
            
            for i in range(self.n_experts):
                weights[i].append(W[i])
        
        self.online_learner.reset()
        # plot 
        for weight in weights:
            plt.plot(weight[:self.plot_length])
        plt.legend(names)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()
        
        
    def compute_regret(self):
        # cumulative loss of each expert 
        cumulative_model_loss = [0] * self.n_experts
        # loss by algorithm
        algo_cumulative_loss = 0
        
        for data in self.test_data:
            self.online_learner.update_point([data],[data])
            
            current_model_loss = self.online_learner.get_current_loss()
            for i in range(self.n_experts):
                cumulative_model_loss[i] += current_model_loss[i]
            
            algo_prediction = self.online_learner.get_algo_prediction()
            
            algo_cumulative_loss += self.online_learner.loss(data,algo_prediction)
        
        self.online_learner.reset()
        # compute regret (difference between algorith and best expert)
        return algo_cumulative_loss - min(cumulative_model_loss)
    
    def compute_choose_right_expert(self,right_expert = -1):
        '''
        input:  the index of "right" expert. If not provided, default to use the 
        best expert in hindsight
        '''
        # weight matrix
        W_matrix = []
        
        # model cumulative loss
        model_cumulative_loss = [0] * self.n_experts
        
        for data in self.test_data:
            self.online_learner.update_point([data],[data])
            model_current_loss = self.online_learner.get_current_loss()
            W_matrix.append(self.online_learner.get_weight().copy())
            for i in range(self.n_experts):
                model_cumulative_loss[i] += model_current_loss[i]
        
        if right_expert < 0 :
            right_expert  = model_cumulative_loss.index(min(model_cumulative_loss))
        
        step = 0
        for weight in W_matrix:
            step += weight[right_expert]
        
        self.online_learner.reset()
        return step/len(self.test_data)
        
        
        
    
    
            
    
            
            
        
        
        
        
        
        
        
        
        
        
        
        
        