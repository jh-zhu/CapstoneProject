#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 13:44:36 2019

@author: Jiahao
"""
import matplotlib.pyplot as plt


class testOL(object):
    
    def __init__(self,online_learner,X,Y,plot_length = 0):
        '''
        argument: a trained online_learner, X: test_data input, Y: test_data label
        '''
        self.online_learner = online_learner
        # test data input
        self.X = X
        # test data label
        self.Y = Y
        self.n_experts = self.online_learner.n
        if plot_length > 0:
            self.plot_length = plot_length
        else:
            self.plot_length = len(self.Y)
    
    
    def compute_weight(self):
        '''
        Compute the weihgt change of experts on test data
        '''
        weights = [[] for i in range(self.n_experts)]
        for x,y in zip(self.X,self.Y):
            self.online_learner.update_point(x,y)
            W = self.online_learner.get_weight()
            
            for i in range(self.n_experts):
                weights[i].append(W[i])
        
        self.online_learner.reset()        
        return weights
    
    
    
    def weight_plot(self,title,xlabel,ylabel):
        
        weights = self.compute_weight()
        
        experts = self.online_learner.models
        names = []
        for expert in experts:
            names.append(expert.get_name())
        
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
        cumulative_algo_loss = 0
        algo_cumulative_loss=0
        
        for x,y in zip(self.X,self.Y):
            # online learner see a point
            # when update_point is called, predictions made by experts,
            # losses occured for experts, and prediction made by algorithm 
            # are all generated. No need to call other functions, just 
            # get the statsitcs you want
            self.online_learner.update_point(x,y)
            
            # For experts: add current loss to cumulative loss
            current_model_loss = self.online_learner.get_current_loss()
            for i in range(self.n_experts):
                cumulative_model_loss[i] += current_model_loss[i]
            
            # get the prediction made by algo
            algo_prediction = self.online_learner.get_algo_prediction()
            
            # cumulate the loss made by algo
            algo_cumulative_loss += self.online_learner.loss(y,algo_prediction)
        
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
        
        if right_expert < 0 :
            # model cumulative loss
            model_cumulative_loss = [0] * self.n_experts
            for x,y in zip(self.X,self.Y):
                self.online_learner.update_point(x,y)
                model_current_loss = self.online_learner.get_current_loss()
                W_matrix.append(self.online_learner.get_weight().copy())
                for i in range(self.n_experts):
                    model_cumulative_loss[i] += model_current_loss[i]
            
            right_expert  = model_cumulative_loss.index(min(model_cumulative_loss))
            
        # step is the number of the right expert predicts right
        step = 0
        for weight in W_matrix:
            step += weight[right_expert]
        
        self.online_learner.reset()
        return right_expert, step/len(self.X)
        
        
        
    
    
            
    
            
            
        
        
        
        
        
        
        
        
        
        
        
        
        