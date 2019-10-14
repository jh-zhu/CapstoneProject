#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Jiahao, Ming, Yitong
"""

import random
import math

'''
This is the parent online learner module. Different online learners can inherit from this parent
'''
class learner(object):
    '''
    Online learner takes several models/experts and train them. Then, it takes point one by one, makes prediction,
    compute loss for each expert and reassign weights of experts'''

    def __init__(self,models,redis=0):
        '''
        Super Constructor
        input: models = a list of experts
               redis: the portpotion of weight to redistribute ([0,1])
        instance variable: n = number of experts
                W = an array of weight of each expert
        '''
        self.models = models
        self.n = len(models)
        self.W = [1] * self.n # initial weight for each model is 1
        self.redis = redis
        
    def train(self, train_data):
        '''
        Train all the experts using training data
        input: train_data = training data
        '''
        for mod in self.models:
            mod.train(train_data)
    
    def get_weight(self):
        '''
        return the weights of experts
        '''
        return self.W
    
    def update_sequence(self,X,Y):
        '''
        Pass a list of points to the experts and do all the work. ie. make prediction, adjust weights, etc.
        input: X - 2D matrix, rows as records, columns as features. Y - A list of labels
        output: return nothing
        '''
        for x,y in zip(X,Y):
            self.update_point(x,y)
            
    def update_point(self,x,y): 
        '''
        See a point and make an update
        input: x = a list of feature values of a point, y the label value
        '''
        predictions = self.predict(x)
        raw_losses=[None]*self.n
        
        for i,pr in enumerate(predictions):
            raw_losses[i]=self.loss(y,pr)
        
        if self.redis ==0:
            losses = raw_losses
        else:
            losses = self.redist_weight(raw_losses)
            
        self.update_weight(losses)
        
    def update_weight(losses):
        '''
        Based on the loss of each expert prediction, update the weight assigned to each expert
        '''
        
    
    def vote(self,y_predicts):
        '''Majority vote. Return the prediction given by higest vot'''
        count = {}
        for i,p in enumerate(y_predicts):
            count[p] = count[p] + self.W[i] if p in count.keys() else self.W[i]  
        
        h_vote = max(count.values())
        predict = []
        
        for k,v in count.items():
            if v == h_vote:
                predict.append(k)
        return random.choice(predict)
    
    def normalize_weight(self):
        '''
        Normalize the weight to sum to 1
        '''
        s = sum(self.W)
        for i,w in enumerate(self.W):
            self.W[i] = self.W[i]/s
    
    def algo_predict(self,x):
        '''
        Pridiction made by the online learning algorithm
        input: x a list of feature values of a record
        output: a prediction made by the algorithm
        '''
        predictions = self.predict(x)
        predict = 0
        for w,v in zip(self.W,predictions):
            predict += w*v
        
        return predict
    
    def predict(self,x):
        '''
        Pass a point, each model makes a prediction
        input: x a list of feature values of a record
        output: a list of predictions made by each expert
        '''
        predictions = [None] * self.n
        for i,model in enumerate(self.models):
            predictions[i] = model.predict(x)
            
        return predictions
    
    def loss(self,label,y_predict):
        '''
        Compute the loss given true and predicted y
        input: label=true y 
                y_predict=predicted y
        output: L2 of the prediction error        
        '''
        return (label - y_predict)*(label - y_predict)
    
    
    def redist_weight(self,raw_loss):
        '''
        Redistribute loss accured this round
        Input: raw losses accured this round
        Output: redistributed losses
        '''
        losses = [0] * self.n
        for i,r_l in enumerate(raw_loss):
            redist_loss = r_l * self.redis # loss to redistribute
            remain_loss = r_l - redist_loss # loss remained to that expert 
            redist_loss_each = redist_loss / self.n
            for j in range(self.n):
                if j==i:
                    losses[j] += remain_loss
                    losses[j] += redist_loss_each
                else:
                    losses[j] += redist_loss_each
        
        return losses
    
    
class exponential_weighted_average(learner):

    def __init__(self,models,learning_rate,redis=0):
        super().__init__(models,redis)
        self.learning_rate = learning_rate
    
    def update_weight(self,losses):
        
        for i in range(self.n):
            self.W[i] = self.W[i] * math.exp(-self.learning_rate * losses[i])
        
        self.normalize_weight()


class follow_the_lead(learner):
    '''
    Follow the expert that has the least cumulative loss
    '''
    def __init__(self,models,redis=0):
        super().__init__(models,redis)
        self.cum_loss = [0] * self.n #cumulative loss
        
        
    def update_weight(self,losses):
        '''
        Find the expert that has the least cumulative loss, and assign all the weight to
        it'''
        
        for i,loss in enumerate(losses):
            self.cum_loss[i] += loss
        val, idx = min((val, idx) for (idx, val) in enumerate(self.cum_loss))
        
        self.W = [0] * self.n
        self.W[idx]= 1 
        

        
        
        
