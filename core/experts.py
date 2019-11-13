#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 12:27:17 2019

@author: yitongcai,Ming,Jiahao
"""
#import statsmodels.tsa.statespace.sarimax as sarimax
import numpy as np
from scipy.optimize import fmin
import math
import sklearn.linear_model as linear
import sklearn.svm as svm
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb


class experts(object):
    def  __init__(self):
        self.X_train = None    # training data for X with unchanged size
        self.y_train = None    # training data for y with unchanged size
        self.coeff = None         # There is no constants included in the coeffient list
        self.name = None
        self.model = None
        self.fitted_model=None
        self.endog = False     # if this expert is time dependent
        self.adaptive = False  # if the model is adaptive or not
        
    def get_name(self):
        return self.name
    
    def set_hyper_params(self, **params):
        self.model.set_params(**params)
            
    def get_hyper_params(self):
        return self.fitted_model.get_params()
    
    def train(self, X_train, y_train):
        '''
        :param: X_train: all records of all features (2d list)
        :param: y_train: all records of y (list [y1, ... yn])
        '''
        pass
    
    def train_update(self,new_x, new_y):
        '''
        pass new observation x and y in and remove the first observation,
        then update the model with a window size of N
        '''
        pass
    
    def predict(self,x_test):
        '''
        pass one point and make one prediction
        :param x_test: one record of all the features (list [f1, ... fk])
        :return: prediction y
        '''
        pass
    
class MLmodels(experts):
    def __init__(self):
        super().__init__()
        
    def train(self, X_train, y_train):
        self.X_train, self.y_train = X_train, y_train
        self.fitted_model =  self.model.fit(self.X_train, self.y_train)     
       
    def train_update(self, new_x, new_y):
        if self.X_train:
            self.X_train.pop(0)
            self.X_train.append(new_x)
        if self.y_train:
            self.y_train.pop(0)
            self.y_train.append(new_y)
        self.train(self.X_train, self.y_train)             
        
    def predict(self,x_test):
        '''
        Predict a single point
        input: a list of length p(# of features )
        output: a float number: prediction
        '''
        # wrap a list into a 1*p matrix
        x_test = np.array([x_test])
        pred_y = self.fitted_model.predict(x_test)
        return pred_y[0]
    
        
class LinearRegression(MLmodels):
    def __init__(self, alpha, l1_ratio):
        '''
        :param alpha: Constant that multiplies the penalty terms 
                     [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
        :param l1_ratio: 
                        np.arange(0.0, 1.0, 0.1)
        '''
        super().__init__()
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.name = 'LinearRegression'
        self.model = linear.ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, random_state=0)
     

class SVR(MLmodels):
    '''
    degree for 'poly'(default=3)
    epsilon=0.1, verbose=False'''
    def __init__(self, kernel, gamma, C, epsilon=0.1):
        '''
        :param kernel: ’rbf’,‘linear’, ‘poly’, ‘sigmoid’, ‘precomputed’ 
        :param gamma: kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’
                      measure how exactly fit the training data set (higher means more exact)
        :param C: penalty parameter C of the error term
        '''
        super().__init__()
        self.kernel = kernel
        self.gamma = gamma
        self.C = C
        self.epsilon = epsilon
        self.name = 'SVR_{}'.format(self.kernel)
        self.model = svm.SVR(kernel=self.kernel, C=self.C, gamma=self.gamma, epsilon=self.epsilon)


class RandomForest(MLmodels):
    def __init__(self, n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features):
        '''
        :param n_estimators: number of trees 
                            [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
        :param max_depth: maximum number of levels in tree
                         [int(x) for x in np.linspace(10, 110, num = 11)]
        :param min_samples_split: minimum number of samples required to split a node
                                  [2, 5, 10]
        :param min_samples_leaf: minimum number of samples required at each leaf node
                                 [1, 2, 4]
        :param max_features: number of features to consider at every split
                            ['auto', 'sqrt']
        '''
        
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.model = RandomForestRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth,
                                           min_samples_split=self.min_samples_split, 
                                           min_samples_leaf=self.min_samples_leaf, 
                                           max_features=self.max_features) 
        
        
class XGBoost(MLmodels):
    def __init__(self, max_depth, learning_rate, n_estimators, subsample, colsample_bytree, 
                 gamma, alpha, lambd):
        '''
        :param learning_rate: step size shrinkage used to prevent overfitting
                              Range is [0,1]
        :param max_depth: determines how deeply each tree is allowed to grow during any boosting round
        :param subsample: percentage of samples used per tree. Low value can lead to underfitting
        :param colsample_bytree: percentage of features used per tree. High value can lead to overfitting
        :param n_estimators: number of trees you want to build
        :param gamma: whether a given node will split based on the expected reduction in loss after the split.
                      A higher value leads to fewer splits.
        :param alpha: L1 regularization on leaf weights
        :param lambda: L2 regularization on leaf weights 
        '''
        
        super().__init__()
        self.learning_rate = learning_rate
        self.max_depth = max_depth 
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.gamma = gamma
        self.alpha = alpha
        self.lambd = lambd
        self.model = xgb.XGBRegressor(objective ='reg:squarederror', booster='gbtree', tree_method='auto',
                                      max_depth=self.max_depth, learning_rate=self.learning_rate, 
                                      n_estimators=self.n_estimators, subsample=self.subsample,
                                      colsample_bytree=self.colsample_bytree, gamma=self.gamma,
                                      eg_alpha=self.alpha, reg_lambda=self.lambd)
        
        
class AR(experts):
    '''
    AR(p) model, first trained, and then used to make prediction
    '''
    def __init__(self,p):
        super().__init__()
        self.endog = True  
        self.p=p
        self.train_data = None #training data
        self.memory_data = None #saved data used to make prediction
        self.name = 'AR{}'.format(self.p)


    def train(self,X_train, train_data):

        '''
        Train the data using train_data
        '''
        self.train_data=train_data
        self.memory_data= self.train_data[-self.p:]
        initials=[0.1]*(self.p + 1) # p phi and 1 sigma

        self.coeff=fmin(self.MLE,initials)

    def train_update(self,x,y):
        '''
        pass a point in and train the model with the new point
        '''
        self.train_data.append(y)
        self.train_data.pop(0)
        self.coeff=fmin(self.MLE,[0.1]*(self.p + 1))

    def MLE(self,initial):
        '''
        MLE used to solve the coefficients of AR(p) model
        '''

        Phi,Sigma=np.array(initial[:-1]),initial[-1]
        N = len(self.train_data)
        Z=[0]*N
        Z[0]=0
        Summation = 0
        for i in range(self.p,N):
            'X is from data initialization'
            Z[i]=self.train_data[i]-Phi.dot(np.array(self.train_data[i-self.p:i]).T)
            Summation += -1*((Z[i]**2)/(2*Sigma**2))
        res=(-1*(N-1)/2) * np.log(2*math.pi)-((N-1)/2) * np.log(Sigma **2) + Summation
        return -res

    def predict(self,x_test):
        '''
        pass one point and make one prediction
        '''
        return np.flip(np.array(self.coeff[:-1])).dot(np.array(self.memory_data[-self.p:]).T)
        # It's a AR model, get the first argument (x)
        
    def update_data(self,x,y):
        '''
        update endog data
        '''
        self.memory_data.append(y)
        self.memory_data.pop(0)
        return


# =============================================================================
# class SARIMAX(experts):
#     '''
#     SARIMAX((p,d,q), (P,D,Q,m)) model is ARIMA + Seasonality + Exogeneous X
#     SARIMAX((p,d,q), (0,0,0,1)) is ARIMAX(p,d,q)
#     '''
#     def __init__(self,p,d,q,P,D,Q,m):
#         '''
#         :param p: order of AR model
#         :param d: order of differece
#         :param q: order of MA model
#         :param P: order of AR model of seasonality
#         :param D: order of differece of seasonality
#         :param Q: order of MA model of seasonality
#         :param m: number of periods in each season ex. daily:365 / quarter:4 / month:12 / annual:1
#         '''
#         super().__init__()
#         self.p=p 
#         self.d=d
#         self.q=q
#         self.P=P 
#         self.D=D
#         self.Q=Q
#         self.m=m
#         self.name = 'SARIMA(({},{},{}),({},{},{},{}))'.format(self.p, self.d, self.q,
#                                                               self.P, self.D, self.Q, self.m)
#         
#         # adaptive is turned on for this model 
#         self.adaptive = True
#     
#     def set_hyper_params(self, **params):
#         self.model.update(**params)
#      
#     def train(self, X_train, y_train):
#         self.X_train, self.y_train = X_train, y_train
#         self.model = sarimax.SARIMAX(endog=self.y_train, exog=self.X_train, order=(self.p, self.d, self.q),
#                                      seasonal_order=(self.P, self.D, self.Q, self.m),trend='c')
#         self.fitted_model = self.model.fit(disp=0)
#         self.coeff = self.fitted_model.params[1:-1]
#         
#     def train_update(self, new_x, new_y):
#         # decide if new_x is the first input for X_train 
#         if self.X_train:
#             self.X_train.pop(0)
#             self.X_train.append(new_x)
#         else:
#             if new_x:
#                 self.X_train = [new_x]
#             # This is for SARIMA case without exogenous input X, in this case, new_x is None
#             else:
#                 self.X_train = new_x
#                 
#          # decide if new_y is the first input for y_train
#         if self.y_train is not None:
#             self.y_train.pop(0)
#             self.y_train.append(new_y)
#         else:
#             self.y_train = [new_y]
#         # re-train the whole new input X and y (problem: slow)    
#         self.train(self.X_train, self.y_train)
#    
#     def predict(self,x_test):
#         train_data_length = len(self.X_train)
#         # wrap x_test
#         x_test = np.array([x_test])
#         pred_y = self.fitted_model.predict(start=train_data_length,end=None, exog=x_test)
#         return pred_y[0]
# 
# 
# =============================================================================