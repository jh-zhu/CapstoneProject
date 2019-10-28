#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 01:42:35 2019

@author: yitongcai
"""
from core.experts import *

import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.grid_search import GridSearchCV


class Hyperparams_Optimization(object):
    
    def  __init__(self, base_model, X_train, y_train, **kwargs):
        self.base_model = base_model
        self.X_train = X_train
        self.y_train = y_train
        
        
    def RandomSearch(self, **kwargs):
        random_range = {k:v for k, v in kwargs.items()}
        ''' Use the random grid to search for best hyperparameters'''
        # Random search of parameters, using 5 fold cross validation, 
        # search across 100 different combinations, and use all available cores
        model_random = RandomizedSearchCV(estimator=self.base_model, param_distributions=random_grid,
                                          n_iter = 100, scoring='neg_mean_squared_error', 
                                          cv = 5, random_state=42, n_jobs=-1, return_train_score=True)
        
    
        # Fit the random search model
        model_random.fit(self.X_train, self.y_train);
        model_random.best_estimator_
        model_random.best_params_
        model_random.best_score_
        
    def GridSearch(self, **kwargs):
        grid_range = {k:v for k, v in kwargs.items()}
        model_grid = GridSearchCV(estimator=self.base_model, param_distributions=grid_range,
                                  n_iter = 100, scoring='neg_mean_squared_error', 
                                  cv = 5, random_state=42, n_jobs=-1, return_train_score=True)
        model_grid.fit(self.X_train, self.y_train)
        model_grid.best_estimator_
        model_grid.best_params_
        model_grid.best_score_
        
from sklearn.model_selection import cross_val_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

param_hyperopt= {
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(1)),
    'max_depth': scope.int(hp.quniform('max_depth', 5, 15, 1)),
    'n_estimators': scope.int(hp.quniform('n_estimators', 5, 35, 1)),
    'num_leaves': scope.int(hp.quniform('num_leaves', 5, 50, 1)),
    'boosting_type': hp.choice('boosting_type', ['gbdt', 'dart']),
    'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0),
    'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
}

    def hyperopt(self, param_space, X_train, y_train, X_test, y_test, num_eval):
        
        def objective_function(params):
            clf = lgb.LGBMClassifier(**params)
            score = cross_val_score(clf, X_train, y_train, cv=5).mean()
            return {'loss': -score, 'status': STATUS_OK}
    
        trials = Trials()
        best_param = fmin(objective_function, 
                          param_space, 
                          algo=tpe.suggest, 
                          max_evals=num_eval, 
                          trials=trials,
                          rstate= np.random.RandomState(1))
        loss = [x['result']['loss'] for x in trials.trials]
        
        best_param_values = [x for x in best_param.values()]
        
        if best_param_values[0] == 0:
            boosting_type = 'gbdt'
        else:
            boosting_type= 'dart'
        
        clf_best = lgb.LGBMClassifier(learning_rate=best_param_values[2],
                                      num_leaves=int(best_param_values[5]),
                                      max_depth=int(best_param_values[3]),
                                      n_estimators=int(best_param_values[4]),
                                      boosting_type=boosting_type,
                                      colsample_bytree=best_param_values[1],
                                      reg_lambda=best_param_values[6],
                                     )
                                      
        clf_best.fit(X_train, y_train)
        
        print("##### Results")
        print("Score best parameters: ", min(loss)*-1)
        print("Best parameters: ", best_param)
        print("Test Score: ", clf_best.score(X_test, y_test))
        print("Parameter combinations evaluated: ", num_eval)
        
        return trials
        
        
        
        
        
 '''Random Forest''' 
 # Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
#First create the base model to tune
rf = RandomForestRegressor(random_state = 42) 
 
hyper_opt = Hyperparams_Optimization(rf, X_train, y_train)    
hyper_opt.RandomSearch(rf, n_estimators=n_estimators, max_features=max_features, max_depth = max_depth,
             min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf,bootstrap=bootstrap)
             
             
             
             
             
             
             
             
             
             
             
         