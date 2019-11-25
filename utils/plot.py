#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Jiahao, Yitong
"""

import numpy as np
import matplotlib.pyplot as plt

#src_path = '/Users/Jiahao/Documents/classes/capstone/online_learning/'
#src_path = '/Users/yitongcai/Coding/CapstoneProject/'
#src_path='/scratch/mmy272/test/CapstoneProject/'
#os.chdir(src_path)

    
def plot_weight(W,model_names = None,title=None,size = (12,4),output_path = None):
    '''
    plot weight weight change of all experts 
    
    Input: W: weight matrix, rows are time steps, columns are experts
           model_names: a list of model names
           size: plot size 
           output_path: string, a path to output file if want plot to be saved
           
    '''
    fig = plt.figure(figsize = size)
    
    for weight in W:
        _ = plt.plot(weight)
#        if model_names is not None:
#            for i, w in enumerate(weight):
#                if w == max(weight):
#                    plt.legend(model_names[i], loc='best')
        
        
    plt.xlabel('time')
    plt.ylabel('weights')

    if title is not None:
        plt.title(title)
    
    #plt.show()
    if output_path is not None:
        plt.savefig(output_path)
    else:
        plt.show()
        
        
def plot_regret(sigmas,regrets,title=None,size = (12,4),output_path = None):
    '''
    Plot regret over noise 
    
    Input: sigmas: a list of different noises
           regrets:  regret onder different noise level
           output_path: string, a path to output file if want plot to be saved
    
    '''
    fig = plt.figure(figsize = size)
    _ = plt.scatter(sigmas,regrets)
        
    plt.xlabel('sigma')
    plt.ylabel('regrets')
    
    if title is not None:
        plt.title(title)
    
    #plt.show()
    if output_path is not None:
        plt.savefig(output_path)
    else:
        plt.show()

def plot_choose_right_expert(sigmas,percents,title=None,size = (12,4),output_path = None):
    '''
    Plot percent of time choosing right experts over noise 
    
    Input: sigmas: a list of different noises
           regrets:  regret onder different noise level
           size: plot size 
           output_path: string, a path to output file if want plot to be saved
    '''
    
    fig = plt.figure(figsize = size)
    _ = plt.scatter(sigmas,percents)
        
    plt.xlabel('sigma')
    plt.ylabel('percent of time choosing right experts')
    
    if title is not None:
        plt.title(title)
    
    #plt.show()
    if output_path is not None:
        plt.savefig(output_path)
    else:
        plt.show()
        

