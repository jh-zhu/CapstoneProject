#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Jiahao, Yitong
"""

import numpy as np
import matplotlib.pyplot as plt

    
def plot_weight(W,model_names = None,title=None, size = (12,4),output_path = None):
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
    
#    if text is not None:
#        plt.text(text)
    
    if output_path is not None:
        plt.savefig(output_path)
    else:
        plt.show()
        
    return fig
        
        
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
        
    return fig

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
        
    return fig

def plot_random_grid(num_of_experts, algo_abs_loss, title=None,size = (12,4),output_path = None):
    '''
    Plot algorithm absolute loss over number of exeprts for grid/random serch for comparison
    
    Input: num_of_experts: a list of different number of experts used
           algo_abs_loss: a list of algorithm absolute loss 
           size: plot size 
           output_path: string, a path to output file if want plot to be saved
    '''
    
    fig = plt.figure(figsize = size)
    _ = plt.plot(num_of_experts, algo_abs_loss, '-o')
    for a,b in zip(num_of_experts, algo_abs_loss):
        plt.annotate('%s'%(round(b,2)),xy=(a,b))

    plt.xlabel('Number of Exeprts')
    plt.ylabel('Algorithm Cumulative Loss')
    
    if title is not None:
        plt.title(title)
    
    if output_path is not None:
        plt.savefig(output_path)
    else:
        plt.show()
        
    return fig