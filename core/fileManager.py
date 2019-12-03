#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 14:14:58 2019

@author: mingmingyu
"""
import pandas as pd
import os, shutil
import numpy as np
'''
srun python3 /scratch/mmy272/test/CapstoneProject/core/fileManager.py after all the computation is done.
'''

class fileName(object):
    
    def __init__(self):
        self.data_folder = '/scratch/mmy272/test/CapstoneProject/data/'
        self.output_folder = '/scratch/mmy272/test/output/'
        self.output_combine='/scratch/mmy272/test/output_combine/'
        
    def combine_files(self):
        '''
        combine all the csv files into 2 excels, one prediction and one loss,
        then delete all the folders 
        output files:
            1. prediction.xlsx has n number of sheets, n=# sigma, in each sheet,
            there are m columns, m=# experts
            2. loss.xlsx has n number of sheets, n=# sigma, in each sheet,
            there are m columns, m=# experts
        '''
        prediction = self.output_combine+ 'prediction/'
        loss = self.output_combine + 'loss/'
        
        if not os.path.exists(prediction):
            os.makedirs(prediction)
        if not os.path.exists(loss):
            os.makedirs(loss)
        
        for sigma in os.listdir(self.output_folder):
            if sigma=='.DS_Store':
                continue
            df_prediction = pd.DataFrame()
            df_loss = pd.DataFrame()
            
            for result in os.listdir(self.output_folder+sigma):  
                if result=='.DS_Store':
                    continue
                df=pd.read_csv(self.output_folder+sigma+'/'+result)
                df_prediction[result] = np.array(df['prediction'])
                df_loss[result] = np.array(df['loss'])
            
            
            df_prediction.to_csv(prediction+sigma+'.csv',index=False)
            df_loss.to_csv(loss+sigma+'.csv',index=False)
                
    def clean(self):
        '''
        remove all the files in the output folder 
        '''
        folder = self.output_folder
        for the_file in os.listdir(folder):
            file_path = os.path.join(folder, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path): shutil.rmtree(file_path)
            except Exception as e:
                print(e)
        
    
if __name__ == '__main__':
    
    
# =============================================================================
#     file=fileName()
#     #file.output_folder = '/Users/mingmingyu/Downloads/output/'
#     #file.output_combine = '/Users/mingmingyu/Downloads/output_combine/'
#     file.combine_files()
#     file.clean()
#     
# =============================================================================    