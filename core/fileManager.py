#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 14:14:58 2019

@author: mingmingyu
"""
import pandas as pd
import os, shutil
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
        prediction = self.output_combine+ 'prediction.xlsx'
        loss = self.output_combine + 'loss.xlsx'
        
        writer_pred = pd.ExcelWriter(prediction, engine='xlsxwriter')
        writer_loss = pd.ExcelWriter(loss, engine='xlsxwriter')
        
        for sigma in os.listdir(self.output_folder):
            for result in os.listdir(self.output_folder+sigma):
                df=pd.read_csv(self.output_folder+sigma+'/'+result,header=None) 
                df_prediction=pd.DataFrame(df['prediction'],columns=[result])
                df_prediction.to_excel(writer_pred, sheet_name=sigma)
                
                df_loss=pd.DataFrame(df['loss'],columns=[result])
                df_loss.to_excel(writer_loss, sheet_name=sigma)
                
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
    
    
    file=fileName()
    file.combine_files()
    file.clean()
    
    