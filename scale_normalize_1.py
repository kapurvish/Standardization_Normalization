# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 22:22:55 2019

@author: Vishal Kapur
"""

import sklearn
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import math
#from sklearn.linear_model import LogisticRegression
pd.set_option('display.max_columns',32)

df= pd.read_csv('Absenteeism_at_work.csv',delimiter=';')
df1= pd.read_csv('absenteeism_processed.csv',delimiter=';')

print(df.head())

print(df.columns)
df =df[['Transportation expense','Distance from Residence to Work','Service time','Age',
         'Work load Average/day ','Weight', 'Height', 'Body mass index','Absenteeism time in hours']]
df.rename(columns={'Transportation expense' : 'Transportation',
                   'Distance from Residence to Work':'Distance',
                   'Service time':'Service',
                   'Work load Average/day ':'Workload',
                   'Body mass index':'BMI',
                   'Absenteeism time in hours':'AbsentHours'                   
                   },inplace=True)
df =df.astype(np.float64)

print(df.head())
print(df.describe())
df.boxplot(figsize=(12,8))
plt.show()
#Before scaling the boxplot 


std_scaler = StandardScaler(copy=True,with_mean=True,with_std=True)
scaled_array =std_scaler.fit_transform(df)
scaled_df= pd.DataFrame(scaled_array,columns=df.columns)

scaled_df.boxplot(figsize=(12,8))
#After scaling the boxplot 
plt.show()



