# -*- coding: utf-8 -*-
"""
Created on Wed May 15 15:10:31 2024

@author: colin
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import glob

os.chdir('C:/Users/colin/Documents/L3 ENS/Stage/Précision')

for file in glob.glob('*.csv'):
    print(file)

df = pd.read_csv(r"C:\Users\colin\Documents\L3 ENS\Stage\Précision\x0y0z0.csv", sep=',',skiprows=2,header=None)

#MISE EN FORME DU DATAFRAME

#garder ques les colonnes des marqueurs et pas du rigid body
for j in range(df.shape[1]):
    if df.iat[0,j]=='Marker':
        break
    
for i in range(2,j):
    df = df.drop(i,axis=1)
 
#supprimer les lignes dont on a pas besoin
df = df.drop(0,axis=1)
df = df.drop([0,2,3],axis=0)

#renommer les colonnes  
df = df.rename(columns={df.columns[0]:'Time'})
for i in range(1,df.shape[1]):
    df = df.rename(columns={df.columns[i]:'Marker'+str(int((i-1)/3)+1)+df.iat[1,i]})
    
df = df.drop([df.index[0],df.index[1]],axis=0) #supprimer les lignes dont on a plus besoin

df = df.reset_index(drop = True) #réindexer les lignes à partir de 0

#EXPLOITATION DES DONNEES

