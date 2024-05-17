# -*- coding: utf-8 -*-
"""
Created on Fri May 17 12:08:15 2024

@author: colin
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import glob

os.chdir('C:/Users/colin/Documents/L3 ENS/Stage/Précision')

for file in glob.glob('*.csv'):
    
    df = pd.read_csv(file,sep=',',skiprows=2,header=None)
    
    if df.shape[1]!=45:
        print(file,': problème')
        
    else:
        #MISE EN FORME DU DATAFRAME

        #garder que les colonnes des marqueurs et pas du rigid body
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

        time = pd.to_numeric(df['Time'])

        #tracer les distances entre toutes les combinaisons de marqueurs
        k=0
        plt.figure(figsize=(18,15))
        for i in range(1,7):
            for j in range(1,i):
                k+=1
                plt.subplot(5,3,k)
                Mi = 'Marker'+str(i)
                Mj = 'Marker'+str(j)
                d = np.sqrt((pd.to_numeric(df[Mi+'X'])-pd.to_numeric(df[Mj+'X']))**2+(pd.to_numeric(df[Mi+'Y'])-pd.to_numeric(df[Mj+'Y']))**2+(pd.to_numeric(df[Mi+'Z'])-pd.to_numeric(df[Mj+'Z']))**2)

                plt.plot(time,d)
                plt.xlabel('t')
                plt.ylabel('d'+str(i)+str(j))
                plt.title(str(file)[:-4]+' d'+str(i)+str(j)+'(t)')
                plt.gcf().subplots_adjust(wspace=0.3,hspace = 0.6)
        plt.savefig(str(file)[:-4]+".png")
        plt.show()
        
