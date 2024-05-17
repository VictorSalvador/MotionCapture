# -*- coding: utf-8 -*-
"""
Created on Fri May 17 16:28:42 2024

@author: vicsalcas
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import glob

os.chdir('C:/Users/vicsalcas/Desktop/VIC/PhD/MoCap/Calibration/Data')

# Set height and markers
Height = 'y2'
Marker1 = 4
Marker2 = 6

# Create subplot for the desired height
fig, axs = plt.subplots(3,3,figsize=(10,10),sharey=True)
fig.delaxes(axs[0][0])
fig.delaxes(axs[0][2])
fig.delaxes(axs[2][0])
fig.delaxes(axs[2][2])

for file in glob.glob('*.csv'):
    
    df = pd.read_csv(file,sep=',',skiprows=2,header=None)
    if file.find(Height) < 0: continue
    plt.suptitle('Distance between markers ' + str(Marker1) + ' and ' + str(Marker2) + '. Height is ' + Height)
    
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
        
        Mi = 'Marker'+str(Marker1)
        Mj = 'Marker'+str(Marker2)
        d = np.sqrt((pd.to_numeric(df[Mi+'X'])-pd.to_numeric(df[Mj+'X']))**2+(pd.to_numeric(df[Mi+'Y'])-pd.to_numeric(df[Mj+'Y']))**2+(pd.to_numeric(df[Mi+'Z'])-pd.to_numeric(df[Mj+'Z']))**2)
        axs[int(file[1]),int(file[5])].plot(time,d)
        axs[int(file[1]),int(file[5])].set_title('(x,z) = (' + file[1] + ',' + file[5] + ')')

plt.show()