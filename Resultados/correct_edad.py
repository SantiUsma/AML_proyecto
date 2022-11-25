import pandas as pd
import numpy as np

name='original_fold1'

fold1 = pd.read_csv(name+'.csv')

nombre=fold1['path']
gt=fold1['gt']
probs=fold1['predictions']

gt_1=[]
gt_2=[]
gt_3=[]

probs_1=[]
probs_2=[]
probs_3=[]

filas1=[]
filas2=[]
filas3=[]

for i in range(len(gt)):
    trimester=int(nombre[i].split('_')[-1])
    if trimester==1:
        gt_1.append(gt[i])
        probs_1.append(probs[i])
        filas1.append([gt[i],probs[i],nombre[i]])
    if trimester==2:
        gt_2.append(gt[i])
        probs_2.append(probs[i])
        filas2.append([gt[i],probs[i],nombre[i]])
    if trimester==3:
        gt_3.append(gt[i])
        probs_3.append(probs[i])
        filas3.append([gt[i],probs[i],nombre[i]])

columnas=['gt','predictions','path']
df1=pd.DataFrame(filas1,columns=columnas)
df2=pd.DataFrame(filas2,columns=columnas)
df3=pd.DataFrame(filas3,columns=columnas)

df1.to_csv(name+'_1.csv', index=False)
df2.to_csv(name+'_2.csv', index=False)
df3.to_csv(name+'_3.csv', index=False)