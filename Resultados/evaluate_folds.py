from sklearn.metrics import precision_recall_curve
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from sklearn.metrics import auc

def pr_curve_manual(dataframe):
    thresholds = np.arange(0, 1, 0.001)
    precision = []
    recall = []
    Fmeasure = []
    preds=dataframe['predictions']
    minim=np.min(preds)
    max=np.max(preds)
    med=np.mean(preds)
    median=np.median(preds)
    print(f'El minimo es:{minim} el maximo es:{max} la media es:{med} la mediana es:{median}')
    for t in thresholds: 
        TP = dataframe[dataframe['predictions']>t]['gt'].sum()
        FP = len(dataframe[dataframe['predictions']>t]['gt']) - TP
        FN = dataframe[dataframe['predictions']<t]['gt'].sum()
        TN = len(dataframe[dataframe['predictions']<t]['gt']) - FN
        pre = TP/(TP+FP)
        re = TP/(TP+FN)
        Fm = (2*pre*re)/(pre+re)
        if pre != 0 and re != 0:
            precision.append(pre)
            recall.append(re)
            Fmeasure.append(Fm)
        elif pre == 0 and re == 0:
            print(f'El umbral es {t}')
            precision.append(0)
            recall.append(0)
            Fmeasure.append(0)
    mAP=auc(recall,precision)
    return precision, recall, Fmeasure,mAP

print('corro')
#Fold 1
fold1 = pd.read_csv('original_fold1_1.csv')
#fold1 = pd.read_csv('Swin_edad_trans_fold1.csv')
#fold1 = pd.read_csv('Swin_stable_like_fold1.csv')
#pre_1,re_1,th_1 = precision_recall_curve(fold1['gt'].values,fold1['predictions'].values)

#Fmeasure_1 = (2*pre_1*re_1)/(pre_1+re_1) 
pre_1, re_1, Fmeasure_1 = pr_curve_manual(fold1)
Fmax_1 = np.max(Fmeasure_1)
max_ind_1 = np.argmax(Fmeasure_1)
auc_1 = auc(re_1,pre_1)

plt.figure()
plt.plot(re_1[:-1],pre_1[:-1])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.xlim([0,1])
plt.ylim([0,1])
plt.title(f'Fold1 - AUC {round(auc_1,2)} - Fmax {round(Fmax_1,2)} - Pre {round(pre_1[max_ind_1],2)} - Re {round(re_1[max_ind_1],2)}')
plt.savefig('original_fold1_1.jpeg')

#Fold 2
fold2 = pd.read_csv('original_fold2_1.csv')
#fold2 = pd.read_csv('Swin_edad_trans_fold2.csv')
#fold2 = pd.read_csv('Swin_stable_like_fold2.csv')
#pre_2,re_2,th_2 = precision_recall_curve(fold2['gt'].values,fold2['predictions'].values)

#Fmeasure_2 = (2*pre_2[:-2]*re_2[:-2])/(pre_2[:-2]+re_2[:-2]) 
pre_2, re_2, Fmeasure_2 = pr_curve_manual(fold2)
Fmax_2 = np.max(Fmeasure_2)
max_ind_2 = np.argmax(Fmeasure_2)
auc_2 = auc(re_2,pre_2)

plt.figure()
plt.plot(re_2,pre_2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.xlim([0,1])
plt.ylim([0,1])
plt.title(f'Fold2 - AUC {round(auc_2,2)} - Fmax {round(Fmax_2,2)} - Pre {round(pre_2[max_ind_2],2)} - Re {round(re_2[max_ind_2],2)} ')
plt.savefig('original_fold2_1.jpeg')

#Both folds
#all_folds_gt =list(fold1['gt'].values)+list(fold2['gt'].values)
#all_folds_pred =list(fold1['predictions'].values)+list(fold2['predictions'].values)
#pre,re,th = precision_recall_curve(all_folds_gt,all_folds_pred)

#Fmeasure = (2*pre[:-2]*re[:-2])/(pre[:-2]+re[:-2]) 
all_folds = pd.concat([fold1,fold2])
#pre, re, Fmeasure = pr_curve_manual(all_folds)
pre,re,Fmeasure=np.mean([pre_1[max_ind_1],pre_2[max_ind_2]]),np.mean([re_1[max_ind_1],re_2[max_ind_2]]),np.mean([Fmax_1,Fmax_2])

#Fmax = np.max(Fmeasure)
#max_ind = np.argmax(Fmeasure)
#auc_overall = auc(re,pre)

plt.figure()
plt.plot(re,pre)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.xlim([0,1])
plt.ylim([0,1])
plt.title(f'Swin baseline Precision-Recall curve')
plt.savefig('Overall_original_1.jpeg')

print(f'Precision:{pre}')
print(f'Cobertura:{re}')

print('Auc')
print(auc_1)
print(auc_2)
#print(auc_overall)
print('F score')
print(Fmax_1)
print(Fmax_2)
#print(Fmax)