import pickle as pkl
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_score,recall_score,f1_score
import numpy as np
from glob import glob
import os
from scipy.special import softmax
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import pandas as pd
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
    #print(f'El minimo es:{minim} el maximo es:{max} la media es:{med} la mediana es:{median}')
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
            #print(f'El umbral es {t}')
            precision.append(0)
            recall.append(0)
            Fmeasure.append(0)
    mAP=auc(recall,precision)
    Fmax = np.max(Fmeasure)
    max_ind = np.argmax(Fmeasure)
    return precision[max_ind], recall[max_ind], Fmax,mAP
#with open('probs.mat','rb') as f:
#    prediccion=pkl.load(f)

#with open('etiquetas.mat','rb') as f:
#    etiquetas=pkl.load(f)

def evaluar_total(etiquetas,prediccion,fold,adv,mode,image_demo=False):
    aux_etiquetas=[]
    aux_prediction=[]

    for i in range(len(etiquetas)):
        for j in range(len(etiquetas[i])):
            aux_etiquetas.append(etiquetas[i][j])
            aux_prediction.append(prediccion[i][j])
    etiquetas=np.array(aux_etiquetas)
    #print(etiquetas)


    #for i in range(len(prediccion)):
    #    for j in range(len(prediccion[i])):
            
    prediccion=np.array(aux_prediction)

    if mode!='demo':
        if fold==1:
            if adv:
                paths_N=glob("/media/SSD3/asusma/DELFOS2/DELFOS/Swin-Transformer/dataset_adv_fold1_original/val/Negative/"+'*.jpeg')
                paths_P=glob("/media/SSD3/asusma/DELFOS2/DELFOS/Swin-Transformer/dataset_adv_fold1_original/val/Positive/"+'*.jpeg')

                real_names_N=glob("/media/SSD3/asusma/DELFOS2/DELFOS/Swin-Transformer/dataset_fold1_original/val/Negative/"+'*.jpeg')
                real_names_P=glob("/media/SSD3/asusma/DELFOS2/DELFOS/Swin-Transformer/dataset_fold1_original/val/Positive/"+'*.jpeg')
            else:
                paths_N=glob("/media/SSD3/asusma/DELFOS2/DELFOS/Swin-Transformer/dataset_fold1_original/val/Negative/"+'*.jpeg')
                paths_P=glob("/media/SSD3/asusma/DELFOS2/DELFOS/Swin-Transformer/dataset_fold1_original/val/Positive/"+'*.jpeg')
        else:
            if adv:
                paths_N=glob("/media/SSD3/asusma/DELFOS2/DELFOS/Swin-Transformer/dataset_adv_fold2_original/val/Negative/"+'*.jpeg')
                paths_P=glob("/media/SSD3/asusma/DELFOS2/DELFOS/Swin-Transformer/dataset_adv_fold2_original/val/Positive/"+'*.jpeg')

                real_names_N=glob("/media/SSD3/asusma/DELFOS2/DELFOS/Swin-Transformer/dataset_fold2_original/val/Negative/"+'*.jpeg')
                real_names_P=glob("/media/SSD3/asusma/DELFOS2/DELFOS/Swin-Transformer/dataset_fold2_original/val/Positive/"+'*.jpeg')
            else:    
                paths_N=glob("/media/SSD3/asusma/DELFOS2/DELFOS/Swin-Transformer/dataset_fold2_original/val/Negative/"+'*.jpeg')
                paths_P=glob("/media/SSD3/asusma/DELFOS2/DELFOS/Swin-Transformer/dataset_fold2_original/val/Positive/"+'*.jpeg')     

        print(len(etiquetas))

        trimester1=[]
        trimester2=[]
        trimester3=[]

        trimester1_label=[]
        trimester2_label=[]
        trimester3_label=[]

        dict_resultados={}
        dict_trimestre1={}
        dict_trimestre2={}
        dict_trimestre3={}

        dict_resultados_gt={}
        dict_trimestre1_gt={}
        dict_trimestre2_gt={}
        dict_trimestre3_gt={}

        for i in range(len(paths_N)):
            l=paths_N[i].split('/')[-1].split('.')[1]
            patient=paths_N[i].split('/')[-1].split('.')[0]
            
            if l==1 or int(l)==1:
                #trimester1.append(prediccion[i])
                #trimester1_label.append(etiquetas[i])
                if patient not in dict_trimestre1.keys():
                    dict_trimestre1[patient]=[softmax(prediccion[i])]
                    dict_trimestre1_gt[patient]=etiquetas[i]
                else:
                    dict_trimestre1[patient].append(softmax(prediccion[i]))
                    dict_trimestre1_gt[patient]=etiquetas[i]

            if l==2 or int(l)==2:
                #trimester2.append(prediccion[i])
                #trimester2_label.append(etiquetas[i])
                if patient not in dict_trimestre2.keys():
                    dict_trimestre2[patient]=[softmax(prediccion[i])]
                    dict_trimestre2_gt[patient]=etiquetas[i]
                else:
                    dict_trimestre2[patient].append(softmax(prediccion[i]))
                    dict_trimestre1_gt[patient]=etiquetas[i]

            if l==3 or int(l)==3:
                #trimester3.append(prediccion[i])
                #trimester3_label.append(etiquetas[i])
                if patient not in dict_trimestre3.keys():
                    dict_trimestre3[patient]=[softmax(prediccion[i])]
                    dict_trimestre3_gt[patient]=etiquetas[i]
                else:
                    dict_trimestre3[patient].append(softmax(prediccion[i]))
                    dict_trimestre1_gt[patient]=etiquetas[i]
            if patient not in dict_resultados.keys():
                dict_resultados[patient]=[softmax(prediccion[i])]
                dict_resultados_gt[patient]=etiquetas[i]
            else:
                dict_resultados[patient].append(softmax(prediccion[i]))
                dict_trimestre1_gt[patient]=etiquetas[i]

        for i in range(len(paths_N),len(paths_N)+len(paths_P)):
            l=paths_P[i-len(paths_N)].split('/')[-1].split('.')[1]
            patient=paths_P[i-len(paths_N)].split('/')[-1].split('.')[0]
            
            if l==1 or int(l)==1:
                #trimester1.append(prediccion[i])
                #trimester1_label.append(etiquetas[i])
                if patient not in dict_trimestre1.keys():
                    dict_trimestre1[patient]=[softmax(prediccion[i])]
                    dict_trimestre1_gt[patient]=etiquetas[i]
                else:
                    dict_trimestre1[patient].append(softmax(prediccion[i]))
                    dict_trimestre1_gt[patient]=etiquetas[i]
            if l==2 or int(l)==2:
                #trimester2.append(prediccion[i])
                #trimester2_label.append(etiquetas[i])
                if patient not in dict_trimestre2.keys():
                    dict_trimestre2[patient]=[softmax(prediccion[i])]
                    dict_trimestre2_gt[patient]=etiquetas[i]
                else:
                    dict_trimestre2[patient].append(softmax(prediccion[i]))
                    dict_trimestre1_gt[patient]=etiquetas[i]
            if l==3 or int(l)==3:
                #trimester3.append(prediccion[i])
                #trimester3_label.append(etiquetas[i])
                if patient not in dict_trimestre3.keys():
                    dict_trimestre3[patient]=[softmax(prediccion[i])]
                    dict_trimestre3_gt[patient]=etiquetas[i]
                else:
                    dict_trimestre3[patient].append(softmax(prediccion[i]))
                    dict_trimestre1_gt[patient]=etiquetas[i]
            if patient not in dict_resultados.keys():
                dict_resultados[patient]=[softmax(prediccion[i])]
                dict_resultados_gt[patient]=etiquetas[i]
            else:
                dict_resultados[patient].append(softmax(prediccion[i]))
                dict_trimestre1_gt[patient]=etiquetas[i]


        keys=dict_resultados.keys()
        keys1=dict_trimestre1.keys()
        keys2=dict_trimestre2.keys()
        keys3=dict_trimestre3.keys()
        etiquetas=[]
        prediccion=[]
        probs=[]
        paths=[]

        for i in keys:
            resultados=np.array(dict_resultados[i])
            if resultados.shape[0]<2:
                logits=resultados[0]
            else:
                logits=np.max(resultados,axis=1)
            if logits[1]>0.5:
                prediccion.append(1)
                probs.append(logits[1])
            else:
                prediccion.append(0)
                probs.append(logits[1])
            #prediccion.append(np.argmax(logits))
            etiquetas.append(dict_resultados_gt[i])
            if i in keys1:
                trimester1_label.append(dict_resultados_gt[i])
                trimester1.append(np.argmax(logits))
                paths.append(i+'_1')
            if i in keys2:
                trimester2_label.append(dict_resultados_gt[i])
                trimester2.append(np.argmax(logits))
                paths.append(i+'_2')
            if i in keys3:
                trimester3_label.append(dict_resultados_gt[i])
                trimester3.append(np.argmax(logits))
                paths.append(i+'_3')

        curva=precision_recall_curve(etiquetas,probs)

        check='Swin_stablelike2_fold2'

        with open(f'Curvas/{check}.mat','wb') as f:
            pkl.dump(curva,f)

        mAP=average_precision_score(etiquetas,probs)

        with open(f'Curvas/{check}_mAP.mat','wb') as f:
            pkl.dump(mAP,f)

        #print(precision_recall_fscore_support(etiquetas,prediccion))
        #print('Dataset completo')
        #print(precision_score(etiquetas,prediccion))
        #print(recall_score(etiquetas,prediccion))
        #print(f1_score(etiquetas,prediccion))

        #print('Trimestre 1')
        #print(precision_score(trimester1_label,trimester1))
        #print(recall_score(trimester1_label,trimester1))
        #print(f1_score(trimester1_label,trimester1))

        #print('Trimestre 2')
        #print(precision_score(trimester2_label,trimester2))
        #print(recall_score(trimester2_label,trimester2))
        #print(f1_score(trimester2_label,trimester2))

        #print('Trimestre 3')
        #print(precision_score(trimester3_label,trimester3))
        #print(recall_score(trimester3_label,trimester3))
        #print(f1_score(trimester3_label,trimester3))

        #print('mAP')
        #print(mAP)
        #print(etiquetas)
        #print(prediccion)
        #print(probs)

        columnas=['gt','predictions','path']
        filas=[]
        for i in range(len(etiquetas)):
            filas.append([etiquetas[i],probs[i],paths[i]])
        df2=pd.DataFrame(filas,columns=columnas)
        if fold==1:
            df2.to_csv('Resultados/Test_fold1.csv', index=False)
        else:
            df2.to_csv('Resultados/Test_fold2.csv', index=False)

        pre,rec,f_score,mAP=pr_curve_manual(df2)
        print(f'La precision es: {pre}')
        print(f'La cobertura es: {rec}')
        print(f'La F score es: {f_score}')
        print(f'El mAP es: {mAP}')
    else:
            assert image_demo!=False, 'No se proporciono una imagen de prueba'
            print(f'La probabilidad de ser CHD es: {prediccion[0][1]}')