#!/usr/bin/env python3


# ------------------------------------------------------------------------
# Script to evalute the network on the validation set.
#
# Original code from:
# Thomas M. Bury. Deep learning for early warning signals of tipping points. 
# PNAS, 118(39),2021
# GitHub repository: https://github.com/ThomasMBury/deep-early-warnings-pnas
#
# Adapted and modified by Adrien Guidat for the master thesis "Investigation on EWS detection and classification through Deep Neural Networks" (2025).
#
# ------------------------------------------------------------------------

import os
import zipfile
import time




start_time = time.time()

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import pandas
import numpy as np
import random
import sys



import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.activations import swish

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime

random.seed(1)


model_type = 1 
kk = 1 
print('Train a classifier of type {} with index {}'.format(model_type, kk))


# Set size of training library and length of time series
(lib_size, ts_len) = (510000, 500) #defines the ID max we will extract, here we go till 510000




if model_type==1:
    pad_left = 225 if ts_len==500 else 725
    pad_right = 225 if ts_len==500 else 725

if model_type==2:
    pad_left = 450 if ts_len==500 else 1450
    pad_right = 0



print('Load in file containing training data')
data_dir = 'mon_dossier_24k/output_resids/output_resids/'  

sequences = list()


df_targets = pandas.read_csv('mon_dossier_24k/labels.csv',
                        index_col='sequence_ID')

df_targets['class_label'] = df_targets['class_label'].fillna(0) 

le_counter_H=0
le_counter_N=0
le_counter_F=0
le_counter_P=0
le_counter_T=0



print('Extract time series from zip file')

tsid_reals = list()
group_rows = []
y=0

tsid=500000
bif_numb=97
born_low=bif_numb - 8*bif_numb/40 #2
born_mid=bif_numb - 4*bif_numb/40 #1

while tsid<lib_size+1:
    filepath = os.path.join(data_dir, f'resids{tsid}.csv')
    if os.path.exists(filepath):
        if tsid<500000:
            if tsid in df_targets.index and df_targets.loc[tsid, 'class_label'] == 15 and le_counter_F<bif_numb: 
                df = pandas.read_csv(filepath)
                values = df[['Residuals']].values 
                sequences.append(values)
                tsid_reals.append(tsid)
                if le_counter_F < born_low: y=1
                elif le_counter_F < born_mid: y=2
                else: y=3
                group_rows.append([tsid,y])
                le_counter_F+=1
                
            if tsid in df_targets.index and df_targets.loc[tsid, 'class_label'] == 14 and le_counter_H<bif_numb: 
                df = pandas.read_csv(filepath)
                values = df[['Residuals']].values
                sequences.append(values)
                tsid_reals.append(tsid)
                if le_counter_H < born_low: y=1
                elif le_counter_H < born_mid: y=2
                else: y=3
                group_rows.append([tsid,y])
                le_counter_H+=1
                
            if tsid in df_targets.index and df_targets.loc[tsid, 'class_label'] == 13 and le_counter_N<bif_numb: 
                df = pandas.read_csv(filepath)
                values = df[['Residuals']].values
                sequences.append(values)
                tsid_reals.append(tsid)
                if le_counter_N < born_low: y=1
                elif le_counter_N < born_mid: y=2
                else: y=3
                group_rows.append([tsid,y])
                le_counter_N+=1
        
            if ( (le_counter_H>=bif_numb) and (le_counter_N>=bif_numb) ): tsid=500000
            
        if tsid>=500000:                                                                                        
            if tsid in df_targets.index and df_targets.loc[tsid, 'class_label'] == 4 and le_counter_P<bif_numb: 
                df = pandas.read_csv(filepath)
                values = df[['Residuals']].values
                sequences.append(values)
                tsid_reals.append(tsid)
                if le_counter_P < born_low: y=1
                elif le_counter_P < born_mid: y=2
                else: y=3
                group_rows.append([tsid,y])
                le_counter_P+=1
                
            if tsid in df_targets.index and df_targets.loc[tsid, 'class_label'] == 3 and le_counter_T<bif_numb: 
                df = pandas.read_csv(filepath)
                values = df[['Residuals']].values
                sequences.append(values)
                tsid_reals.append(tsid)
                if le_counter_T < born_low: y=1
                elif le_counter_T < born_mid: y=2
                else: y=3
                group_rows.append([tsid,y])
                le_counter_T+=1
                
                
            if tsid in df_targets.index and df_targets.loc[tsid, 'class_label'] == 1 and le_counter_F<bif_numb: 
                df = pandas.read_csv(filepath)
                values = df[['Residuals']].values
                sequences.append(values)
                tsid_reals.append(tsid)
                if le_counter_F < born_low: y=1
                elif le_counter_F < born_mid: y=2
                else: y=3
                group_rows.append([tsid,y])
                le_counter_F+=1
                
            if tsid in df_targets.index and df_targets.loc[tsid, 'class_label'] == 2 and le_counter_H<bif_numb: 
                df = pandas.read_csv(filepath)
                values = df[['Residuals']].values
                sequences.append(values)
                tsid_reals.append(tsid)
                if le_counter_H < born_low: y=1
                elif le_counter_H < born_mid: y=2
                else: y=3
                group_rows.append([tsid,y])
                le_counter_H+=1
                
            if tsid in df_targets.index and df_targets.loc[tsid, 'class_label'] == 0 and le_counter_N<bif_numb: 
                df = pandas.read_csv(filepath)
                values = df[['Residuals']].values
                sequences.append(values)
                tsid_reals.append(tsid)
                if le_counter_N < born_low: y=1
                elif le_counter_N < born_mid: y=2
                else: y=3
                group_rows.append([tsid,y])
                le_counter_N+=1
            
            if ( (le_counter_T>=bif_numb) and (le_counter_P>=bif_numb) and (le_counter_F>=bif_numb) and (le_counter_N>=bif_numb) and (le_counter_H>=bif_numb) ): tsid=lib_size
            
    tsid+=1    
 


df_groups = pandas.DataFrame(group_rows, columns=["sequence_ID","dataset_ID"])
df_groups.to_csv("mon_dossier_24k/groups_confusion.csv", index=False)


sequences = np.array(sequences)
print("extracted!")




df_groups_v2 = pandas.read_csv('mon_dossier_24k/groups_confusion.csv')
print(df_groups_v2.columns)


# train/validation/test split denotations
df_groups = pandas.read_csv('mon_dossier_24k/groups_confusion.csv',
                            index_col='sequence_ID')

#Padding input sequences
print("padding...")
for i, tsid in enumerate(tsid_reals):
# for i in range(lib_size):
    pad_length = int(pad_left*random.uniform(0, 1))
    for j in range(0,pad_length):
        sequences[i,j] = 0

    pad_length = int(pad_right*random.uniform(0, 1))
    for j in range(ts_len - pad_length, ts_len):
        sequences[i,j] = 0
print("padded")
print("normalizing")
# normalizing input time series by the average.
for i, tsid in enumerate(tsid_reals):
# for i in range(lib_size):
    values_avg = 0.0
    count_avg = 0
    for j in range (0,ts_len):
        if sequences[i,j] != 0:
          values_avg = values_avg + abs(sequences[i,j])
          count_avg = count_avg + 1
    if count_avg != 0:
        values_avg = values_avg/count_avg
        for j in range (0,ts_len):
            if sequences[i,j] != 0:
                sequences[i,j] = sequences[i,j]/values_avg 
                

print("normalized!")
final_seq = sequences

# apply train/test/validation labels
train = [final_seq[i] for i, tsid in enumerate(tsid_reals) if df_groups['dataset_ID'].loc[tsid]==1]
validation = [final_seq[i] for i, tsid in enumerate(tsid_reals) if df_groups['dataset_ID'].loc[tsid]==2]
test = [final_seq[i] for i, tsid in enumerate(tsid_reals) if df_groups['dataset_ID'].loc[tsid]==3]

df_targets['class_label'] = df_targets['class_label'].fillna(0) 

train_target = [df_targets['class_label'].loc[tsid] for i, tsid in enumerate(tsid_reals) if df_groups['dataset_ID'].loc[tsid]==1]
validation_target = [df_targets['class_label'].loc[tsid] for i, tsid in enumerate(tsid_reals) if df_groups['dataset_ID'].loc[tsid]==2]
test_target = [df_targets['class_label'].loc[tsid] for i, tsid in enumerate(tsid_reals) if df_groups['dataset_ID'].loc[tsid]==3]
print("splitted!")





train = np.array(train)
validation = np.array(validation)
test = np.array(test)

train_target = np.array(train_target)
validation_target = np.array(validation_target)
test_target = np.array(test_target)

train_target = train_target #- 1
validation_target = validation_target #- 1
test_target = test_target #- 1




model_name = 'best_model_24k_FULL_DISTINCT_{}_{}_length{}.keras'.format(kk,model_type,ts_len) 
model = load_model(model_name,compile=False)




y_pred_proba = model.predict(test)  
y_pred = np.argmax(y_pred_proba, axis=1)  


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(test_target, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()


from sklearn.metrics import classification_report


print(classification_report(test_target, y_pred))






