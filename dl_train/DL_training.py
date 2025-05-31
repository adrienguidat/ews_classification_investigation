#!/usr/bin/env python3


"""

Script to train the network
    
# ------------------------------------------------------------------------
# Original code from:
# Thomas M. Bury. Deep learning for early warning signals of tipping points. 
# PNAS, 118(39),2021
# GitHub repository: https://github.com/ThomasMBury/deep-early-warnings-pnas
#
# Adapted and modified by Adrien Guidat for the master thesis "Investigation on EWS detection and classification through Deep Neural Networks" (2025).
#
# ------------------------------------------------------------------------
    

"""

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

# model_type
# 1: both left and right sides of time series are padded
# 2: only left side of time series is padded


model_type = 1 
kk = 1 
print('Train a classifier of type {} with index {}'.format(model_type, kk))



# Set size of training library and length of time series
(lib_size, ts_len) = (510000, 500) # to decide till where go in the IDs of the resids, here we go till 510000




if model_type==1:
    pad_left = 225 if ts_len==500 else 725
    pad_right = 225 if ts_len==500 else 725

if model_type==2:
    pad_left = 450 if ts_len==500 else 1450
    pad_right = 0





# get zipfile of time series
print('Load in zip file containing training data')
data_dir = 'mon_dossier_24k/output_resids/output_resids/'  


sequences = list()


df_targets = pandas.read_csv('mon_dossier_24k/labels.csv',
                        index_col='sequence_ID')

df_targets['class_label'] = df_targets['class_label'].fillna(0) #change nans by 0 

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
            if tsid in df_targets.index and df_targets.loc[tsid, 'class_label'] == 15 and le_counter_F<bif_numb:   #changed to 15,1 and 13 to not take from Bury's original set
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
            if tsid in df_targets.index and df_targets.loc[tsid, 'class_label'] == 4 and le_counter_P<bif_numb:  #here we are in our trainset
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
df_groups.to_csv("mon_dossier_24k/groups.csv", index=False)


sequences = np.array(sequences)
print("extracted!")
# Get target labels for each data sample



df_groups_v2 = pandas.read_csv('mon_dossier_24k/groups.csv')
print(df_groups_v2.columns)


# train/validation/test split denotations
df_groups = pandas.read_csv('mon_dossier_24k/groups.csv',
                            index_col='sequence_ID')

#Padding input sequences
print("padding...")
for i, tsid in enumerate(tsid_reals):
# for i in range(lib_size):
    pad_length = int(pad_left*random.uniform(0, 1))
    for j in range(0,pad_length):
        sequences[i, j] = 0

    pad_length = int(pad_right*random.uniform(0, 1))
    for j in range(ts_len - pad_length, ts_len):
        sequences[i, j] = 0
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

print("costa")
print(np.any(np.isnan(train)))
print(np.any(np.isinf(train)))
print(np.any(np.isnan(train_target)))
print(np.any(np.isinf(train_target)))
print("John")
print(np.unique(train_target))
print('Johnny')

# keeps track of training metrics
f1_name = 'training_results_{}_{}.txt'.format(kk, model_type)
f2_name = 'training_results_{}_{}.csv'.format(kk, model_type)

f_results= open(f1_name, "w")
f_results2 = open(f2_name, "w")

start_time = time.time()


# hyperparameter settings
CNN_layers = 1
LSTM_layers = 1
pool_size_param = 2 
learning_rate_param = 0.0001 #0.0005    
batch_param =5000 #250000
dropout_percent = 0.10 #0.1
filters_param = 128 #50  
mem_cells = 128 #128  
mem_cells2 = 128 #64
kernel_size_param = 6 #12 
epoch_param = 1000 #1500 
initializer_param = 'lecun_normal' #'glorot_uniform' #'lecun_normal'


model = Sequential()

# add layers

model.add(Conv1D(filters=filters_param, kernel_size=kernel_size_param, activation='swish', padding='same',input_shape=(ts_len, 1),kernel_initializer = initializer_param))
model.add(BatchNormalization())

model.add(Dropout(dropout_percent))
model.add(MaxPooling1D(pool_size=pool_size_param))

model.add(Conv1D(filters=64, kernel_size=12, activation='swish', padding='same', kernel_initializer=initializer_param))
model.add(BatchNormalization())
model.add(Dropout(dropout_percent))
model.add(MaxPooling1D(pool_size=2))



model.add(LSTM(mem_cells, return_sequences=True, kernel_initializer = initializer_param)) #False parce que comme on retire la prochaine LSTM, qui etait pas definie donc a False par defaut (pas besoin de preciser), bien elle sortait pas bon format en sortie, donc il faut remonter mettre el Fase au dessus!
model.add(Dropout(dropout_percent))

model.add(LSTM(mem_cells2,kernel_initializer = initializer_param)) #on retire un LSTM car tres couteux et on augmente memcelle de l'autre pour contrebalancer
model.add(Dropout(dropout_percent))
model.add(Dense(5, activation='softmax',kernel_initializer = initializer_param))

# name for output pickle file containing model info
model_name = 'best_model_24k_FULL_DISTINCT_{}_{}_length{}.keras'.format(kk,model_type,ts_len) #changed .pkl en .keras
print("estamos aqui")
print('looool')
# Set up optimiser
adam = Adam(learning_rate=learning_rate_param)
chk = ModelCheckpoint(model_name, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy', 'sparse_categorical_accuracy'])



# Train model
history = model.fit(train, train_target, epochs=epoch_param, batch_size=batch_param, callbacks=chk, validation_data=(validation,validation_target))


model = load_model(model_name)

end_time = time.time()
print("Temps d'execution : ")
print((end_time-start_time)/3600)