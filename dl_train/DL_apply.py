#!/usr/bin/env python3

'''
Code to generate ensemble predictions from the DL classifiers
on a give time series of residuals

# ------------------------------------------------------------------------
# Original code from:
# Thomas M. Bury. Deep learning for early warning signals of tipping points. 
# PNAS, 118(39),2021
# GitHub repository: https://github.com/ThomasMBury/deep-early-warnings-pnas
#
# Adapted and modified by Adrien Guidat for the master thesis "Investigation on EWS detection and classification through Deep Neural Networks" (2025).
#
# ------------------------------------------------------------------------

'''


import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import gc
import numpy as np
import pandas as pd
import random
import sys
import itertools

import tensorflow as tf
from tensorflow.keras.models import load_model
from datetime import datetime

random.seed(1)




for i in range(1,11):


  # Filepath to residual time series to make predictions on
  filepath = 'test_pitch/data/resids/hoop_picth_500_resids_{}.csv'.format(i) 

  # Filepath to export ensemble DL predictions to
  filepath_out = 'test_pitch/ml_preds/diy_preds_forced_V5_{}_.csv'.format(i) 

  # Type of classifier to use (1500 or 500)
  ts_len=500

  '''
  The following two parameters control how many sample points along the
  timeseries are used, and the length between them.  For instance, for an input
  time series equal to or less then length 1500, mult_factor=10 and
  pad_samples=150 means that it will do the unveiling in steps of 10 datapoints,
  at 150 evenly spaced points across the entire time series (150 x 10 = 1500).
  Needs to be adjusted according to whether you are using the trained 500 length
  or 1500 length classifier.
  '''

  # Steps of datapoints in between each DL prediction
  mult_factor = 10

  # Total number of DL predictions to make
  # Use 150 for length 1500 time series. Use 50 for length 500 time series.
  pad_samples = 50



  # Load residual time series data
  df = pd.read_csv(filepath).dropna()
  resids = df['Residuals'].values.reshape(1,-1,1)
  # Length of inupt time series
  seq_len = len(df)
  


  def get_dl_predictions(resids, model_type, kk):

      '''
      Generate DL prediction time series on resids
      from DL classifier with type 'model_type' and index kk.
      '''

      # Setup file to store DL prediction
      predictions_file_name = 'test_pitch/ml_preds/y_pred_forced_V5_{}_{}.csv'.format(kk,model_type)
      f1 = open(predictions_file_name,'w')

      # Load in specific DL classifier
      model_name = 'best_model_24k_FULL_DISTINCT_{}_{}_length{}.keras'.format(kk,model_type,ts_len) 
      model = load_model(model_name,compile=False)

      # Loop through each possible length of padding
      # Start with revelaing the DL algorith only the earliest points
      for pad_count in range(pad_samples-1, -1, -1):

          temp_ts = np.zeros((1,ts_len,1))

          ts_gap = ts_len-seq_len
          pad_length = mult_factor*pad_count

          if pad_length + ts_gap > ts_len:
              zero_range = ts_len
          else:
              zero_range = pad_length + ts_gap

          if zero_range == ts_len:
              # Set all ML predictions to zero
              y_pred = np.zeros(5).reshape(1,5) 
          else:
              for j in range(0, zero_range):
                  temp_ts[0,j] = 0
              for j in range(zero_range, ts_len):
                  temp_ts[0,j] = resids[0,j-zero_range]

              # normalizing inputs: take averages, since the models were also trained on averaged data.
              values_avg = 0.0
              count_avg = 0
              for j in range (0,ts_len):
                  if temp_ts[0,j] != 0:
                      values_avg = values_avg + abs(temp_ts[0,j])
                      count_avg = count_avg + 1
              if count_avg != 0:
                  values_avg = values_avg/count_avg
              for j in range (0,ts_len):
                  if temp_ts[0,j] != 0:
                      temp_ts[0,j] = temp_ts[0,j]/values_avg

              # Compute DL prediction
              y_pred = model.predict(temp_ts)



          # Write predictions to file
          np.savetxt(f1, y_pred, delimiter=',')
          print('Predictions computed for padding={}'.format(pad_count*mult_factor))

      # Delete model and do garbage collection to free up RAM
      tf.keras.backend.clear_session()
      if zero_range != ts_len:
          del model
      gc.collect()
      f1.close()

      return




  print('Compute DL predictions for model_type=1, kk=1')

  get_dl_predictions(resids, 1, 1)




  # Compute average prediction 
  list_df_preds = []
  for model_type in [1]:
      for kk in [1]:
          filename = 'test_pitch/ml_preds/y_pred_forced_V5_{}_{}.csv'.format(kk,model_type)
          df_preds = pd.read_csv(filename,header=None)
          df_preds['time_index'] = df_preds.index
          df_preds['model_type'] = model_type
          df_preds['kk'] = kk
          list_df_preds.append(df_preds)


  # Concatenate
  df_preds_all = pd.concat(list_df_preds).reset_index(drop=True)

  # Compute mean over all predictions
  df_preds_mean = df_preds_all.groupby('time_index').mean()


  # Add bifurcation probability as 1-null probability
  df_preds_mean[5] = 1- df_preds_mean[3]                    

  # Export predictions
  df_preds_mean[[0,1,2,3,4,5]].to_csv(filepath_out,index=False,header=False) 






