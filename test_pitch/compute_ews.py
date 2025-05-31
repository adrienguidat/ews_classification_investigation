#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compute EWS from residuals for the final 500 points of the 1500-point simulations

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



# import python libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import ewstools



#---------------
# import data
#-------------

# import residual data from 1500-point simulations
columns = ['tsid','Time','Residuals','Variance','Lag-1 AC']
df_resids_forced = pd.read_csv('test_pitch/data/ews/df_ews_forced.csv', 
                        usecols=columns,
                        ).dropna()

df_resids_null = pd.read_csv('test_pitch/data/ews/df_ews_null.csv',
                        usecols=columns,
                        ).dropna()


#-----------
# compute ews
#-----------

# EWS parameters
dt2 = 1 # spacing between time-series for EWS computation
rw = 0.25 # rolling window
span = 0.2 # bandwidth
lags = [1] # autocorrelation lag times
ews = ['var','ac']



# compute ews over final 500 points of residuals - forced trajectories
df_resids = df_resids_forced

list_df_ews = []
tsid_vals = df_resids['tsid'].unique()
for tsid in tsid_vals:
    
    # resids for this tsid - take final 500 points
    df_resids500 = df_resids[df_resids['tsid']==tsid].iloc[-500:]
    series_resids500 = df_resids500.set_index('Time')['Residuals']
    
    ews_dic = ewstools.core.ews_compute(series_resids500, 
                      roll_window = rw,
                      smooth='None',
                      span = span,
                      lag_times = lags, 
                      ews = ews,
                      )
    
    # The DataFrame of EWS
    df_ews_temp = ews_dic['EWS metrics']
    
    # Include a column in the DataFrames for realisation number and variable
    df_ews_temp['tsid'] = tsid
    
    # Add DataFrames to list
    list_df_ews.append(df_ews_temp)
    print('Complete for tsid = {}'.format(tsid))

# Concatenate EWS DataFrames. Index [Realisation number, Variable, Time]
df_ews = pd.concat(list_df_ews).reset_index().set_index(['tsid','Time'])

# Change col name to Residuals
df_ews.rename(columns={'State variable':'Residuals'},
              inplace=True)

# Export EWS data
df_ews.to_csv('test_pitch/data/ews/df_ews_forced.csv')





# compute ews over final 500 points of residuals - null trajectories
df_resids = df_resids_null

list_df_ews = []
tsid_vals = df_resids['tsid'].unique()
for tsid in tsid_vals:
    
    # resids for this tsid - take final 500 points
    df_resids500 = df_resids[df_resids['tsid']==tsid].iloc[-500:]
    series_resids500 = df_resids500.set_index('Time')['Residuals']
    
    ews_dic = ewstools.core.ews_compute(series_resids500, 
                      roll_window = rw,
                      smooth='None',
                      span = span,
                      lag_times = lags, 
                      ews = ews,
                      )
    
    # The DataFrame of EWS
    df_ews_temp = ews_dic['EWS metrics']
    
    # Include a column in the DataFrames for realisation number and variable
    df_ews_temp['tsid'] = tsid
    
    # Add DataFrames to list
    list_df_ews.append(df_ews_temp)
    print('Complete for tsid = {}'.format(tsid))

# Concatenate EWS DataFrames. Index [Realisation number, Variable, Time]
df_ews = pd.concat(list_df_ews).reset_index().set_index(['tsid','Time'])


# Change col name to Residuals
df_ews.rename(columns={'State variable':'Residuals'},
              inplace=True)



# Export EWS data
df_ews.to_csv('test_pitch/data/ews/df_ews_null.csv')




