# -*- coding: utf-8 -*-
#!/usr/bin/env python3 
"""

Choose a ratio for training/validation/testing here.

Script to:
    Take in all label files and output single list of labels
    Split data into a ratio for trianing/validation/testing
    
    
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


import numpy as np
import pandas as pd
import os
import csv
import sys



bif_total = int(sys.argv[1])
batch_num = int(sys.argv[2])




labels = []

with open("log_resultats.txt", "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            idx, label = line.split(",")
            labels.append({"sequence_ID": 500000+int(idx), "class_label": int(label)*4}) # the '4' is to be changed depending on which label is desired for the current use of the code 
        except ValueError:                                                               #int(label) will be = 0 if no  bifurcation, else will be = 1 -> multiply by a factor to change
            print("oups")                                                                #labelling. 


df_labels = pd.DataFrame(labels)


df_labels.to_csv("output_labels/out_labels.csv", header=True, index=False)





