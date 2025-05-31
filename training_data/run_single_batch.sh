#!/bin/bash

#Script to generate a single batch of data

# ------------------------------------------------------------------------
# Original code from:
# Thomas M. Bury. Deep learning for early warning signals of tipping points. 
# PNAS, 118(39),2021
# GitHub repository: https://github.com/ThomasMBury/deep-early-warnings-pnas
#
# Adapted and modified by Adrien Guidat for the master thesis "Investigation on EWS detection and classification through Deep Neural Networks" (2025).
#
# ------------------------------------------------------------------------


# Get command line parameters
batch_num=555 # Batch number (we send each batch to a different CPUs to run in parallel)
ts_len=500 # Length of time series to generate for training data

# Other parameters
bif_max=1000 # Number of each type of bifurcation requested (we do 1000 per batch)


# Make batch-specific directory for storing output
mkdir -p output_diy/ts_$ts_len/batch$batch_num
# Move to the batch-specific directory
cd output_diy/ts_$ts_len/batch$batch_num

# Set up timer
start=`date +%s`

touch log_resultats.txt

echo Job released


# While loop
for i in $(seq 0 1000) #bif_max
do
echo "Valeur de i: $i"

echo Run gen_model_diy.py
python3 ../../../gen_model.py $i 

cp ../../../c.model c.model
cp ../../../model.f90 model.f90

echo Run run_cont.auto
# This should not take more than 10 mins - if it does, cancel the run and create new model
timeout 600 /home/users/a/g/aguidat/auto-07p/auto-07p/bin/auto ../../../run_cont.auto 


echo Run stoch_sims_diy.py
python3 ../../../stoch_sims.py $i



done



# Convert label data and split into training, test, validation
echo "Convert data to correct form for training"
python3 ../../../to_traindata.py $bif_max $batch_num


# Compute residual dynamics after Lowess smoothing for each time series
echo "Compute residuals"
python3 ../../../compute_resids.py $bif_max $batch_num


# End timer
end=`date +%s`
runtime=$((end-start))
echo "Job successfully finished in time of $runtime" seconds.

# Change back to original directory
cd ../../../


