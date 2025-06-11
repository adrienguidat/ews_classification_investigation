# ews_classification_investigation
This repository contains the code used for the master thesis "Investigation on EWS detection and classification through Deep Neural Networks" undertook by Adrien Guidat under the supervision of E. Massart and P. Absil at EPL, UCLouvain.
It is heavily based on the code developed by Bury et al. for their article "Thomas M. Bury. Deep learning for early warning signals of tipping points. PNAS, 118(39),(2021)". Their code is available at : https://github.com/ThomasMBury/deep-early-warnings-pnas .


## Requirement

The bifurcation continuation software AUTO-07P is required to generate the training data. It is available at https://github.com/auto-07p/auto-07p .

## Directories

**./dl_train:** Code to train the DL classifier.

**./test_pitch:** Code to test the classifier on a system exhibiting a pitchfork bifurcation.

**./training_data:** Code to generate training data for the deep learning algorithm.


1. **Generate the training data** The script 'run_single_batch.sh' runs the generation of time series exhibiting pitchfork bifurcations. To change the type of bifurcation, one has to change the generated model in 'gen_model.py'. 'c.model', 'model.f90' and 'run_cont.auto' are files needed by AUTO-07P.
2. **Training the network** The file 'DL_training.py' trains the network on the generated training set. The file 'DL_confusion.py' evaluates the network on the validation test extracted from the generated training set. Finally the file 'DL_apply.py' is used to make predictions of the model on any given time series.
3. **Testing the network** The file 'sim_hoop_pitch.py' simulates a rotating hoop system exhibiting pitchfork bifurcations and generates time series. The file 'sim_hoop_pitch_null.py' generates time series without bifurcations. 'compute_ews.py' and 'compute_ktau.py' serve to evaluate generic EWS on the time series to later compare them to the predictions. 'compute_roc.py' is used to assess predictions accuracy by tracing ROC curves.  
   
  

 

