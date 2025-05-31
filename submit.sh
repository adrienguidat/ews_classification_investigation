# ------------------------------------------------------------------------
# SLURM script to run code
# Created by Adrien Guidat for the master thesis "Investigation on EWS detection and classification through Deep Neural Networks" (2025).
#
# ------------------------------------------------------------------------



#!/bin/bash
#SBATCH --job-name=sub1
#SBATCH --output=res1.txt
#SBATCH --ntasks=1
#SBATCH --time=20:00:00
#SBATCH --mem-per-cpu=5000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1





ml releases/2020b

ml TensorFlow/2.5.0-fosscuda-2020b
ml scikit-learn/0.23.2-fosscuda-2020b
ml matplotlib/3.3.3-fosscuda-2020b




#source ../venv/bin/activate  -> to use when running a python code, should have been already installed by the command: python3 -m venv venv


#./DL_apply.py
#./run_single_batch.sh        ->if batch file, no need to activate the virtual environment

#deactivate()                 ->if has been activated


