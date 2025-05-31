#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

SCRIPT TO:
Get info from b.out files (AUTO files)
Run stochastic simulations up to bifurcation points for ts_len+200 time units
Detect transition point using change-point algorithm
Ouptut time-series of ts_len time units prior to transition

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
import matplotlib.pyplot as plt
import csv
import os
import shutil
import subprocess



# Function to convert b.out files into readable form
from convert_bifdata import convert_bifdata

    
# Function to simulate model
from sim_model import sim_model


# Function to detect transition point in time series
from trans_detect import trans_detect


# Create model class
class Model():
    pass
 

# Create directory if does not exist
if not os.path.exists('output_sims'):
    os.makedirs('output_sims')
if not os.path.exists('output_labels'):
    os.makedirs('output_labels')
if not os.path.exists('output_counts'):
    os.makedirs('output_counts')

# Get command line variables
import sys

indice = int(sys.argv[1])



# Noise amplitude
sigma_tilde = 0.01
print('Using sigma_tilde value of {}'.format(sigma_tilde))


seq_id = 500000 + indice




# Parameter labels
parlabels_a = ['a'+str(i) for i in np.arange(1,11)]
parlabels_b = ['b'+str(i) for i in np.arange(1,11)]
parlabels = parlabels_a + parlabels_b


#----------
# Extract info from b.out files
#–-------------

# Initiate list of models
list_models = []

# Assign attributes to model objects from b.out files
print('Extract data from b.out files')

out = convert_bifdata('output_auto/b.outa2')
    
# Assign bifurcation properties to model object
bif_param = out['bif_param']
bif_type = out['type']
bif_value = out['value']



model_temp = Model()


# Assign bifurcation properties to model object

model_temp.bif_value=0
if bif_value<100 and bif_value>-100:
    model_temp.bif_value = bif_value
print("bif_type")
    
if bif_type=='BP':                                 #changed type to desired type 'BP'=branch point, 'HB' = Hopf, 'LP' = limit point=fold
    with open("log_resultats.txt", "a") as f:
        f.write(f"{indice},1\n")                   # 1 indicates a bifurcation
        model_temp.bif_type =  'PF'
else:
    with open("log_resultats.txt", "a") as f:
        f.write(f"{indice},0\n")                   # 0 indicates no bifurcation
        model_temp.bif_type =  'NULL'
        

model_temp.bif_param = 'a2'


def load_auto_output(filepath,intg=0):
    # Lire le fichier et extraire les lignes contenant les données
    data_lines = []
    with open(filepath, 'r') as file:
        for line in file:
            if line.strip() and line.strip()[0].isdigit():
                parts = line.strip().split()
                if intg==1:print(parts)
                if len(parts) == 8:
                    try:
                        a2 = float(parts[4])
                        x = float(parts[6])
                        data_lines.append((a2, x))
                    except ValueError:
                        continue
                
    return pd.DataFrame(data_lines, columns=['a2', 'x'])
    

def is_symmetric(df_up,df_down, tol=0.1):
    #function to detect symmetry betwen df_up and df_down
    counter = 0
    
    # Merge sur 'a2' pour comparer les bonnes paires
    merged = df_up.merge(df_down, on='a2', suffixes=('_up', '_down'))

    for _, row in merged.iterrows():
        if abs(abs(row['x_up']) - abs(row['x_down'])) > tol:
            counter += 1

    return counter <= 10

      
    
    

def plot_branch(df_up,df_down, filename='branch_plot.png'):
    #function to plot df_up and df_down

    plt.figure(figsize=(8, 5))

    plt.plot(df_up['a2'], df_up['x'], label='Lower branch', color='blue')
    plt.plot(df_down['a2'], df_down['x'], label='Upper branch', color='orange')


    plt.axhline(0, color='gray', linestyle='--')
    plt.xlabel('a2 (bifurcation parameter)')
    plt.ylabel('x (state variable)')
    plt.title('Branch profile from AUTO output')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()





def extract_bp_label(filepath):
    """
    Parse a b.outA2 AUTO file to find the first branch point (TY=1)
    and return its label (LAB).
    """
    with open(filepath, 'r') as file:
        for line in file:
            if line.strip() and line.strip()[0].isdigit():
                parts = line.strip().split()
                if len(parts) == 8:
                    try:
                        ty = int(parts[2])
                        lab = int(parts[3])
                        if ty == 1:  # TY=1 means branch point
                            return lab
                    except ValueError:
                        continue
    return None









def run_auto_with_params(label, isw, run_name):
    
    #90 model
    xxbasefile = 'model.f90'
    xxfile = f'model_{run_name}.f90'

    with open(xxbasefile, 'r') as f:
        contentxx = f.read()

    with open(xxfile, 'w') as f:
        f.write(contentxx)
        
    
    
    
    #c model
    cbasefile = 'c.model'
    cfile = f'c.model_{run_name}'

    with open(cbasefile, 'r') as f:
        contentc = f.read()

    #contentc = contentc.replace("NDIM=   2, IPS =   1, IRS =   0, ILP =   1", f"NDIM=   2, IPS =   1, IRS =   {label:3d}, ILP =   1")  # Set starting label
    #contentc = contentc.replace("NTST=  15, NCOL=   4, IAD =   3, ISP =   1, ISW = 1, IPLT= 0, NBC= 0, NINT= 0", f"NTST=  15, NCOL=   4, IAD =   3, ISP =   1, ISW = {isw:3d}, IPLT= 0, NBC= 0, NINT= 0")    # Direction
    

    with open(cfile, 'w') as f:
        f.write(contentc)
        
    
    
        
    print("in run_atuo")
    # Create a modified AUTO constants file
    basefile = '../../../run_cont.auto'
    runfile = f'{run_name}.auto'

    with open(basefile, 'r') as f:
        content = f.read()
        
    insertion_code = """
    branchpoints = out("BP")
    for solution in branchpoints:
        bp = load(solution, ISW=-1, NTST=50)
        # Compute forwards
        print("Solution label", bp["LAB"], "forwards")
        fw = run(bp)
        # Compute backwards
        print("Solution label", bp["LAB"], "backwards")
        bw = run(bp, DS=-0.01)
        both = fw + bw
        save(fw,'b_down.outa2')
        save(bw,'b_up.outa2')
        !mv b.b_down.outa2* output_auto
        !mv b.b_up.outa2* output_auto
        merged = merge(both)
        out = out + merged
    out = relabel(out)
    """


    content = content.replace("    save(out,'out'+par)", f"    save(out,'b_{run_name}.outa2')")
    content = content.replace('!mkdir -p output_auto\n', '')
    content = content.replace("for par in ['a1','a2']:\n","for par in ['a2']:\n")
    content = content.replace("    !mv b.out* output_auto",f"    !mv b.b_{run_name}.out* output_auto")
    #content = content.replace("    !mv s.out* output_auto",f"    !mv s.b_{run_name}.out* output_auto")
    content = content.replace("    !rm s.out*","    !rm s.*")
    content = content.replace("    !rm d.out*","    !rm d.*")
    content = content.replace("    model = load('model')", f"    model = load('model_{run_name}')")
    content = content.replace("print('HEY')","print('INCUSTO')")
    #content = content.replace("print('Running AUTO')","!cp s.outa2 s.b_branch_down.outa2")
    content = content.replace("out = run(model)", "out = run(model)\n" + insertion_code)
    content = content.replace("    #!rm s.out*","    !rm s.*")



    with open(runfile, 'w') as f:
        f.write(content)
        
    

    
    #os.system(f"timeout 600 /home/users/a/g/aguidat/auto-07p/auto-07p/bin/auto {runfile}")
    try:
      result = subprocess.run(
          ["timeout", "600", "/home/users/a/g/aguidat/auto-07p/auto-07p/bin/auto", runfile],  #Type @r xxx yyy zzz to run AUTO with equations- le xxx.f90 restart data- le s.yyy and constants- le c.zzz "@R model outa2 model_{run_name}"
          stdout=subprocess.PIPE,
          stderr=subprocess.PIPE,
          universal_newlines=True,  # remplace `text=True` pour Py 3.6
          check=True
      )

      print("AUTO ran successfully")
      print("STDOUT:", result.stdout)
    except subprocess.CalledProcessError as e:
      print("AUTO exited with error")
      print("STDERR:", e.stderr)
    except FileNotFoundError:
      print("AUTO executable or timeout not found")
    except subprocess.TimeoutExpired:
      print("AUTO run timed out")
      
  


def follow_two_branches(bp_label):
    print("in followww")
    #run_auto_with_params(bp_label, isw=1, run_name='branch_up')
    run_auto_with_params(bp_label, isw=-1, run_name='branch_down')

    # Convert both outputs (if you want to merge later)
    df_up = load_auto_output('output_auto/b.b_down.outa2') #convert_bifdata('output_auto/b.b_branch_up.outa2')['data']
    df_down = load_auto_output('output_auto/b.b_up.outa2') #convert_bifdata('output_auto/b.b_branch_down.outa2')['data']

    return df_down,df_up



bp_label = extract_bp_label("output_auto/b.outa2")

if bp_label is not None:
    print(f"✅ Found BP with label: {bp_label}")
    df_down, df_up = follow_two_branches(bp_label) #df_up, 
    plot_branch(df_down,df_up, filename='full_symmetry_check_'+str(indice)+'.png')
    print("is symmetric?: ")
    val=is_symmetric(df_up,df_down)
    print(val)
    
    with open("log_resultats_bis.txt", "a") as f:
            f.write(f"{indice},{val}\n")
else:
    print("❌ No branch point (TY=1) found in b.outa2.")




# Import parameter values for the model
with open('output_model/pars.csv') as csvfile:
    pars_raw = list(csv.reader(csvfile))
par_list = [float(p[0]) for p in pars_raw]
par_dict = dict(zip(parlabels,par_list))
# Assign parameters to model object
model_temp.pars = par_dict
    

# Import equilibrium data as an array
with open('output_model/equi.csv') as csvfile:
    equi_raw = list(csv.reader(csvfile))   
        
equi_list = [float(e[0]) for e in equi_raw]
equi_array = np.array(equi_list)
# Assign equilibria to model object
model_temp.equi_init = equi_array

# Import recovery rate
with open('output_model/rrate.csv') as csvfile:
    rrate_raw = list(csv.reader(csvfile))          
rrate = float(rrate_raw[0][0])

# Add model to list
list_models.append(model_temp)
    

# Separate models into their bifurcation types
hb_models = [model for model in list_models if model.bif_type == 'HB']
bp_models = [model for model in list_models if model.bif_type == 'BP']
lp_models = [model for model in list_models if model.bif_type == 'LP']
pf_models = [model for model in list_models if model.bif_type == 'PF']





#-------------------
## Simulate models
#------------------
    
# Construct noise as in Methods
rv_tri = np.random.triangular(0.75,1,1.25)
sigma = np.sqrt(2*rrate) * sigma_tilde * rv_tri

ts_len=500 #500



print('Begin simulating model up to bifurcation points')
# Loop through model configurations (different bifurcation params)
for i in range(len(list_models)):
    model = list_models[i]
    
    # Pick sample spacing randomly from [0.1,0.2,...,1]
    dt_sample = np.random.choice(np.arange(1,11)/10)
    # Define length of simulation
    # This is 200 points longer than ts_len
    # to increase the chance that we can extract ts_len data points prior to a transition
    # (transition can occur before the bifurcation is reached)
    series_len = ts_len + 200

    #PF

    print('Simulating a PF traj')
    df_out = sim_model(model, dt_sample=dt_sample, series_len=series_len,
                        sigma=sigma)
    # Detect transition point
    trans_time = trans_detect(df_out)
    # Only if trans_time > ts_len, keep and cut trajectory
    if trans_time > ts_len:
        df_cut = df_out.loc[trans_time-ts_len:trans_time-1].reset_index()
        # Have time-series start at time t=0
        df_cut['Time'] = df_cut['Time']-df_cut['Time'][0]
        df_cut.set_index('Time', inplace=True)
        # Export
        df_cut[['x']].to_csv('output_sims/tseries'+str(seq_id)+'.csv')
        df_label = pd.DataFrame([1])
        df_label.to_csv('output_labels/label'+str(seq_id)+'.csv',
                        header=False, index=False)

        seq_id += 1
        print('   Achieved {} steps - exporting'.format(ts_len))
    else:
        print('   Transitioned before {} steps - no export'.format(ts_len))
            

print('Simulations finished\n')

list_counts = np.array([counter_PF, counter_TR, counter_LP, counter_HB])
np.savetxt('output_counts/list_counts.txt',list_counts, fmt='%i')
    
    









    
