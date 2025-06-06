#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Script to generate dynamical systems with random parameters and a stable equi:
    Generate random parameters
    Simulate dyanmical system
    Check convergence
    If yes, output parameters and equilibrium
    
    Dynamical system
    dx/dt = a1 + a2 x + a3 y + a4 x^2 + a5 xy + a6 y^2 + a7 x^3 + a8 x^2 y + a9 x y^2 + a10 y^3
    dy/dt = b1 + b2 x + b3 y + b4 x^2 + b5 xy + b6 y^2 + b7 x^3 + b8 x^2 y + b9 x y^2 + b10 y^3
    
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
import scipy.integrate as spi
import os
import random



# Make folders for this directory
if not os.path.exists('output_model'):
    os.makedirs('output_model')

import sys
indice = int(sys.argv[1])


# Stop when system with convergence found
conv=False
while conv==False:
    
    
    pars = np.random.normal(loc=0, scale=0.2, size=20) 


    sparsity = np.random.uniform(0.6, 1.0)  
    non_crucial = list(set(range(20)) - {1,6,3,0,10,2})  # autorise b3 too

    n_to_zero = int(len(non_crucial) * sparsity)
    zero_indices = random.sample(non_crucial, n_to_zero)
    for index_zero in zero_indices:
        pars[index_zero] = 0
    
    
    pars[0] = 0
    pars[10] = 0
    
    a=0
    
    
    if np.random.rand() < 0.3:
        a = np.random.uniform(-1, 1)  # a1
    if np.random.rand() < 0.3:
        b = np.random.uniform(-0.2, 0.2)  # b1
    

    

    # mu, function of bifurcation type
    mu = round(np.random.uniform(-2, 2),2)
    subcritical = np.random.rand() < 0.5
    if subcritical:
        mu = -abs(mu)
        cubic_sign = 1
    else:
        cubic_sign = -1
        
    rnd = round(np.random.uniform(0.2, 2),2)

    # dx/dt = a1 + a2 x + a3 y + a4 x^2 + a5 xy + a6 y^2 + a7 x^3 + a8 x^2 y + a9 x y^2 + a10 y^3
    # dy/dt = b1 + b2 x + b3 y + b4 x^2 + b5 xy + b6 y^2 + b7 x^3 + b8 x^2 y + b9 x y^2 + b10 y^3
    
    pars_a = pars[:10]
    pars_b = pars[10:]
    
    pars_a[0] = a
    pars_a[1] = mu          
    pars_a[6] = cubic_sign * rnd 
    pars_a[3] = 0 #to not confound with trans
    pars_b[2] = np.random.normal(loc=0, scale=0.3, size=1) 
    pars_b[7] = np.random.normal(loc=0, scale=0.3, size=1) 
    pars_b[1] = np.random.normal(loc=0, scale=0.3, size=1) 
    
              

    # Initial Conditions
    s0 = np.random.normal(loc=0, scale=2, size=2)
    

    
    # Define derivative function
    def f(s,t0,a,b):
        '''
        s: 2D state vector
        a: 10D vector of parameters for x dynamics
        b: 10D vecotr of parameterrs for y dyanmics
        '''
        x = s[0]
        y = s[1]
        # Polynomial forms up to third order
        polys = np.array([1,x,y,x**2,x*y,y**2,x**3,x**2*y,x*y**2,y**3])
        # Output
        dydt = np.array([np.dot(a,polys), np.dot(b,polys)])
        
        return dydt
    
    
    ## Simulate the system
    t = np.arange(0., 100, 0.01)
    s, info_dict = spi.odeint(f, s0, t, args=(pars_a,pars_b),
                              full_output=True,
                              hmin=1e-14,
                              mxhnil=0)
    
    # Put into pandas
    df_traj = pd.DataFrame(s, index=t, columns=['x','y'])
    
    # Does the sysetm blow up?
    if df_traj.abs().max().max() > 1e3:
        print('System blew up - run new model')
        continue
    
    # Does the system contain Nan?
    if df_traj.isna().values.any():
        print('System contains Nan value - run new model')
        continue
        
    # Does the system contain inf?
    
    if np.isinf(df_traj.values).any():
        print('System contains Inf value - run new model')
        continue
    
    # Does the system converge?
    # Difference between max and min of last 10 data points
    diff = df_traj.iloc[-10:-1].max() - df_traj.iloc[-10:-1].min()
    # L2 norm
    norm = np.sqrt(np.square(diff).sum())
    # Define convergence threshold
    conv_thresh = 1e-8
    if norm > conv_thresh:
        print('System does not converge - run new model')
        continue
    
    
    # If made it this far, system is good for bifurcation continuation
    print('System converges - export equilibria and parameter values\n')
    # Export equilibrim data
    equi = df_traj.iloc[-1].values
    np.savetxt("output_model/equi.csv", equi, delimiter=",")
    
    # Export parameter data
    pars = np.concatenate([pars_a,pars_b])
    np.savetxt("output_model/pars.csv", pars, delimiter=",")


    ## Compute the dominant eigenvalue of this system at equilbrium
    
    # Compute the Jacobian at the equilibrium
    x=equi[0]
    y=equi[1]
    [a1,a2,a3,a4,a5,a6,a7,a8,a9,a10]=pars[:10]
    [b1,b2,b3,b4,b5,b6,b7,b8,b9,b10]=pars[10:]
    
    # df/dx
    j11 = a2 + 2*a4*x + a5*y + 3*a7*x**2 + 2*a8*x*y + a9*y**2
    # df/dy
    j12 = a3 + 2*a6*y + a5*x + 3*a10*y**2 + 2*a9*x*y + a8*x**2
    # dg/dx
    j21 = b2 + 2*b4*x + b5*y + 3*b7*x**2 + 2*b8*x*y + b9*y**2
    # dg/dy
    j22 = b3 + 2*b6*y + b5*x + 3*b10*y**2 + 2*b9*x*y + b8*x**2

    # Assign component to Jacobian
    jac = np.array([[j11,j12],[j21,j22]])
    
    # Compute eigenvalues
    evals = np.linalg.eigvals(jac)
    
    # Compute the real part of the dominant eigenvalue (smallest magnitude)
    re_evals = [lam.real for lam in evals]
    dom_eval_re = max(re_evals)
    # Recovery rate is amplitude of this
    rrate = abs(dom_eval_re)

    # Export recovery rate (magnitude of real part of dom eval)
    np.savetxt('output_model/rrate.csv', np.array([rrate]))  

    conv=True
    
    
    
    
    
    
    