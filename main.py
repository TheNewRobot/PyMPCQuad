import math
import h5py
import numpy as np
from python_mod.plots import fig_animate, fig_plot, create_recording
from python_mod.robot import get_params, dynamics_SRB
from scipy.linalg import expm
import time
from python_mod.refGen import fcn_bound_ref_traj, fcn_gen_XdUd, fcn_get_disturbance
from decimal import Decimal, getcontext
from python_mod.FSM import fcn_FSM
from python_mod.getMPC import sim_MPC, get_QP
from qpsolvers import solve_qp
import copy
from scipy.integrate import solve_ivp
from python_mod.utils import print_progress_bar

# pip freeze > requirements.txt

# # --- Load data from a MATLAB File only to plot
# matlab_file = h5py.File('mpc_experiment.mat', 'r')
# tout_matrix = np.array(matlab_file['tout'])
# tout_matrix = tout_matrix.flatten()
# xout_matrix = np.array(matlab_file['Xout']).T
# uout_matrix = np.array(matlab_file['Uout']).T
# xdout_matrix = np.array(matlab_file['Xdout']).T
# udout_matrix = np.array(matlab_file['Udout']).T
# uext_matrix = np.array(matlab_file['Uext']).T


gait = 0

p = get_params(gait)

p['flag_movie'] = 1
getcontext().prec = 28 
use_Casadi = 0
SimTimeDuration = 2
p['playSpeed'] = 20 # Increase this as multiples of 10

p['predHorizon'] = 15 # N steps
N = p['predHorizon'] 
p['Tmpc'] = 1/50
p['simTimeStep'] = 1/200
dt_sim = p['simTimeStep']
p['p_ext'] = np.array([p['L']/2, p['W']/2, p['d']])

MAX_ITER = int(np.floor(SimTimeDuration/p['simTimeStep']))

if gait >=0:
    p['yaw_d'] = 0 # np.pi/4
    p['acc_d'] = 0.5
    p['vel_d'] = np.array([0.5*np.cos(p['yaw_d']), 0.5*np.sin(p['yaw_d'])]).reshape(-1,1)
    p['wb_d'] = np.array([0.0,0.0,0.0]).reshape(-1,1)
    p['ang_acc_d'] = np.array([0.0,0.0,0.5]).reshape(-1,1)

if gait == -1:
    p['roll_d'] = 0
    p['pitch_d'] = np.pi/4
    p['yaw_d'] = 0
    p['ang_acc_d'] = np.array([0,0.5,0]).reshape(-1,1)


# Model Predictive Controller
# --- initial condition ---
# Xt = [pc dpc vR wb pf]': [30,1]
if gait == 1:
    p, Xt, Ut = fcn_bound_ref_traj(p)
else:
    Xt, Ut = fcn_gen_XdUd(0, [], np.array([[1.0, 1.0, 1.0, 1.0]]).T, p)
    pass


tstart = 0
tend = dt_sim

tout = []
Xout = np.empty((30, 0)) # because in this shape we have the output of the vector 
Uout = np.empty((0, 12))
Xdout  = np.empty((0, 30))
Udout  = np.empty((0, 12)) 
Uext   = np.empty((0, 3))
FSMout = np.empty((0, 4))


# --- simulation ----
h_waitbar = print('Calculating...') #Waitbar

start_time = time.time()
Ut_ref = np.empty((12, 0))  # n_columns is the number of columns in your array
Ud_ref = np.empty((12, 0))  # TODO: Change this inmediately
Selection = []
dt_step = dt_sim/40

for ii in range(1, MAX_ITER + 1):
    
    # --- time vector ---
    # ---Current interpolation of time
    t_ = dt_sim * (ii - 1) + p['Tmpc'] * np.arange(0, p['predHorizon'])

    # --- FSM ---
    if gait == 1:
        #FSM, Xd, Ud, Xt = fcn_FSM_bound(t_, Xt, p)
        pass
    else:
        FSM, Xd, Ud, Xt = fcn_FSM(t_, Xt, p, ii)
    
    # set up selection matrix (makes problem infeasible)
    # S = 0 if feet not in contact
    S = copy.copy(Ud)
    
    S[S != 0] = 1
    Selection.append(S)
    # current stance feet
    idx = np.any(np.reshape(S, (3, 4 * N), order='F'), axis=0)
    stance_feet = np.ones((3, 4 * N)) * idx.reshape(1,-1)

    # future idx for feet in contact for horizon
    idx = np.reshape(idx, (4, p['predHorizon']), order='F')
    num_feet_contact = np.count_nonzero(idx[:, 0])
    
    ## --- MPC ----
    if use_Casadi:
        # solve QP using qpoases through casadi
        # 1 - single shooting, 0 - multi shooting
        single_shoot = 1
        zval = sim_MPC(Xt, Xd, Ud, idx, N, p, single_shoot)
    else:
        
        # form QP using explicit matrices
        [f, G, A, b] = get_QP(Xt,Xd,Ud,idx,N,p)
        #solve QP using quadprog
        zval = solve_qp(G,f,A,b,solver="clarabel")

    # get the foot forces       
    contacts = 1
    contact_seq = idx[:, 0]
    Ut = np.zeros((12, 1))
    
    for i in range(4):
        if contact_seq[i] == 1:
            Ut[i * 3:(i + 1) * 3] = (zval[(contacts - 1) * 3:contacts * 3] * np.array([1.0, 1.0, 1.0])).reshape(-1,1)
            contacts += 1
        else:
            Ut[i * 3:(i + 1) * 3] = np.array([0.0, 0.0, 0.0]).reshape(-1,1)

    # logging
    i_hor = 1
    Ut_ref = np.concatenate((Ut_ref, Ut), axis=1)
    Ud_ref = np.concatenate((Ud_ref, Ud[:, i_hor][:, None]), axis=1)

    ## --- external disturbance ---
    u_ext, p_ext = fcn_get_disturbance(tstart, p)
    p['p_ext'] = p_ext  # position of external force
    u_ext = np.zeros_like(u_ext)

    solution_ode = solve_ivp(dynamics_SRB,
                            [tstart, tend],
                             Xt.flatten(),
                             args=(Ut.flatten(), Xd, u_ext, p))
    
    t = solution_ode.t
    X = solution_ode.y

    ## --- update ---
    Xt = X[:, -1].reshape(-1, 1) #Later this will be flattened
    tstart = tend
    tend = tstart + dt_sim

    ## --- log ---
    lent = len(t[1:])
    tout = np.concatenate((tout, t[1:])) 
    Xout = np.hstack((Xout, X[:, 1:]))
    Uout = np.vstack((Uout, np.tile(Ut.flatten(), (lent, 1))))
    Xdout = np.vstack((Xdout, np.tile(Xd[:, 0], (lent, 1))))
    Udout = np.vstack((Udout, np.tile(Ud[:, 0], (lent, 1))))
    Uext = np.vstack((Uext, np.tile(u_ext.flatten(), (lent, 1))))
    FSMout = np.vstack((FSMout, np.tile(FSM.flatten(), (lent, 1))))
    print_progress_bar(ii, MAX_ITER, prefix='Progress:', suffix='Complete', length=50)


elapsed_time = time.time() - start_time
print(f"Number of iterations: {MAX_ITER}")
print(f"Elapsed Time for the simulation: {elapsed_time:.2f} seconds")

[t,EA,EAd] = fig_animate(tout,
                         Xout.T,
                         Uout,
                         Xdout,
                         Udout,
                         Uext,
                         p)
if p['flag_movie']: 
   create_recording(t,p)

time.sleep(3)
print("\nAll simulations have finished. You can press control + C to quit the script after Figure 2 appears!") 

fig_plot(tout,
         Xout.T,
         Uout,
         Xdout,
         Udout,
         Uext,
         p)

