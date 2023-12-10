import numpy as np
from scipy.linalg import expm
from .utils import hatMap, bz_int, polyval_bz
import copy 

def fcn_bound_ref_traj(p):
    '''
    This function finds the initial condition for periodic bounding
    The calculation is based on the paper (citation):
    Park, Hae-Won, Patrick M. Wensing, and Sangbae Kim. 
    "High-speed bounding with the MIT Cheetah 2: Control design and experiments."
    The International Journal of Robotics Research 36, no. 2 (2017): 167-192.
    '''

    mass, J, g, Tst, Tsw = p['mass'], p['J'], p['g'], p['Tst'], p['Tsw']
    T = Tst + Tsw
    Tair = 1/2 * (Tsw - Tst)

    b_co = np.array([0, 0.8, 1, 1, 0.8, 0])
    b_ = np.mean(b_co)
    
    # Fz
    # 2 * alpha * b_ * Tst = mass * g * T
    alpha_z = (mass * g * T) / (2 * b_ * Tst)
    Fz_co = alpha_z * b_co
    dz_co = bz_int(Fz_co/mass - g, 0, Tst)
    z_co = bz_int(dz_co, 0, Tst)

    # first principle: integration
    dz0 = -1/(Tst + Tair)*(z_co[-1] + Tair*(dz_co[-1] + g*Tst) - 1/2*g*((Tst + Tair)**2 - Tst**2))

    dz_co = bz_int(Fz_co/mass - g, dz0, Tst)
    z_co = bz_int(dz_co, p['z0'], Tst)

    # theta
    alpha_th = 140 * J[1, 1]
    tau_co = -alpha_th * b_co

    dth_co = bz_int(tau_co/J[1, 1], 0, Tst)
    # by symmetry
    dth0 = -1/2 * dth_co[-1]

    th0 = dth0 * Tair / 2
    dth_co = bz_int(tau_co/J[1, 1], dth0, Tst)
    th_co = bz_int(dth_co, th0, Tst)

    # output B-spline coefficient
    p['Fz_co'] = Fz_co
    p['dz_co'] = dz_co
    p['z_co'] = z_co

    p['tau_co'] = tau_co
    p['dth_co'] = dth_co
    p['th_co'] = th_co

    # intial condition
    R0 = expm(hatMap([0.0, th0, 0.0]))
    Xt = np.array([0.0, 0.0, p['z0'], 0.0, 0.0, dz0, *R0.flatten(), 0.0, dth0, 0.0])
    Xt = np.append(Xt, p['pf34'].flatten(order='F'))  # You can try with .flatten(order='F')
    Ut = np.tile(np.array([0.0, 0.0, 1/4*p['mass']*p['g']]).reshape(-1,1), (4, 1))
    return p, Xt, Ut

def fcn_gen_XdUd(t, Xt, bool_inStance, p):
    # parameters
    gait = p['gait']
    # generate reference trajectory
    if isinstance(t, (int, np.int32, np.int64)):
         t = np.array([t])
    lent = len(t)
    Xd = np.zeros((30, lent))
    Ud = np.zeros((12, lent))
    Rground = p['Rground']  # ground slope
    for ii in range(lent):
        # any walking gait
        if gait >= 0:
            acc_d = p['acc_d']
            vel_d = p['vel_d']
            yaw_d = p['yaw_d']
            wb_d = copy.copy(p['wb_d']) # initialy wd is zero but later is updated by the reference
            ang_acc_d = p['ang_acc_d']
            
            # linear motion
            pc_d = np.array([0.0, 0.0, p['z0']])
            dpc_d =  np.zeros(3)
            
            for jj in range(2):
                if t[ii] < (vel_d[jj] / acc_d):
                    dpc_d[jj] = acc_d * t[ii] # TODO here this is a problem
                    pc_d[jj] = 1/2 * acc_d * t[ii]**2
                else:
                    dpc_d[jj] = vel_d[jj]
                    pc_d[jj] = vel_d[jj] * t[ii] - 1/2 * vel_d[jj]**2 / acc_d
            
            # angular motion
            ea_d = np.array([0.0, 0.0, 0.0])
            
            if Xt is not None:
                wb_d[2] = ang_acc_d[2] * t[ii]
                ea_d[2] = 1/2 * ang_acc_d[2] * t[ii]**2
            
            if ea_d[2] >= yaw_d:
                wb_d[2] = 0.0
                ea_d[2] = yaw_d
            
            vR_d = np.reshape(expm(hatMap(ea_d)), (9, 1), order='F')
            pfd = np.reshape(Rground @ p['pf34'], (12, 1), order='F')
        
        # pose control only for one gait
        if gait == -1:
            roll_d = p['roll_d']
            pitch_d = p['pitch_d']
            yaw_d = p['yaw_d']
            ang_acc_d = p['ang_acc_d']
            
            ea_d = np.array([0.0, 0.0, 0.0])
            wb_d = np.array([0.0,0.0, 0.0])
            
            if Xt is not None:
                wb_d[1] = ang_acc_d[1] * t[ii]
                ea_d[1] = 1/2 * ang_acc_d[1] * t[ii]**2
            
            if ea_d[1] >= pitch_d:
                wb_d[1] = 0
                ea_d[1] = pitch_d
            
            pc_d = np.array([0.0, 0.0, p['z0']]) # 3
            dpc_d = np.array([0.0, 0.0, 0.0]) # 3
            vR_d = np.reshape(expm(hatMap(ea_d)), (9, 1), order='F') # 9
            # wb_d = 3
            pfd = np.reshape(Rground @ p['pf34'], (12, 1), order='F') # 12
        
        Xd[:, ii] = np.concatenate([pc_d, dpc_d, vR_d, wb_d, pfd], axis=None) # Vertical concatenation
       
        
        # force for backflip
        if gait == -3:
            Ud[:, ii] = 0.0
        else:
            sum_inStance = np.sum(bool_inStance[:,ii])
            
            if sum_inStance == 0:  # four legs in swing
                Ud[:, ii] = np.zeros(12)
            else:
                Ud[[2, 5, 8, 11], ii] = bool_inStance[:,ii] * (p['mass'] * p['g'] / sum_inStance)
        #breakpoint()
    return Xd, Ud


def fcn_get_disturbance(t, p):
    '''
    calculates and returns external disturbance vectors u_ext and a point vector p_ext based on the input time t and a set of predefined conditions. 
    The disturbance is determined by evaluating Bezier curves with specific control points during certain time intervals.
    '''
    bz_w = np.array([0, 0.5, 1, 1, 0.5, 0]) # Control Points for the Bezier Curve

    if (0.5 <= t <= 1.3):  # small disturbances
        s_w = (t - 0.5) / 0.8
        w = polyval_bz(8 * bz_w, s_w)
    elif (2.3 <= t <= 3.1):
        s_w = (t - 2.3) / 0.8
        w = polyval_bz(22 * bz_w, s_w)
    else:
        w = 0.0

    u_ext = np.array([0.0, w, 0.0])
    p_ext = p['p_dist']  # external force point in body frame

    return u_ext, p_ext
