import cv2
import numpy as np
import matplotlib.pyplot as plt
from .utils import fcn_X2EA, veeMap
from scipy.linalg import logm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import matplotlib.gridspec as gridspec
from .robot import *
import sys
import time

#TODO: Change the p variables to be fixed
def fig_animate(tout,Xout, Uout, Xdout, Udout, Uext,p):
# For Trot
    
    p_simTimeStep = p['simTimeStep']
    p_playSpeed = p['playSpeed']
    flag_movie =  p['flag_movie']

    ## Smoothen for animations
    ts = np.arange(tout[0], tout[-1], p_simTimeStep)

    # Xout = Xout.reshape((len(tout), -1))

    # # Perform interpolation for each row of Xout
    # X_interpolated_rows = [np.interp(ts, tout, X_row) for X_row in Xout]

    # # Stack the interpolated rows horizontally
    # X = np.vstack(X_interpolated_rows).T
    X = np.vstack([np.interp(ts, tout, X_row) for X_row in Xout.T]).T
    U = np.vstack([np.interp(ts, tout, U_row) for U_row in Uout.T]).T
    Xd = np.vstack([np.interp(ts, tout, Xd_row) for Xd_row in Xdout.T]).T
    Ud = np.vstack([np.interp(ts, tout, Ud_row) for Ud_row in Udout.T]).T
    Ue = np.vstack([np.interp(ts, tout, Uext_row) for Uext_row in Uext.T]).T

    # Loop through frames
    # TODO: There was code here
    fig = plt.figure(figsize=(12, 10), constrained_layout=False)
    fig.suptitle('MPC Experiment')
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.2, hspace=0.2)

    N = 3
    M = 3

    nt = len(ts)
    EA = np.zeros((nt, 3))
    EAd = np.zeros((nt, 3))

    for ii in range(nt):
        EA[ii, :] = fcn_X2EA(X[ii,:])
        EAd[ii, :] = fcn_X2EA(X[ii,:])

    # For ZOH force
    t2 = np.repeat(ts, 2)
    t2 = np.append(t2[1:], t2[-1])  # Adjusting the first and last elements
    U2 = np.repeat(U, 2, axis=0)


    # Get angles for each time
    theta = np.empty((0, 3))
    thetad = np.empty((0, 3))
    
    for i in range(len(X)):
        R = np.reshape(X[i, 6:15], (3, 3),order='F')
        Rd = np.reshape(Xd[i, 6:15], (3, 3),order='F')
        theta = np.vstack((theta, veeMap(logm(R)))) # TODO here
        thetad = np.vstack((thetad, veeMap(logm(Rd))))
    
    # Subplot handles
    spec = gridspec.GridSpec(ncols=N, nrows=M, figure=fig)
    h_x = fig.add_subplot(spec[0,2])
    h_dx = fig.add_subplot(spec[1,2])
    h_w = fig.add_subplot(spec[2,2])
    h_u = fig.add_subplot(spec[2,:2])

    # Loop through frames
    for ii in range(0, nt, p_playSpeed):
        try:
            # The main animation, this should be passed to fig plot robot
            pcom = X[ii, 0:3]
            h_main = fig.add_subplot(spec[:2,:2], projection='3d')
            h_main.clear()
            h_main.autoscale(False)
            h_main.grid(True)
            h_main.set_xlim([-0.5, 2])
            h_main.set_ylim([-1, 2])
            h_main.set_zlim([-0.01, 0.9])
            viewPt = [15, 260, 0]
            h_main.view_init(elev=viewPt[0], azim=viewPt[1])
            #breakpoint()
            # Plot robot & GRF
            fig_plot_robot(X[ii, :].reshape(-1,1), U[ii, :].reshape(-1,1), Ue[ii, :].reshape(-1,1), h_main, p,ii) # TODO remember to take the transpose for each of the input

            # fig_plot_robot(X[ii, :], U[ii, :], Ue[ii, :], p)
            
            txt_time = f't = {ts[ii]:.2f} s'
            h_main.text(pcom[0], pcom[1], 0.3, txt_time)
            txt_vd = f'vd = {Xd[ii, 3]:.2f} m/s'
            h_main.text(pcom[0], pcom[1], 0.5, txt_vd)
            txt_v = f'v = {X[ii, 3]:.2f} m/s'
            h_main.text(pcom[0], pcom[1], 0.4, txt_v)

            # States - 
            # TODO: Position is missing
            # Angular Position
            h_x.cla()
            h_x.plot(ts[0:ii], theta[0:ii, 0], 'r', ts[0:ii], theta[0:ii, 1], 'g', ts[0:ii], theta[0:ii, 2], 'b',
                    ts[0:ii], thetad[0:ii, 0], 'r--', ts[0:ii], thetad[0:ii, 1], 'g--', ts[0:ii], thetad[0:ii, 2], 'b--', linewidth=1)
            h_x.set_xlim([ts[0], ts[-1]])
            h_x.legend(['r', 'p', 'y'], loc='upper right')
            h_x.set_title('Angular Position [rad]')

            # Velocity
            h_dx.cla()
            h_dx.plot(ts[0:ii], X[0:ii, 3], 'r', ts[0:ii], X[0:ii, 4], 'g', ts[0:ii], X[0:ii, 5], 'b',
                    ts[0:ii], Xd[0:ii, 3], 'r--', ts[0:ii], Xd[0:ii, 4], 'g--', ts[0:ii], Xd[0:ii, 5], 'b--', linewidth=1)
            h_dx.set_xlim([ts[0], ts[-1]])
            h_dx.legend(['x', 'y', 'z'], loc='upper right')
            h_dx.set_title('Velocity [m/s]')

            # Angular velocity
            h_w.cla()
            h_w.plot(ts[0:ii], X[0:ii, 15], 'r', ts[0:ii], X[0:ii, 16], 'g', ts[0:ii], X[0:ii, 17], 'b',
                    ts[0:ii], Xd[0:ii, 15], 'r--', ts[0:ii], Xd[0:ii, 16], 'g--', ts[0:ii], Xd[0:ii, 17], 'b--', linewidth=1)
            h_w.set_xlim([ts[0], ts[-1]])
            h_w.legend(['r', 'p', 'y'], loc='upper right')
            h_w.set_title('Angular velocity [rad/s]')

            # Control
            h_u.cla()
            h_u.plot(t2[0:2*ii], U2[0:2*ii, 2], 'r', t2[0:2*ii], U2[0:2*ii, 5], 'g', t2[0:2*ii], U2[0:2*ii, 8], 'b',
                    t2[0:2*ii], U2[0:2*ii, 11], 'k', ts[0:ii], Ud[0:ii, 2], 'r--', ts[0:ii], Ud[0:ii, 5], 'g--',
                    ts[0:ii], Ud[0:ii, 8], 'b--', ts[0:ii], Ud[0:ii, 11], 'k--', linewidth=1)
            h_u.set_xlim([ts[0], ts[-1]])
            h_u.legend(['$Fz_1$', '$Fz_2$', '$Fz_3$', '$Fz_4$'], loc='upper right')
            h_u.set_title('Fz [N]')

            # Make movie
            if flag_movie:
                plt.savefig(f'Figures_Python/frame_{ii}.png')  
            plt.pause(0.05)
        except KeyboardInterrupt:
            plt.close('all')
            break
    #plt.show()
        #plt.pause(0.01)  # Adjust the pause time as needed
    return ts, EA, EAd

def fig_plot_robot(Xt, Ut, Ue, h_main, p, ii):
    
    ''' 
    Plots the quadrupeds- Body, Legs, Feet, GRF, Ground (off)
    '''
    # parameters of the Single Rigid Model (SRB)
    L = p["L"]
    W = p["W"]
    h = p["h"]

    # Visual Settings
    body_color = p["body_color"]
    leg_color = p["leg_color"]
    ground_color = p["ground_color"]

    # unpack the STATE VECTOR in column vectors
    pcom = Xt[0:3].reshape((3, 1), order='F')
    #dpc = Xt[3:6].reshape((3, 1))
    R = Xt[6:15].reshape((3, 3), order='F')
    #wb = Xt[15:18].reshape((3, 1), order='F')
    pf34 = Xt[18:30].reshape((3, 4), order='F') # TODO: check order Fortran
    # Ground Reaction Forces (GRF)
    f34 = Ut.reshape((3, 4), order='F')
    # Forward kinematics
    # hips
    Twd2com = np.vstack([np.hstack([R, pcom]), 
                        np.array([0, 0, 0, 1])])
    Tcom2h1 = np.vstack([np.hstack([np.eye(3), np.array([L/2, W/2, 0]).reshape(-1,1)]), 
                         np.array([0, 0, 0, 1])])
    Tcom2h2 = np.vstack([np.hstack([np.eye(3), np.array([L/2, -W/2, 0]).reshape(-1,1)]), 
                        np.array([0, 0, 0, 1])])
    Tcom2h3 = np.vstack([np.hstack([np.eye(3), np.array([-L/2, W/2, 0]).reshape(-1,1)]), 
                        np.array([0, 0, 0, 1])])
    Tcom2h4 = np.vstack([np.hstack([np.eye(3), np.array([-L/2, -W/2, 0]).reshape(-1,1)]),
                         np.array([0, 0, 0, 1])])
    Twd2h1 = np.dot(Twd2com, Tcom2h1) # wd2h1 = world to hip1
    Twd2h2 = np.dot(Twd2com, Tcom2h2)
    Twd2h3 = np.dot(Twd2com, Tcom2h3)
    Twd2h4 = np.dot(Twd2com, Tcom2h4)

    p_h1_wd = Twd2h1[0:3, 3]
    p_h2_wd = Twd2h2[0:3, 3]
    p_h3_wd = Twd2h3[0:3, 3]
    p_h4_wd = Twd2h4[0:3, 3]

    # body offset up by h
    Tcom2h1_up = np.vstack([np.hstack([np.eye(3), np.array([L/2, W/2, h]).reshape(-1,1)]),
                            np.array([0, 0, 0, 1])])
    Tcom2h2_up = np.vstack([np.hstack([np.eye(3), np.array([L/2, -W/2, h]).reshape(-1,1)]),
                            np.array([0, 0, 0, 1])])
    Tcom2h3_up = np.vstack([np.hstack([np.eye(3), np.array([-L/2, W/2, h]).reshape(-1,1)]),
                            np.array([0, 0, 0, 1])])
    Tcom2h4_up = np.vstack([np.hstack([np.eye(3), np.array([-L/2, -W/2, h]).reshape(-1,1)]),
                            np.array([0, 0, 0, 1])])
    Twd2h1_up = np.dot(Twd2com, Tcom2h1_up)
    Twd2h2_up = np.dot(Twd2com, Tcom2h2_up)
    Twd2h3_up = np.dot(Twd2com, Tcom2h3_up)
    Twd2h4_up = np.dot(Twd2com, Tcom2h4_up)

    p_h1_up = Twd2h1_up[0:3,3]
    p_h2_up = Twd2h2_up[0:3,3]
    p_h3_up = Twd2h3_up[0:3,3]
    p_h4_up = Twd2h4_up[0:3,3]

    chain1 = np.column_stack([p_h1_wd, p_h2_wd, p_h4_wd, p_h3_wd])
    chain2 = np.column_stack([p_h1_wd, p_h2_wd, p_h2_up, p_h1_up])
    chain3 = np.column_stack([p_h1_wd, p_h3_wd, p_h3_up, p_h1_up])
    chain4 = np.column_stack([p_h3_wd, p_h4_wd, p_h4_up, p_h3_up])
    chain5 = np.column_stack([p_h4_wd, p_h2_wd, p_h2_up, p_h4_up])
    chain6 = np.column_stack([p_h1_up, p_h2_up, p_h4_up, p_h3_up])

    # Inverse Kinematics
    q = np.zeros(12)
    chain_leg = np.zeros((3, 4, 4))

    for i_leg in range(1, 5): # From 1 to 4 because we are working with a quadruped
        if i_leg == 1:
            p['sign_L'] = 1
            p['sign_d'] = 1
        elif i_leg == 2:
            p['sign_L'] = 1
            p['sign_d'] = -1
        elif i_leg == 3:
            p['sign_L'] = -1
            p['sign_d'] = 1
        elif i_leg == 4:
            p['sign_L'] = -1
            p['sign_d'] = -1

        q_idx = 3 * (i_leg - 1) + np.array([0, 1, 2])
        
        q[q_idx] = fcn_invKin3(Xt, pf34[:, i_leg-1], p,ii) #
        chain_leg[:, :, i_leg-1] = legKin(Twd2com, q[q_idx], p)

    # Extracting individual chain_leg matrices
    chain_leg1 = chain_leg[:, :, 0]
    chain_leg2 = chain_leg[:, :, 1]
    chain_leg3 = chain_leg[:, :, 2]
    chain_leg4 = chain_leg[:, :, 3]

    # Plots
    # Plot body
    ax = h_main
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    body_chain = [chain1, chain2, chain3, chain4, chain5, chain6]
    for chain in body_chain:
        ax.add_collection3d(Poly3DCollection([chain.T], color=body_color, edgecolor='k', linewidths=0.5))

    # Plot legs
    leg_chain = [chain_leg1, chain_leg2, chain_leg3, chain_leg4]
    for chain in leg_chain:
        ax.add_collection3d(Line3DCollection([chain.T], linewidths=1.5, color=leg_color))
    
    # Plot feet
    for i in range(4):
        ax.plot3D(pf34[0, i], pf34[1, i], pf34[2, i], 'o',  color=leg_color, markersize=2.5, markerfacecolor=leg_color)

    # Plot GRF
    scale = 1e-2
    for i_leg in range(4):
        chain_f = np.column_stack([pf34[:, i_leg], pf34[:, i_leg] + scale * f34[:, i_leg]])
        ax.plot3D(chain_f[0, :], chain_f[1, :], chain_f[2, :], 'r', linewidth=1.5)

    # Plot external force
    p_ext_R = np.dot(R, p['p_ext']) + pcom
    
    chain_Ue = np.column_stack([p_ext_R, p_ext_R + 0.01 * Ue])
    ax.plot3D(chain_Ue[0, :], chain_Ue[1, :], chain_Ue[2, :], 'c', linewidth=1.5)

    #ax.set_box_aspect([np.ptp(arr) for arr in [ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]])
    ax.set_aspect('equal')

    # Plot ground
    Rground = p['Rground']
    goffset = 3
    if p['gait'] != -2:
        chain0 = np.dot(Rground, goffset * np.array([[-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0]]).T) + np.tile(np.vstack([pcom[:2], [0]]), (1, 4))
        # TODO, something goes here.
        chain0 =  goffset * np.array([[-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0]]).T
        # ax.add_collection3d(Poly3DCollection([chain0.T], alpha=0.5, color=ground_color,edgecolor='k'))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

def legKin(Twd2com, q, p):
    '''
    Returns the position of the 4 legs as a column vector
    '''
    L = p['L']
    W = p['W']
    d = p['d']
    l1 = p['l1']
    l2 = p['l2']
    sign_L = p['sign_L']
    sign_d = p['sign_d']

    Tcom2h = np.block([[rx(q[0]), np.array([sign_L * L / 2, sign_d * W / 2, 0]).reshape(-1,1)],
                       [0, 0, 0, 1]])
    Th2s = np.block([[ry(q[1]), np.array([0, sign_d * d, 0]).reshape(-1,1)],
                       [0, 0, 0, 1]])
    Ts2k = np.block([[ry(q[2]), np.array([l1, 0, 0]).reshape(-1,1)],
                       [0, 0, 0, 1]])
    Tk2f = np.block([[np.eye(3), np.array([l2, 0, 0]).reshape(-1,1)],
                       [0, 0, 0, 1]])

    # Successive Homogeneous Transformations
    Twd2h = np.dot(Twd2com, Tcom2h) #world to hip
    Twd2s = np.dot(Twd2h, Th2s)     #world to shoulder
    Twd2k = np.dot(Twd2s, Ts2k)     #world to knee
    Twd2f = np.round(np.dot(Twd2k, Tk2f),decimals=5)     #world to foot

    #Extraction of the Position
    p_h_wd = Twd2h[0:3, 3]
    p_s_wd = Twd2s[0:3, 3]
    p_k_wd = Twd2k[0:3, 3]
    p_f_wd = Twd2f[0:3, 3]
    chain = np.column_stack((p_h_wd, p_s_wd, p_k_wd, p_f_wd))
    #breakpoint()
    return chain

def fig_plot(tout, Xout, Uout, Xdout, Udout, Uext, p):
    '''
    Organizes the data to be plotted (states)
    '''
    # Robot path
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot(Xout[:, 0], Xout[:, 1], Xout[:, 2])
    # ax.set_xlabel('x(ts)')
    # ax.set_ylabel('y(ts)')
    # ax.set_zlabel('z(ts)')
    # ax.grid(True)

    # Z direction foot forces
    X1 = tout
    YMatrix1 = np.column_stack((Uout[:, 2], Udout[:, 2]))
    YMatrix2 = np.column_stack((Uout[:, 5], Udout[:, 5]))
    YMatrix3 = np.column_stack((Uout[:, 8], Udout[:, 8]))
    YMatrix4 = np.column_stack((Uout[:, 11], Udout[:, 11]))

    # State plots
    theta = []
    thetad = []
    for i in range(len(Xout)):
        R = np.reshape(Xout[i, 6:15], (3, 3),order='F')
        Rd = np.reshape(Xdout[i, 6:15], (3, 3),order='F')
        theta.append(veeMap(logm(R)))  # veeMap(logm(R))'
        thetad.append(veeMap(logm(Rd)))  # veeMap(logm(Rd))'
    
    # List to numpy array
    theta = np.array(theta)
    thetad = np.array(thetad)

    YMatrix5 = np.column_stack((Xout[:, 0:2], Xdout[:, 0:2]))
    YMatrix6 = np.column_stack((Xout[:, 3:5], Xdout[:, 3:5]))
    YMatrix7 = np.column_stack((theta[:, 0:3], thetad[:, 2]))
    YMatrix8 = np.column_stack((Xout[:, 15:18], Xdout[:, 17]))

    # Plot 2 column figure
    createfigure(X1, YMatrix1, YMatrix2, YMatrix3, YMatrix4,
                 YMatrix5, YMatrix6, YMatrix7, YMatrix8)

def createfigure(X1, YMatrix1, YMatrix2, YMatrix3, YMatrix4, YMatrix5, YMatrix6, YMatrix7, YMatrix8):
    '''
    Creates all the handlers and sets up the properties for the plots (states)
    '''
    try:
        # Create subplots
        fig2, axs = plt.subplots(4, 2, figsize=(12, 16))
        font_name = 'cmr10'
        fig2.suptitle('MPC Experiment - State Plots')
        # fz plots (For each of the 4 legs)
        axs[0, 0].plot(X1, YMatrix1[:, 0], linewidth=2, label='MPC')
        axs[0, 0].plot(X1, YMatrix1[:, 1], linewidth=2, label='ideal')
        axs[0, 0].set_ylabel('$f_1$ (N)', fontweight='bold', fontsize=12, fontname=font_name)
        axs[0, 0].legend()
        axs[0, 0].grid(True)
        axs[0, 0].set_xlim([np.min(X1), np.max(X1)])
        axs[0, 0].set_ylim([0, 40])

        axs[1, 0].plot(X1, YMatrix2[:, 0], linewidth=2, label='MPC')
        axs[1, 0].plot(X1, YMatrix2[:, 1], linewidth=2, label='ideal')
        axs[1, 0].set_ylabel('$f_2$ (N)', fontweight='bold', fontsize=12, fontname=font_name)
        axs[1, 0].legend()
        axs[1, 0].grid(True)
        axs[1, 0].set_xlim([np.min(X1), np.max(X1)])
        axs[1, 0].set_ylim([0, 40])

        axs[2, 0].plot(X1, YMatrix3[:, 0], linewidth=2, label='MPC')
        axs[2, 0].plot(X1, YMatrix3[:, 1], linewidth=2, label='ideal')
        axs[2, 0].set_ylabel('$f_3$ (N)', fontweight='bold', fontsize=12, fontname=font_name)
        axs[2, 0].legend()
        axs[2, 0].grid(True)
        axs[2, 0].set_xlim([np.min(X1), np.max(X1)])
        axs[2, 0].set_ylim([0, 40])

        axs[3, 0].plot(X1, YMatrix4[:, 0], linewidth=2, label='MPC')
        axs[3, 0].plot(X1, YMatrix4[:, 1], linewidth=2, label='ideal')
        axs[3, 0].set_xlabel('time (s)', fontweight='bold', fontsize=12)
        axs[3, 0].set_ylabel('$f_4$ (N)', fontweight='bold', fontsize=12, fontname=font_name)
        axs[3, 0].legend()
        axs[3, 0].grid(True)
        axs[3, 0].set_xlim([np.min(X1), np.max(X1)])
        axs[3, 0].set_ylim([0, 40])

        # state plots
        # Position of the Center of Mass in X and Y
        axs[0, 1].plot(X1, YMatrix5[:, 0], linewidth=2, label=r'$p_x$')
        axs[0, 1].plot(X1, YMatrix5[:, 1], linewidth=2, label=r'$p_y$')
        axs[0, 1].plot(X1, YMatrix5[:, 2], linewidth=2, label=r'$p_{x,ref}$', linestyle='--', color='purple')
        axs[0, 1].plot(X1, YMatrix5[:, 3], linewidth=2, label=r'$p_{y,ref}$', linestyle='--', color='orange')
        axs[0, 1].set_ylabel('$\mathbf{p}$ (m)', fontweight='bold', fontsize=12, fontname=font_name)
        axs[0, 1].legend()
        axs[0, 1].grid(True)
        axs[0, 1].set_xlim([np.min(X1), np.max(X1)])
        axs[0, 1].set_ylim([-0.2, 0.7])

        # Velocity of the Center of Mass in X and Y
        axs[1, 1].plot(X1, YMatrix6[:, 0], linewidth=2, label=r'$v_x$')
        axs[1, 1].plot(X1, YMatrix6[:, 1], linewidth=2, label=r'$v_y$')
        axs[1, 1].plot(X1, YMatrix6[:, 2], linewidth=2, label=r'$v_{x,ref}$', linestyle='--', color='purple')
        axs[1, 1].plot(X1, YMatrix6[:, 3], linewidth=2, label=r'$v_{y,ref}$', linestyle='--', color='orange')
        axs[1, 1].set_ylabel('$\mathbf{v}$ (m/s)', fontweight='bold', fontsize=12, fontname=font_name)
        axs[1, 1].legend()
        axs[1, 1].grid(True)
        axs[1, 1].set_xlim([np.min(X1), np.max(X1)])
        axs[1, 1].set_ylim([-0.2, 0.8])

        # Euler angles of the SRB
        axs[2, 1].plot(X1, YMatrix7[:, 0], linewidth=2, label=r'$\phi$')
        axs[2, 1].plot(X1, YMatrix7[:, 1], linewidth=2, label=r'$\theta$')
        axs[2, 1].plot(X1, YMatrix7[:, 2], linewidth=2, label=r'$\psi$')
        axs[2, 1].plot(X1, YMatrix7[:, 3], linewidth=2, label=r'$\psi_{ref}$', linestyle='--', color='purple')
        axs[2, 1].set_ylabel('$\mathbf{\Theta}$ (rad)', fontweight='bold', fontsize=12, fontname=font_name)
        axs[2, 1].legend()
        axs[2, 1].grid(True)
        axs[2, 1].set_xlim([np.min(X1), np.max(X1)])
        axs[2, 1].set_ylim([-0.155368462626757, 1.1])



        # Rate of change of the Euler Angles of the SRB
        axs[3, 1].plot(X1, YMatrix8[:, 0], linewidth=2, label=r'$\omega_\phi$')
        axs[3, 1].plot(X1, YMatrix8[:, 1], linewidth=2, label=r'$\omega_\theta$')
        axs[3, 1].plot(X1, YMatrix8[:, 2], linewidth=2, label=r'$\omega_\psi$')
        axs[3, 1].plot(X1, YMatrix8[:, 3], linewidth=2, label=r'$\omega_{\phi,ref}$', linestyle='--', color='purple')
        axs[3, 1].set_xlabel('time (s)', fontweight='bold', fontsize=12, fontname=font_name)
        axs[3, 1].set_ylabel('$\mathbf{\omega}$ (m/s)', fontweight='bold', fontsize=12, fontname=font_name)
        axs[3, 1].legend()
        axs[3, 1].grid(True)
        axs[3, 1].set_xlim([np.min(X1), np.max(X1)])
        axs[3, 1].set_ylim([-0.155368462626757, 1.2])

        plt.tight_layout()
        fig2.subplots_adjust(top=0.95)
        plt.savefig('Figures_Python/state.png', dpi=300)
        plt.show()
    except  KeyboardInterrupt:
        plt.close('all')


def create_recording(ts,p):
    nt = len(ts)
    first_frame = cv2.imread('Figures_Python/frame_0.png')
    height, width, _ = first_frame.shape
    try:
        name = 'Figures_Python/test_python.mp4'
        vidfile = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*'mp4v'), 15, (width, height))  # Adjust parameters as needed
    except cv2.error as e:
        name = 'Figures_Python/test_python.avi'
        vidfile = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*'MJPG'), 15, (width, height))  # Adjust parameters as needed
    for ii in range(0, nt, p['playSpeed']):
        #breakpoint()
        img = cv2.imread(f'Figures_Python/frame_{ii}.png')
        sys.stdout.write('\rProcessing Video frame (' +  str(ii) + '/' + str(nt) + ')')
        sys.stdout.flush()
        if img is None:
            print(f"Error reading frame_{ii}.png")
            continue 
        vidfile.write(img)      
    vidfile.release()

