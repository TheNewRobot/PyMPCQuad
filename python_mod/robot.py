import numpy as np
from .utils import hatMap

def  get_params(gait):
    '''
    Table II on the paper is implemented on code here
    '''

    p = {}
    p['predHorizon'] = 7
    p['simTimeStep'] = 1/250
    p['Tmpc'] = 4/100
    p['gait'] = gait
    p['Umax'] = 50
    p['decayRate'] = 1
    p['freq'] = 30
    p['Rground'] = np.eye(3)
    p['Qf'] = np.diag([1e5, 2e5, 3e5, 5e2, 1e3, 150, 1e3, 1e4, 800, 40, 40, 10])

     # ---- gait ----
    if gait == 1:  # 1 - bound
        p['Tst'] = 0.1 # Stance 
        p['Tsw'] = 0.18 # Swing time
        p['predHorizon'] = 7
        p['simTimeStep'] = 1/100
        p['Tmpc'] = 2/100
        p['decayRate'] = 1
        p['R'] = np.diag(np.tile([0.1, 0.1, 0.1], [4, 1]))
        p['Q'] = np.diag([5e4, 2e4, 1e6, 4e3, 5e2, 5e2, 1e4, 5e4, 1e3, 1e2, 5e2, 1e2])
        p['Qf'] = np.diag([2e5, 5e4, 5e6, 8e3, 5e2, 5e2, 1e4, 5e4, 5e3, 1e2, 1e2, 1e2])
    elif gait == 2:  # 2 - pacing
        p['Tst'] = 0.12
        p['Tsw'] = 0.12
        p['R'] = np.diag(np.tile([0.1, 0.2, 0.1], [4, 1]))
        p['Q'] = np.diag([5e3, 5e3, 9e4, 5e2, 5e2, 5e2, 7e3, 7e3, 7e3, 5e1, 5e1, 5e1])
    elif gait == 3:  # 3 - gallop
        p['Tst'] = 0.08
        p['Tsw'] = 0.2
        p['R'] = np.diag(np.tile([0.1, 0.2, 0.1], [4, 1]))
        p['Q'] = np.diag([3e3, 3e3, 4e6, 5e2, 1e3, 150, 1e4, 1e4, 800, 1e2, 5e1, 5e1])
    elif gait == 4:  # 4 - trot run
        p['Tst'] = 0.12
        p['Tsw'] = 0.2
        p['Tmpc'] = 3/100
        p['predHorizon'] = 6
        p['decayRate'] = 1
        p['R'] = np.diag(np.tile([0.1, 0.18, 0.08], [4, 1]))
        p['Q'] = np.diag([1e5, 1e5, 1e5, 1e3, 1e3, 1e3, 2e3, 1e4, 800, 100, 40, 10])
        p['Qf'] = np.diag([1e5, 1.5e5, 2e4, 1.5e3, 1e3, 100, 2e3, 2e3, 800, 100, 60, 10])
    elif gait == 5:  # 4 - crawl
        p['Tst'] = 0.3
        p['Tsw'] = 0.1
        p['R'] = np.diag(np.tile([0.1, 0.2, 0.1], [4, 1]))
        p['Q'] = np.diag([5e5, 5e5, 9e5, 5, 5, 5, 3e3, 3e3, 3e3, 3, 3, 3])
    else:  # 0 - trot
        p['predHorizon'] = 6
        p['simTimeStep'] = 1/100
        p['Tmpc'] = 8/100
        p['Tst'] = 0.3
        p['Tsw'] = 0.15
        p['R'] = np.diag(np.tile([0.1, 0.2, 0.1], [4, 1]))
        p['Q'] = np.diag([1e5, 2e5, 3e5, 5e2, 1e3, 1e3, 1e3, 1e4, 800, 40, 40, 10])
        p['Qf'] = p['Q']

    # Physical Parameters
    p['mass'] = 5.5
    p['J'] = np.diag([0.026, 0.112, 0.075]) # Matriz of inertia
    p['g'] = 9.81
    p['mu'] = 1     #Friction coeficient
    p['z0'] = 0.2   #Nominal height of the CoM with respect to the floor
    p['pf34'] = np.array([ [0.15, 0.094, 0],
                           [0.15, -0.094, 0], 
                           [-0.15, 0.094, 0], 
                           [-0.15, -0.094, 0]]).T

    p['L'] = 0.301  #Body length
    p['W'] = 0.088  #Body width
    p['d'] = 0.05 
    p['h'] = 0.05   #Body width
    p['l1'] = 0.14  #Link length
    p['l2'] = 0.14  #Link length

    # Swing phase
    p['Kp_sw'] = 300

    # Colors
    p['body_color'] = [129/255, 117/255, 230/255]  # grey
    p['leg_color'] = [56/255, 82/255, 156/255]  # black
    p['ground_color'] = [219/255, 219/255, 219/255]  # yellow

    p['p_dist'] = np.array([p['L']/2, p['W']/2, p['d']])
    #p['p_dist'] = np.array([0, 0, 0]) 

    return p

def fcn_invKin3(X, pf, p,ii):
    pf = np.reshape(pf, (len(pf), 1))  #Into a column vector

    # States
    pcom = X[0:3]
    R = np.reshape(X[6:15], (3, 3),order='F') #Into a rotation matrix

    # Parameters
    L = p['L']
    W = p['W']
    sign_L = p['sign_L']
    sign_d = p['sign_d']

    Twd2com = np.block([[R, pcom],
                       [0, 0, 0, 1]])
    Tcom2h = np.block([[np.eye(3), np.array([sign_L * L / 2, sign_d * W / 2, 0]).reshape(-1,1)],
                       [0, 0, 0, 1]])
    Twd2h = np.dot(Twd2com, Tcom2h)
    p_f_wd = pf[0:3]
    p_h_wd = Twd2h[0:3, 3].reshape(-1, 1)

    

    # h1 to f1 in world frame
    p_h2f_wd = p_f_wd - p_h_wd # TODO: Investigate why is -0.094 instead of +0.094

    # h1 to f1 in body frame
    p_h2f_b = np.dot(R.T, p_h2f_wd)

    # Inverse kinematics
    q = invKin(p_h2f_b, p)
   
    return q

def invKin(p_h2f_b, p):
    '''
    Inverse Kinematics of one leg
    '''
    l1 = p['l1']
    l2 = p['l2']
    d = p['d']
    sign_d = p['sign_d']

    if p_h2f_b.shape[0] == 1:
        p_h2f_b = p_h2f_b.T

    vp = p_h2f_b

    # q1
    vpyz = vp[1:3]
    ryz = np.linalg.norm(vpyz, 2)
    a = np.arcsin(vp[1,0] / ryz)
    b = np.arcsin(d / ryz)
    q1 = a - sign_d * b

    # q2
    r = np.linalg.norm(vp, 2)
    vd = sign_d * np.array([0, d * np.cos(q1), d * np.sin(q1)])
    rf = np.linalg.norm(vp.flatten() - vd, 2)
    vf = np.dot(rx(q1).T, (vp.flatten() - vd))
    
    if vf[2] <= 0:
        a = np.arccos(vf[0] / rf)
    else:
        if vf[0] >= 0:
            a = -np.arcsin(vf[2] / rf)
        else:
            a = np.pi + np.arcsin(vf[2] / rf)

    cosb = (l1**2 + rf**2 - l2**2) / (2 * l1 * rf)
    
    b = np.arccos(cosb)
    q2 = a + b

    # q3
    cosc = (l1**2 + l2**2 - rf**2) / (2 * l1 * l2)
    c = np.arccos(cosc)
    
    q3 = -(np.pi - c)

    q = [q1, q2, q3]
    return q

def rx(q):
    '''
    Rotation in x direction
    '''
    return np.array([[1,         0,          0],
                     [0, np.cos(q), -np.sin(q)],
                     [0, np.sin(q),  np.cos(q)]])

def ry(q):
    '''
    Rotation in y direction
    '''
    return np.array([[ np.cos(q),  0, np.sin(q)],
                     [         0,  1,         0],
                     [-np.sin(q),  0, np.cos(q)]])

 ## --- simulate without any external disturbances ---
def dynamics_SRB(t, Xt, Ut, Xd, U_ext, p):
    '''
    Formualtes the pdynamics of a quadruped as a Single Rigid Body
    '''
    mass = p['mass']
    J = p['J']
    g = 9.81
    # Decompose
    pc = np.reshape(Xt[0:3], [3, 1], order='F')
    dpc = np.reshape(Xt[3:6], [3,1], order='F')
    R = np.reshape( Xt[6:15],[3 ,3],order='F')
    wb =  np.reshape(Xt[15:18],[3,1], order='F')
    pf34 =  np.reshape( Xt[18:30],[3,4], order='F')
    pfd34 =  np.reshape(Xd[18:30,0],(3,4), order='F')
    
    # Foot positions
    r34 = pf34 - np.tile(pc, (1, 4))
    # GRF
    f34 = np.reshape(Ut,[3, 4], order='F')

    # Dynamics
    ddpc = 1/mass * (np.sum(f34, axis=1, keepdims=True).flatten() + U_ext) + np.array([0, 0, -g])
    
    dR = R @ hatMap(wb.flatten())
    tau_s = np.zeros((3, 1))
    for ii in range(4):
        
        tau_s += hatMap(r34[:, ii]) @ f34[:, ii].reshape((3, 1),order='F')
    
    tau_ext = hatMap(R @ p['p_ext']) @ U_ext
    tau_tot = np.sum(tau_s, axis=1, keepdims=True).flatten() + tau_ext
    dwb = np.linalg.inv(J) @ (R.T @ tau_tot - hatMap(wb.flatten()) @ (J @ wb.flatten()))
    dpf = p['Kp_sw'] * (pfd34.flatten(order='F') - pf34.flatten(order='F'))
    dXdt = np.concatenate([dpc.flatten(), ddpc, dR.flatten(order='F'), dwb, dpf])
    return dXdt