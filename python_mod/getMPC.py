import casadi as ca
import numpy as np
from scipy.linalg import logm, block_diag
from .utils import veeMap, hatMap
from scipy.signal import cont2discrete
import time

def get_QP(Xt, Xd, Ud, idx, N, params):
    # Define friction constraints
    

    mu = params['mu']
    g = -9.81
    dT = params['Tmpc']

    # Get system matrices
    A = get_A(Xd)
    B = get_B(Xt, Xd, idx, 1, params)
    C = np.eye(A.shape[0])
    sys = (A, B, C, 0)
    A, B, C, _, _ = cont2discrete(sys, dT)
    n = A.shape[1]

    # Get current and desired states
    xt = Xt[:3, :]
    vt = Xt[3:6, :]
    Rt = np.reshape(Xt[6:15, :], (3, 3),order='F')
    thetat = veeMap(logm(Rt)).reshape(-1,1)
    wt = Xt[15:18, :]
    wt_world = np.dot(Rt, wt)
    #breakpoint()
    X_cur = np.vstack((xt, vt, thetat, wt_world, g))

    xd = Xd[:3, :]
    vd = Xd[3:6, :]
    wd = Xd[15:18, :]
    wd_world = []
    thetad = []
    X_des = np.vstack((xd, vd))

    # Define weights for the cost function
    Qx = 1e6 * np.eye(3)
    Qv = 1e6 * np.eye(3)
    Qa = 1e6 * np.eye(3)
    Qw = 1e6 * np.eye(3)
    
    Q_i = np.block([[Qx, np.zeros((3, 10))],
                [np.zeros((3, 3)), Qv, np.zeros((3, 7))],
                [np.zeros((3, 6)), Qa, np.zeros((3, 4))],
                [np.zeros((3, 9)), Qw, np.zeros((3, 1))],
                [np.zeros((1, 12)), 1e-5 * np.eye(1)]])
    P = Q_i  # terminal cost

    # Build QP Matrices
    A_hat = np.zeros((n * N, n))
    B = np.empty((0, n))
    b = np.empty((0, 1))
    B_hat_list = []
    Q_hat = np.empty((0, 0))
    R_hat_list = []
    A_ineq_list = []
    b_ineq_list = []
    
    # Iterate rows to get A_hat and Q_hat
    thetad = np.zeros((3, 0))
    wd_world = np.zeros((3, 0))
    
    for i in range(1, N + 1):
        A_hat[(i - 1) * n:i * n, :] = np.linalg.matrix_power(A, i)
        
        Q_hat = np.block([[Q_hat, np.zeros((Q_hat.shape[0], n))],
                        [np.zeros((n, Q_hat.shape[1])), Q_i]])

        # Get euler angles for each horizon
        Rd = np.reshape(Xd[6:15, i - 1], [3, 3], order='F')
        thetad = np.hstack([thetad, np.reshape(veeMap(logm(Rd)), [3, -1], order='F')])
        wd_world = np.hstack([wd_world, np.dot(Rd, wd[:, i - 1]).reshape(-1,1)])
        
    Q_hat[-n:, -n:] = P
    X_des = np.vstack([X_des, thetad, wd_world, g * np.ones((1, N))]) 
    for i in range(1,N+1):
        # Augmented Cost
        B_i = get_B(Xt, Xd, idx, i-1, params)
        sys = (A, B_i, C, 0)
        _, B_i, _, _, _ = cont2discrete(sys, dT)

        a = np.vstack((np.zeros((i * n, n)), np.eye(n), A_hat[:-i * n, :]))
        B_hat_list.append(np.dot(a[n:, :], B_i))
        R_i = 1e1 * np.eye(B_i.shape[1])
        R_hat_list.append(R_i)

        num_feet_contact = np.count_nonzero(idx[:, i - 1])
        A_ineq_i = np.array([[-1, 0, -mu],
                             [1, 0, -mu],
                             [0, -1, -mu],
                             [0, 1, -mu],
                             [0, 0, -1],
                             [0, 0, 1]])
        A_ineq_i = np.kron(np.eye(num_feet_contact), A_ineq_i)
        A_ineq_list.append(A_ineq_i)

        Fzd = Ud[[2, 5, 8, 11], i - 1]
        fi_z_lb = 0 * np.max(Fzd)
        fi_z_ub = 1.5 * np.max(Fzd)
        b_ineq_i = np.array([0.0, 0.0, 0.0, 0.0, -fi_z_lb, fi_z_ub]).reshape(-1, 1)
        b_ineq_i = np.tile(b_ineq_i, (num_feet_contact, 1))
        b_ineq_list.append(b_ineq_i)
    B_hat = np.hstack(B_hat_list)
    R_hat = block_diag(*R_hat_list)
    A_ineq = block_diag(*A_ineq_list) 
    b_ineq = np.vstack(b_ineq_list)
    R_N = np.eye(B_i.shape[1])
    p = R_N.shape[1]
    # R_hat[-p:, -p:] = R_N

    # Augmented cost: 1/2 * U^T * G * U + U^T * F
    G = 2 * (R_hat + np.dot(B_hat.T, np.dot(Q_hat, B_hat)))
    y = np.reshape(X_des, (-1, 1),order='F')
    F = 2 * np.dot(B_hat.T, np.dot(Q_hat, np.dot(A_hat, X_cur) - y))
    return F, G, A_ineq, b_ineq

def get_B(Xt, Xd, idx, horizon_i, p):
    # Function to get time-varying B matrix over a horizon
    
    # Inputs
    # Xt: current states
    # Xd: desired states over the MPC horizon
    # idx: index of feet in contact over the horizon
    # horizon_i: current time index in the horizon
    # p: dictionary of parameters
    
    # Outputs
    # B_aug: augmented system dynamics matrix

    # System parameters
    mass = p['mass']
    J = p['J']  # inertia tensor in body frame {B}

    # Rotation matrix R(psi)
    R_psi = get_Rpsi(Xd)

    # Inertia matrix in world frame
    I_w = np.dot(R_psi, np.dot(J, R_psi.T))
    I_inv = np.linalg.inv(I_w)

    # Footstep locations r in body frame
    pc = np.reshape(Xt[0:3], (3, 1), order='F')
    pf34 = np.reshape(Xt[18:30], (3, 4), order='F')
    r34 = pf34 - np.tile(pc, (1, 4))
    r1_hat = hatMap(r34[:, 0])
    r2_hat = hatMap(r34[:, 1])
    r3_hat = hatMap(r34[:, 2])
    r4_hat = hatMap(r34[:, 3])

    num_feet_contact = np.sum(idx[:, horizon_i] != 0)

    # Continuous time LTV system dynamics B of varying size
    B_feet = np.hstack([np.dot(I_inv, r1_hat),
                        np.dot(I_inv, r2_hat),
                        np.dot(I_inv, r3_hat),
                        np.dot(I_inv, r4_hat)])
    
    rows, __ = B_feet.shape

    # Map each value in idx size(4, 1) to B_feet size(3, 12)
    # Maps each contact feet to corresponding B matrix column entries
    B_feet = B_feet * np.kron(idx[:, horizon_i].T, np.ones((rows, rows)))

    # Remove all zero columns
    B_feet = B_feet[:, ~np.all(B_feet == 0, axis=0)]

    zero_B = np.zeros_like(B_feet)
    one_B = np.tile(np.eye(3), (1, num_feet_contact))
    B_psi = np.vstack([zero_B, one_B / mass, zero_B, B_feet])

    B_aug = np.vstack([B_psi, np.zeros((1, B_psi.shape[1]))])
    return B_aug


def get_A(Xd):
    # Function to get avg A matrix over a horizon
    
    # Inputs
    # Xd: state vector
    
    # Outputs
    # A_aug: augmented system dynamics matrix
    
    # states = [p, p_dot, theta, omega, gravity], all states are in world frame
    # note here MPC uses w in world frame while dynamics_SRB.m uses w in body frame

    zero_A = np.zeros((3, 3))
    one_A = np.eye(3)
    R_psi = get_Rpsi(Xd)  # Assuming get_Rpsi is implemented elsewhere

    # Continuous time dynamics
    A_psi = np.block([[zero_A, zero_A, zero_A, one_A],
                    [zero_A, zero_A, zero_A, zero_A],
                    [zero_A, zero_A, R_psi, zero_A],
                    [zero_A, zero_A, zero_A, zero_A]])
    g_aug = np.vstack([np.zeros((3, 1)), np.array([0, 0, 1]).reshape(-1,1), np.zeros((3, 1)), np.zeros((3, 1))])
    A_aug = np.block([[A_psi, g_aug],
                    [np.zeros((1, A_psi.shape[1])), 0]])

    return A_aug


def get_Rpsi(Xd):
    # Function to get average Rotation matrix over a horizon
    
    # Inputs
    # Xd: desired states over the MPC horizon
    
    # Outputs
    # R_psi: Average rotation matrix
    
    # Rotation matrix R(psi)
    N = Xd.shape[1]
    euler_angles = np.zeros((3, 0))
    for i in range(N):
        Rd = np.reshape(Xd[6:15, i], (3, 3), order='F')
        euler_angles = np.column_stack((euler_angles, veeMap(logm(Rd))))
        psi = np.mean(euler_angles[2, :])
        R_psi = np.array([[np.cos(psi), np.sin(psi), 0],
                        [-np.sin(psi), np.cos(psi), 0],
                        [0, 0, 1]])
    return R_psi


def sim_MPC(Xt, Xd, Ud, idx, N, params, single_shoot):
    dT = params['Tmpc']
    grav = -9.81
    mu = params['mu']

    # Get current and desired states
    xt = Xt[0:3, :]
    vt = Xt[3:6, :]
    Rt = np.reshape(Xt[6:15, :], [3, 3], order='F')
    thetat = np.reshape(np.asarray(veeMap(logm(Rt))), [1, -1])
    wt = Xt[15:18, :]
    wt_world = Rt.dot(wt)
    X_cur = np.vstack([xt, vt, thetat, wt_world, grav])

    xd = Xd[0:3, :]
    vd = Xd[3:6, :]
    thetad = np.zeros((3, 0))
    wd = Xd[15:18, :]
    wd_world = np.zeros((3, 0))
    for k in range(N):
        Rd = np.reshape(Xd[6:15, k], [3, 3], order='F')
        thetad = np.hstack([thetad, np.reshape(np.asarray(veeMap(logm(Rd))), [1, -1], order='F')])
        wd_world = np.hstack([wd_world, Rd.dot(wd[:, k])])
    X_des = np.vstack([xd, vd, thetad, wd_world, np.full((1, N), grav)])

    # Get dynamics function
    A = get_A(Xd)
    B = get_B(Xt, Xd, idx, 1, params)
    C = np.eye(A.shape[0])
    sys = ca.ss(A, B, C, 0)
    A, _, _ = ca.ssdata(ca.c2d(sys, dT))

    # Get augmented states and control for the horizon
    n_states = 13
    n_contact_total = 0
    for k in range(N):
        n_contact_total += 3 * len(np.nonzero(idx[:, k])[0])
    U_i = ca.SX.sym('Ui', n_contact_total)

    X = ca.SX.sym('X', n_states, (N + 1))
    contact_idx = 0
    for k in range(N):
        B_i = get_B(Xt, Xd, idx, k, params)
        sys = ca.ss(A, B_i, C, 0)
        _, B_i, _ = ca.ssdata(ca.c2d(sys, dT))

        if single_shoot:
            X[:, 0] = X_cur
            num_feet_contact = len(np.nonzero(idx[:, k])[0])
            n_contact = 3 * num_feet_contact
            X[:, k + 1] = A @ X[:, k] + B_i @ U_i[contact_idx:contact_idx + n_contact]
            contact_idx += n_contact

    # Get objective function and constraints
    obj = 0
    g = np.array([])
    args = {'ubg': np.array([]), 'lbg': np.array([])}

    # TODO: Define Q and R matrices
    Q = np.eye(n_states)
    R = 1e-1 * np.eye(n_contact_total)

    contact_idx = 0
    for k in range(N):
        X_i = X[:, k]
        num_feet_contact = len(np.nonzero(idx[:, k])[0])
        n_contact = 3 * num_feet_contact
        u_i = U_i[contact_idx:contact_idx + n_contact]
        contact_idx += n_contact

        B_i = get_B(Xt, Xd, idx, k, params)
        R_i = 1e-1 * np.eye(B_i.shape[1])

        obj += ca.mtimes([(X_i - X_des[:, k]).T, Q, X_i - X_des[:, k]]) + ca.mtimes([u_i.T, R_i, u_i])

        Fzd = Ud[np.array([2, 5, 8, 11]), k]
        fi_z_lb = -1.5 * max(Fzd)
        fi_z_ub = 1.5 * max(Fzd)

        A_ineq_i = np.array([[-1, 0, -mu],
                             [1, 0, -mu],
                             [0, -1, -mu],
                             [0, 1, -mu],
                             [0, 0, -1],
                             [0, 0, 1]])
        A_ineq_i = np.kron(np.eye(num_feet_contact), A_ineq_i)
        b_ineq_i = np.array([0, 0, 0, 0, -fi_z_lb, fi_z_ub])

        if single_shoot:
            g = np.concatenate([g, np.array(ca.vertcat(*A_ineq_i @ u_i))])
            b_i = b_ineq_i
            b_i_ub = np.tile(b_i, num_feet_contact)
            args['lbg'] = np.array([-np.inf])
            args['ubg'] = np.concatenate([args['ubg'], b_i_ub])
        else:
            X_ini = X[:, 0]
            g = np.concatenate([g, X_ini - X_cur])
            b_ini_i = np.zeros(X_ini.shape)

            X_i_next = X[:, k + 1]
            g = np.concatenate([g, X_i_next - (A @ X_i + B_i @ u_i)])
            b_eq_i = np.zeros(X_i.shape)

            g = np.concatenate([g, A_ineq_i @ u_i])

            b_i_lb = np.concatenate([b_ini_i, b_eq_i, np.tile(-np.inf * np.ones(b_ineq_i.shape), num_feet_contact)])
            b_i_ub = np.concatenate([b_ini_i, b_eq_i, np.tile(b_ineq_i, num_feet_contact)])
            args['lbg'] = np.concatenate([args['lbg'], b_i_lb])
            args['ubg'] = np.concatenate([args['ubg'], b_i_ub])

    if single_shoot:
        optim_var = U_i
    else:
        optim_var = ca.vertcat(ca.reshape(X, n_states * (N + 1), 1), U_i)

    qp_prob = {'f': obj, 'x': optim_var, 'g': g}
    opts = {'printLevel': 'none'}

    # TODO: Define qpsol solver
    solver = ca.qpsol('solver', 'qpoases', qp_prob, opts)

    # Find optimal solution
    sol = solver(lbg=args['lbg'], ubg=args['ubg'])
    num_feet_contact = len(np.nonzero(idx[:, 0])[0])
    zval = np.full(sol['x'].shape, np.nan)
    zval[:3 * num_feet_contact] = sol['x'][:3 * num_feet_contact]
    return zval
