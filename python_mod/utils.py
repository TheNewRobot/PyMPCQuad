import numpy as np
from scipy.linalg import logm
from math import factorial

def fcn_X2EA(X):
    '''
        Its grabbing the rotation matrix from the state vector X[6:15] (as defined in the paper) turning it into a matrix and 
        performing the vee map operator
    '''
    vR = X[6:15] 
    R = np.reshape(vR, (3, 3), order='F')  
    EA = veeMap(logm(R))
    return EA

def veeMap(mat): 
    '''
        Performs the vee map operator, very popular in computer vision and robotics. Given a 3x3 skew-symmetric matrix, the vee map produces a 3-element vector. 
        In the context of rotation matrices, the vee map is often used to convert a rotation matrix to its corresponding axis-angle representation.
    '''
    out = np.zeros(3)
    out[0] = -mat[1, 2]
    out[1] = mat[0, 2]
    out[2] = -mat[0, 1]
    return out

def hatMap(v):
    '''
    Mathematical operation that transforms a three-dimensional vector into a skew-symmetric matrix. 
    '''
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def bz_int(alpha, x0, s_max=1):
    """
    Performs B-spline integration.

    Parameters:
    - alpha: B-spline coefficients (row vector)
    - x0: Initial condition
    - s_max: Integration interval (default is 1)

    Returns:
    - alpha_int: Integrated B-spline coefficients
    """
#     n, m = alpha.shape if alpha.ndim == 2 else (1, len(alpha))

# # Make sure alpha is a row vector
#     if n > m:
#         alpha = alpha.T
    M = len(alpha)
    AA = np.zeros((M + 1, M + 1))

    for ii in range(M):
        AA[ii, ii:ii+2] = [-1, 1]

    AA = M / s_max * AA
    AA[M, 0] = 1

    # Reshape x0 to a column vector
    x0 = np.asarray([x0])
   
    # Stack alpha and x0 horizontally and solve for alpha_int
    alpha_int = np.linalg.solve(AA, np.hstack((alpha, x0)))
    return alpha_int

def polyval_bz(alpha, s):
    '''
    Function to evaluate Bezeir polynomials
    '''
    alpha = alpha.flatten()
    b = np.zeros_like(s)
    M = alpha.size - 1

    # Define coefficients for binomial expansion
    binomial_coefficients = {
        2: [1.0, 2.0, 1.0],
        3: [1.0, 3.0, 3.0, 1.0],
        4: [1.0, 4.0, 6.0, 4.0, 1.0],
        5: [1.0, 5.0, 10.0, 10.0, 5.0, 1.0],
        6: [1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0],
        7: [1.0, 7.0, 21.0, 35.0, 35.0, 21.0, 7.0, 1.0],
        8: [1.0, 8.0, 28.0, 56.0, 70.0, 56.0, 28.0, 8.0, 1.0],
        9: [1.0, 9.0, 36.0, 84.0, 126.0, 126.0, 84.0, 36.0, 9.0, 1.0],
        10: [1.0, 10.0, 45.0, 120.0, 210.0, 252.0, 210.0, 120.0, 45.0, 10.0, 1.0],
    }

    c = binomial_coefficients.get(M, [0])


    for k in range(M + 1):
        binomial_coeff = factorial(M) // (factorial(k) * factorial(M - k))
        b += alpha[k] * binomial_coeff * s**k * (1 - s)**(M - k)
    return b

def print_progress_bar(iteration, total, prefix='', suffix='', length=30, fill='█'):
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r', flush=True)
    # Print a new line when the progress is complete
    if iteration == total:
        print()