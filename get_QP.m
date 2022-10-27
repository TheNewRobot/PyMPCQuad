function [F, G, A_ineq, b_ineq, b_ineq_x0] = get_QP(A,B,C,N,Ud,p)
    % Inputs
    % A (system dynamics)                       : n x n
    % B (control input)                         : n x p
    % C (system output)                         : n x m
    % N is prediction horizon                   : scalar
    % u_min is min the control bound            : scalar
    % u_max is max the control bound            : scalar
    % y_min is min the output bound             : scalar
    % y_max is min the output bound             : scalar
    % Q_i and R_i is assumed to be constant for i = 1 to N
    % Augments
    % Q_hat (augmented Q)                       : n(N) x n(N)
    % R_hat (augmented R)                       : p(N) x p(N)
    % A_hat (augmented A)                       : n(N) x n(N)
    % B_hat (augmented B)                       : n(N) x p(N)
    % M_hat (augmented M)                       : (2(p+m)+2m)(N) x n(N)
    % E_hat (augmented E)                       : (2(p+m)+2m)(N) x p(N)
    % b_hat (augmented E)                       : (2(p+m)+2m)(N) x 1(N)
    % Outputs
    % G (augmented cost)                        : p(N) x p(N)
    % F (augmented cost)                        : p(N) x 1
    % augmented cost                            : 1/2 * X^T * G * X + X^T * F * x0
    % A_ineq (augmented ineq constraint A)      : (p+m)(N) x (p+m)(N)
    % b_ineq (augmented ineq constraint b)      : 2(p+m)(N) x 1
    % b_ineq_x0 (augmented ineq constraint cx)  : 2(p+m)(N) x 1
    % augmented ineq constraint                 : A_ineq * X <= b_ineq + b_ineqx0 * x0

    %% define costs
    P = eye(size(A)); % terminal cost
    Q_i = eye(size(A)); % terminal cost
    R_i = eye(size(B,2)); % terminal cost

    n = size(A,2); % state dimension
    p = size(B,2); % control dimension
    m = size(C,1); % output dimension 
    
    %% Augmented cost
    A_hat = zeros(n*N,n);
    b = [];
    B_hat = zeros(n*N,p*N);

    for i = 1:N
        A_hat((i-1)*n+1:i*n,:) = A^i;
        b = [A^(i-1)*B, b];
        B_hat((i-1)*n+1:i*n,1:i*p) = b;
    end

    Q_hat = [];
    R_hat = [];
    for i = 1:N-1
        Q_hat = blkdiag(Q_hat, Q_i);
        R_hat = blkdiag(R_hat, R_i);
    end
    % add the cost for terminal states and control
    Q_hat = blkdiag(Q_hat,P);
    R_hat = blkdiag(R_hat,R_i);

    % Augmented cost: 1/2 * X^T * G * X + X^T * F * x0 
    G = 2*(R_hat + B_hat'*Q_hat*B_hat);
    F = 2*B_hat'*Q_hat*A_hat;
    
    %% Augmented inequality constraint

    % define friction constraints
    mu = p.mu;
    % set max values of fi_z
    Fzd = Ud([3 6 9 12],i_hor);
    fi_z_lb = 0.5 * Fzd;
    fi_z_ub = 1.5 * Fzd;
    
    Aineq = [1 0 -mu;-1 0 -mu;0 1 -mu;0 -1 -mu;0 0 1; 0 0 -1];
    B_ineq = [0; 0; 0; 0; 1; -1];
    A_ineq_i = kron(eye(4),A_ineq);
    B_ineq_i = [B_ineq; B_ineq; B_ineq; B_ineq];
end