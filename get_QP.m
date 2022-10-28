function [F, G, A_ineq, b_ineq] = get_QP(Xd,Ud,idx,N,params)
% Inputs
% Xd (desired states)
% Ud (desired foot forces)
% idx (index of feet in contact over horizon)
% N is prediction horizon                   : scalar
% params (dict of parameters)

% Augments
% Q_hat (augmented Q)                       : n(N) x n(N)
% R_hat (augmented R)                       : p(N) x p(N)
% A_hat (augmented A)                       : n(N) x n(N)
% B_hat (augmented B)                       : n(N) x p(N)

% Outputs
% G (augmented cost)                        : p(N) x p(N)
% F (augmented cost)                        : p(N) x 1
% augmented cost                            : 1/2 * X^T * G * X + X^T * F * x0
% A_ineq (augmented ineq constraint A)      : (p+m)(N) x (p+m)(N)
% b_ineq (augmented ineq constraint b)      : 2(p+m)(N) x 1
% augmented ineq constraint                 : A_ineq * X <= b_ineq

   %% get system matrices
    A = get_A(Xd);
    horizon_i = 1;
    B_0 = get_B(Xt,idx,horizon_i,params);
    C = eye(size(A));

    n = size(A,2); % state dimension columns (n x n)
    p = size(B_0,2); % control dimension columns (n x p)
    m = size(C,1); % output dimension (m x n)
    
    %% define costs
    P = eye(size(A)); % terminal cost
    Q_i = eye(size(A)); % terminal cost
    R_i = eye(size(B,2)); % terminal cost
    
    %% Augmented cost
    A_hat = zeros(n*N,n);
    B = []; b = [];
    B_hat = []; % set max value for number of columns
    
    for i = 1:N % iterate rows to get A_hat
        A_hat((i-1)*n+1:i*n,:) = A^i; 
    end

    for i = 1:N % iterate columns to get B_hat
        B_i = get_B(Xt,idx,i,params);
        a = [zeros(i*n,n); eye(n); A_hat(1:end-n,:)];
        B_hat = [B_hat; a*B_i];
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

    % get number of feet in contact
    num_feet_contact = length(nonzeros(idx(:,horizon_i)));

    % define friction constraints
    mu = p.mu;
    % set max values of fi_z
    Fzd = Ud([3 6 9 12],i_hor);
    fi_z_lb = 0.5 * Fzd;
    fi_z_ub = 1.5 * Fzd;
    
    Aineq = [1 0 -mu;-1 0 -mu;0 1 -mu;0 -1 -mu;0 0 1; 0 0 -1];
    B_ineq = [0; 0; 0; 0; 1; -1];
    A_ineq_i = kron(eye(num_feet_contact),A_ineq);
    B_ineq_i = repmat(B_ineq,1,num_feet_contact);

    
end