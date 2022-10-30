function [F, G, A_ineq, b_ineq, A_eq, b_eq] = get_QP_fixed(Xt,Xd,Ud,stance_feet,N,params)
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
    % define friction constraints
    mu = params.mu;
    g = -9.81;
    dT = params.Tmpc;
    
    A = get_A(Xd);
    B = get_B_fixed(Xt,Xd,params);
    C = eye(size(A));
    sys=ss(A,B,C,0);
    [A, B, C] = ssdata(c2d(sys,dT)); %discretize A

    n = size(A,2); % state dimension
    p = size(B,2); % control dimension
    m = size(C,1); % output dimension

    %% get current and desired states
    % states = [p p_dot theta omega gravity]
    xt = Xt(1:3,:); vt = Xt(4:6,:);
    Rt = reshape(Xt(7:15,:),[3,3]);
    thetat = veeMap(logm(Rt)); wt = Xt(16:18,:); 
    X_cur = [xt;vt;thetat;wt;g]; 

    xd = Xd(1:3,:); vd = Xd(4:6,:);
    thetad = []; wd = Xd(16:18,:); 
    % get euler angles for each horizon
    for i = 1:N    
        Rd = reshape(Xd(7:15,i),[3,3]);
        thetad = [thetad, veeMap(logm(Rd))];      
    end
    X_des = [xd;vd;thetad;wd;g.*ones(1,N)];

    %% define costs 
    Qx = 1e1*eye(3);
    Qv = 1e1*eye(3);
    Qa = 1.5e1*eye(3);
    Qw = 1e1*eye(3);
    Q_i = blkdiag(Qx, Qv, Qa, Qw, 1e-5);
    P = Q_i; % terminal cost
    %Q_i = blkdiag(params.Q,1);
    %P = Q_i;
    %Q_i = 1e1*eye(size(A));
    %P = 1e2*eye(size(A));
    R_i = 1e1*eye(p);
    
    %% Build QP Matrices
    A_hat = zeros(n*N,n);
    b = [];
    B_hat = zeros(n*N,p*N);

    for i = 1:N
        A_hat((i-1)*n+1:i*n,:) = A^i;
        b = [A^(i-1)*B, b];
        B_hat((i-1)*n+1:i*n,1:i*p) = b;
    end
    
    % Augmented cost
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
    y = reshape(X_des,[],1);
    F = 2*B_hat'*Q_hat*(A_hat*X_cur-y);
        
    %% Augmented constraints
    A_ineq = []; b_ineq = [];
    A_eq = []; b_eq = [];

    % inequality constraints
    A_ineq_i = [-1  0 -mu;...
                 1  0 -mu;...
                 0  -1 -mu;...
                 0  1 -mu;...
                 0  0  -1;...
                 0  0  1];
    A_ineq_i = kron(eye(4),A_ineq_i);  
     
    % set max values of fi_z
    Fzd = Ud([3 6 9 12],i);
    fi_z_lb = -1 * max(Fzd);
    fi_z_ub = 2 * max(Fzd);
    b_ineq_i = [0; 0; 0; 0; -fi_z_lb; fi_z_ub];
    b_ineq_i = repmat(b_ineq_i,4,1);

    % for N horizon
    for i = 1:N
        % build A_ineq * U <= b_ineq and  
        A_ineq = blkdiag(A_ineq, A_ineq_i);
        b_ineq = [b_ineq; b_ineq_i];

        % equality constraints
        % get 0 or 1 for flight or stance of 12 control inputs for N = i
        flight_idx = reshape(~stance_feet(:,(i-1)*4+1:i*4),[],1);
        A_eq = blkdiag(A_eq,flight_idx.*eye(12));
        b_eq = [b_eq; 1e-12*ones(12,1)];
    end

    % debugging
    %idx
    %num_feet_contact
    %size(B_hat)
    %X_cur
    %size(F)

end    
    
    