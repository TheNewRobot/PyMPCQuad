function [F, G, A_ineq, b_ineq] = get_QP(A,B,C,N,Ud,idx,p)
    % Inputs
    % A (system dynamics)                       : n x n
    % B (control input)                         : n x p
    % C (system output)                         : n x m
    % N is prediction horizon                   : scalar
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

    %% define costs
    P = eye(size(A)); % terminal cost
    Q_i = eye(size(A)); % terminal cost
    R_i = eye(size(B,2)); % terminal cost
    
    %% Augmented cost
    [A_0,B_0,~] = get_ABC(Xt,idx,p,i);

    n = size(A_0,2); % state dimension
    p = size(B_0,2); % control dimension
    m = size(C,1); % output dimension 


    A_hat = A_0;
    A_hat2 = [];
    B_hat = zeros(n*N,p*N); % set max value for number of columns
    b = B_0;
    B_hat(1:n,1:p) = b;

    for i = 2:N
        [A_i,B_i,~] = get_ABC(Xt,idx,p,i);
        A_hat = [A_hat; A_i*A_hat(end-3:end,:)];
        A_hat2 = [A_hat2, A_hat(1:end-3,:)];
        b = [A_hat2*B_i, B_i];
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

    % get number of feet in contact
    num_feet_contact = size(stance_feet(:,all(stance_feet,1)),2);

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