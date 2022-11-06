function [zval] = sim_MPC(Xt,Xd,Ud,idx,N,params,single_shoot)
import casadi.*
dT = params.Tmpc;
grav = -9.81;
mu = params.mu;

%% get current and desired states = [p p_dot theta omega gravity]
% current states
xt = Xt(1:3,:); vt = Xt(4:6,:);
Rt = reshape(Xt(7:15,:),[3,3]);
thetat = veeMap(logm(Rt)); 
wt = Xt(16:18,:); % body frame
wt_world = Rt*wt;
X_cur = [xt;vt;thetat;wt_world;grav]; 

% desired states
xd = Xd(1:3,:); vd = Xd(4:6,:);
thetad = []; wd = Xd(16:18,:); % body frame
wd_world = [];
for k = 1:N
    Rd = reshape(Xd(7:15,k),[3,3]);
    thetad = [thetad, veeMap(logm(Rd))];   
    wd_world =  [wd_world, Rd*wd(:,k)];  
end
X_des = [xd;vd;thetad;wd_world;grav.*ones(1,N)];

%% get dynamics function
% discretize A matrix
n_states = 13;
A = get_A(Xd);
B = get_B(Xt,Xd,idx,1,params);
C = eye(size(A));
sys=ss(A,B,C,0);
[A, ~, C] = ssdata(c2d(sys,dT)); 

%% get augmented states and control for the horizon
% get time varying controls U_i for a horizon
n_contact_total = 0;
for k = 1:N
    n_contact_total = n_contact_total + 3*length(nonzeros(idx(:,k)));    
end
U_i = SX.sym('Ui',n_contact_total); % vector for N horizon

% build state matrix for N horizon
X = SX.sym('X',n_states,(N+1)); 
contact_idx = 0;
for k = 1:N
    % discretize B matrix (time varying)
    B_i = get_B(Xt,Xd,idx,k,params);
    sys=ss(A,B_i,C,0);
    [~, B_i, ~] = ssdata(c2d(sys,dT));    
    
    if(single_shoot)
        % build X as a function of U using dynamics
        X(:,1) = X_cur; % initial state
        num_feet_contact = length(nonzeros(idx(:,k)));
        n_contact = 3*num_feet_contact;
        X(:,k+1) = A*X(:,k) + B_i*U_i(contact_idx+1:contact_idx+n_contact);
        contact_idx = contact_idx + n_contact;
    end
end

%% get objective function and constraints
obj = 0; % Objective function
g = [];  % constraints function vector
args = struct; % constraint values
args.ubg = []; args.lbg = [];

% weights Q and R for obj
if(single_shoot)
    Qx = 1e6*eye(3); Qv = 1e6*eye(3);
    Qa = 1.5e6*eye(3); Qw = 1e6*eye(3);
    Q = blkdiag(Qx, Qv, Qa, Qw, 1e-5);
else
    Q = blkdiag(params.Q,1e-6);
end

contact_idx = 0;
for k=1:N
    % get x an u for each step in pred horizon
    X_i = X(:,k);
    num_feet_contact = length(nonzeros(idx(:,k)));
    n_contact = 3*num_feet_contact;
    u_i = U_i(contact_idx+1:contact_idx+n_contact);% u for n feets in contact
    contact_idx = contact_idx + n_contact;

    % get R matrix based on time varying B matrix
    B_i = get_B(Xt,Xd,idx,k,params);
    R_i = 1e-1*eye(size(B_i,2));
    
    % compute objective function
    obj = obj+(X_i-X_des(:,k))'*Q*(X_i-X_des(:,k)) + u_i'*R_i*u_i;
    
    % compute constraints function
    % set max values of fi_z
    Fzd = Ud([3 6 9 12],k);
    fi_z_lb = -1.5 * max(Fzd);
    fi_z_ub = 1.5 * max(Fzd);

    A_ineq_i = [-1  0 -mu;...
                 1  0 -mu;...
                 0  -1 -mu;...
                 0  1 -mu;...
                 0  0  -1;...
                 0  0  1];
    A_ineq_i = kron(eye(num_feet_contact),A_ineq_i);
    b_ineq_i = [0; 0; 0; 0; -fi_z_lb; fi_z_ub];

    if(single_shoot)
        % friction constraints g_i <= b_i
        g = [g; A_ineq_i*u_i];
        
        % form ineq values
        b_i = b_ineq_i;
        b_i_ub = repmat(b_i,num_feet_contact,1);
        args.lbg = -inf;
        args.ubg = [args.ubg; b_i_ub];

    else
        % initial condition constraint
        X_ini = X(:,1);
        g = [g; X_ini - X_cur]; 
        b_ini_i = zeros(size(X_ini));

        % dynamics constraint g_i == 0
        X_i_next = X(:,k+1);
        g = [g; X_i_next - (A*X_i + B_i*u_i)];
        b_eq_i = zeros(size(X_i));

        % friction constraints g_i <= b_i
        g = [g; A_ineq_i*u_i]; 

        % form ineq values
        b_i_lb = [b_ini_i; b_eq_i; repmat(-inf*ones(size(b_ineq_i)),num_feet_contact,1)];
        b_i_ub = [b_ini_i; b_eq_i; repmat(b_ineq_i,num_feet_contact,1)];
        args.lbg = [args.lbg; b_i_lb];  % lower bound of 0 <= A*fi
        args.ubg = [args.ubg; b_i_ub];  % upper bound of A*fi <= 0  
    end     

end

% set up NLP problem
if(single_shoot)
    optim_var = U_i;
else
    optim_var = [reshape(X,n_states*(N+1),1);U_i];
end

qp_prob = struct('f', obj, 'x', optim_var, 'g', g);

% set solver options 
% check qpoases user manual
% https://www.coin-or.org/qpOASES/doc/3.0/manual.pdf
opts = struct;
%opts.error_on_fail = false;
opts.printLevel = 'none';

% set up solver
solver = qpsol('solver', 'qpoases', qp_prob, opts);

% find optimal solution
sol = solver('lbg', args.lbg, 'ubg', args.ubg); 
%sol = solver('x0', args.x0, 'lbg', args.lbg, 'ubg', args.ubg);    

num_feet_contact = length(nonzeros(idx(:,1)));
zval = full(sol.x);
zval = zval(1:3*num_feet_contact);

end
