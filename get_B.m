function B = get_B(Xt,idx,horizon_i,p)
% function to get time varying B matrix over a horizon
% Inputs
% Xt (current states)
% idx (index of feet in contact over horizon)
% params (dict of parameters)
% Outputs
% A (system dynamics)

%% system parameters
mass = p.mass;
J = p.J;       % inertia tensor in body frame {B}
g = 9.81;
dT = 0.1;      % discretization timestep

%% rotation matrix R(psi)
R_psi = [cos(psi),  sin(psi),   0;...
         -sin(psi), cos(psi),   0;...
         0,         0,          1];

%% inertia matrix in world frame
I_w = R_psi*J*R_psi';
I_inv = I_w^-1;

%% footstep locations r in body frame
pc = reshape(Xt(1:3),[3,1]);
pf34 = reshape(Xt(19:30),[3,4]);
r34 = pf34 - repmat(pc,[1,4]);
r1_hat = hatMap(r34(:,1));
r2_hat = hatMap(r34(:,2));
r3_hat = hatMap(r34(:,3));
r4_hat = hatMap(r34(:,4));

%% continuous time LTV system dynamics
% states = [p p_dot theta omega gravity]

% variable size B_psi
B_feet =  [I_inv*r1_hat,  I_inv*r2_hat,   I_inv*r3_hat ,  I_inv*r4_hat];
B_feet = B_feet.*kron(idx(:,horizon_i)',ones(size(B_feet)));
B_feet(:,all(~B_feet,1)) = []; % make zero columns as null

zero_B = zeros(size(B_feet));
num_feet_contact = length(nonzeros(idx(:,horizon_i)));
one_B = repmat(eye(3),1,num_feet_contact);
B_psi = [zero_B; zero_B; B_feet; one_B./mass];

B_aug = [B_psi; zeros(1,size(B_psi,2))];

%% discrete time LTV system dynamics
B = B_aug.*dT;

end


