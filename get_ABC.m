function [A,B,C] = get_ABC(Xt,idx,p,i)

%% system parameters
mass = p.mass;
J = p.J;       % inertia tensor in body frame {B}
g = 9.81;
dT = 0.1;      % discretization timestep

%% rotation matrix R(psi)
Rt = reshape(Xt(7:15,1),[3,3]);
euler_angles = veeMap(Rt);
psi = euler_angles(3);
R_psi = [cos(psi),  sin(psi),   0;...
         -sin(psi), cos(psi),   0;...
         0,         0,          1];

%% inertia matrix in world frame
I_w = R_psi*J*R_psi';

%% footstep locations r in body frame
pc = reshape(Xt(1:3),[3,1]);
pf34 = reshape(Xt(19:30),[3,4]);
r34 = pf34 - repmat(pc,[1,4]);
r1_hat = hatMap(r34(:,1));
r2_hat = hatMap(r34(:,2));
r3_hat = hatMap(r34(:,3));
r4_hat = hatMap(r34(:,4));
r_hat = [r1_hat;r2_hat;r3_hat;r4_hat];

%% continuous time LTV system dynamics
% states = [p p_dot theta omega gravity]
zero_A = zeros(3,3);
one_A = eye(3,3);
I_inv = I_w^-1;

A_psi = [zero_A,  zero_A,   zero_A,   one_A;...
         zero_A,  zero_A,   zero_A,   zero_A;...
         zero_A,  zero_A,   R_psi,    zero_A;...
         zero_A,  zero_A,   zero_A,   zero_A];

% fixed size B_psi
% B_psi = [zero,          zero,           zero,           zero;...
%          zero,          zero,           zero,           zero;...
%          I_inv*r1_hat,  I_inv*r2_hat,   I_inv*r3_hat ,  I_inv*r4_hat;...
%          one./mass,     one./mass,      one./mass,      one./mass];

% variable size B_psi
B_feet =  [I_inv*r1_hat,  I_inv*r2_hat,   I_inv*r3_hat ,  I_inv*r4_hat];
B_feet = B_feet.*kron(idx(:,i)',ones(3,3));
B_feet(:,all(~B_feet,1)) = []; % make zero columns as null

zero_B = zeros(size(B_feet));
num_feet_contact = length(nonzeros(idx(:,i)));
one_B = repmat(eye(3),1,num_feet_contact);
B_psi = [zero_B; zero_B; B_feet; one_B./mass];

A_aug = [A_psi,                 zeros(size(A_psi,1),1);...
        zeros(1,size(A_psi,2)), 1];

B_aug = [B_psi; zeros(1,size(B_psi,2))];

%% discrete time LTV system dynamics
A = A_aug.*dT + eye(size(A_aug));
B = B_aug.*dT;
C = eye(size(A));

end


