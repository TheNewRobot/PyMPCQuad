function B_aug = get_B_fixed(Xt,Xd,p)
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
%% rotation matrix R(psi)
R_psi = get_Rpsi(Xd);

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

%% continuous time LTV system dynamics B of varying size
% states = [p p_dot theta omega gravity]

B_feet =  [I_inv*r1_hat,  I_inv*r2_hat,   I_inv*r3_hat ,  I_inv*r4_hat];
zero_B = zeros(size(B_feet));
one_B = repmat(eye(3),1,4);

B_psi = [zero_B; one_B./mass; zero_B; B_feet];

B_aug = [B_psi; zeros(1,size(B_psi,2))];

end


