function A_aug = get_A(Xd)
%% function to get avg A matrix over a horizon
% to-do: make A time variying
% Inputs
% R_psi (average R over the MPC horizon)
% Outputs
% A (system dynamics)
%% states = [p p_dot theta omega gravity], all states are in world frame
% note here MPC uses w in world frame while dynamics_SRB.m uses w in body
% frame

zero_A = zeros(3,3);
one_A = eye(3,3);
R_psi = get_Rpsi(Xd);

% refer to eth paper for gereral SRB
% https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8968251
% refer to stack exchnage for w body to w world
% https://physics.stackexchange.com/questions/73961/angular-velocity-expressed-via-euler-angles/74014#74014

% continuous time dynamics
A_psi = [zero_A,  zero_A,   zero_A,   one_A;...
         zero_A,  zero_A,   zero_A,   zero_A;...
         zero_A,  zero_A,   R_psi,    zero_A;...
         zero_A,  zero_A,   zero_A,   zero_A];

g_aug = [zeros(3,1); [0;0;1]; zeros(3,1); zeros(3,1)];
A_aug = [A_psi,                 g_aug;...
        zeros(1,size(A_psi,2)), 0];

end