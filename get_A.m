function A_aug = get_A(Xd)
%% function to get avg A matrix over a horizon
% to-do: make A time variying
% Inputs
% R_psi (average R over the MPC horizon)
% Outputs
% A (system dynamics)
%% states = [p p_dot theta omega gravity]
zero_A = zeros(3,3);
one_A = eye(3,3);
R_psi = get_Rpsi(Xd);

% continuous time dynamics
A_psi = [zero_A,  zero_A,   zero_A,   one_A;...
         zero_A,  zero_A,   zero_A,   zero_A;...
         zero_A,  zero_A,   R_psi,    zero_A;...
         zero_A,  zero_A,   zero_A,   zero_A];

g_aug = [zeros(3,1); [0;0;1]; zeros(3,1); zeros(3,1)];
A_aug = [A_psi,                 g_aug;...
        zeros(1,size(A_psi,2)), 0];

end