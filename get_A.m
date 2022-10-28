function A = get_A(Xd)
% function to get avg A matrix over a horizon
% to-do: make A time variying
% Inputs
% Xd (desired states)
% Outputs
% A (system dynamics)

% rotation matrix R(psi)
Rd = reshape(Xd(7:15,:),[3,3*N]);
euler_angles = veeMap(Rd);
psi = mean(euler_angles(3,:));

% states = [p p_dot theta omega gravity]
zero_A = zeros(3,3);
one_A = eye(3,3);

A_psi = [zero_A,  zero_A,   zero_A,   one_A;...
         zero_A,  zero_A,   zero_A,   zero_A;...
         zero_A,  zero_A,   R_psi,    zero_A;...
         zero_A,  zero_A,   zero_A,   zero_A];

A_aug = [A_psi,                 zeros(size(A_psi,1),1);...
        zeros(1,size(A_psi,2)), 1];

dT = 0.1;  
A = A_aug.*dT + eye(size(A_aug));
end