function R_psi = get_Rpsi(Xd)
%% function to get avg Rotation matrix over a horizon
% to-do: make R time variying
% Inputs
% Xd (desired states over the MPC horizon)
% Outputs
% R_psi (Avg rotation dynamics)
%% rotation matrix R(psi)
N = size(Xd,2);
euler_angles = [];
for i = 1:N
    Rd = reshape(Xd(7:15,i),[3,3]);
    euler_angles = [euler_angles, veeMap(logm(Rd))];
    psi = mean(euler_angles(3,:));
    R_psi = [cos(psi),      sin(psi),    0;...
            -sin(psi),     cos(psi),    0;...
            0,             0,           1];
end

