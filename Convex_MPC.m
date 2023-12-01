% This code is used to develop a convex MPC for
% quadruped locomotion

% Reference code used from preprint available at: https://arxiv.org/abs/2012.10002
% video available at: https://www.youtube.com/watch?v=iMacEwQisoQ&t=101s

% Author: Yanran Ding
% Modified by: Sriram SKS Narayanan

%% initialization
clear all;close all;clc
addpath FSM getMPC plots robot trajGen utils refGen
import casadi.*

saveMAT  = true;
%Python Initialization
currentDir = pwd;
% Add the directory containing the Python script to the Python search path
py.sys.path().append(currentDir);

% Specify subfolders to append
subfolders = {'python_mod'};

% Append each subfolder to the Python search path
for i = 1:length(subfolders)
    subfolderPath = fullfile(currentDir, subfolders{i});
    py.sys.path().append(subfolderPath);
end

% Import the sys module to show the current path
sys = py.importlib.import_module('sys');
disp('Python sys.path after appending the current directory:');
disp(char(sys.path));

% Import the Python module
pyModule = py.importlib.import_module('my_test');

%% Test
clc
% % Cal the square function from the Python module
x = 5;  % Example input
result = pyModule.square(x);

%% --- parameters ---
% ---- gait ----
% 0-trot; 
gait = 0;
p = get_params(gait);
p.playSpeed = 10;
p.flag_movie = 1;      % 1 - make movie
use_Casadi = 0;        % 0 - build QP matrices, 1 -casadi with qpoases 

% MPC parameters
p.predHorizon = 15;
N = p.predHorizon;
p.Tmpc = 1/50;
p.simTimeStep = 1/200;
dt_sim = p.simTimeStep;

% simulation timec
SimTimeDuration = 3;  % [sec]
MAX_ITER = floor(SimTimeDuration/p.simTimeStep);

if(gait>=0)
    % desired trajectory for trot turn
    p.yaw_d = pi/4;
    p.acc_d = 0.5;
    p.vel_d = [0.5*cos(p.yaw_d);0.5*sin(p.yaw_d)];
    
    % desired trajectory for trot walk
    % p.yaw_d = 0;
    % p.acc_d = 0.5;
    % p.vel_d = [0.5;0];
    
    p.wb_d = [0;0;0];
    p.ang_acc_d = [0;0;0.5];
end

if(gait==-1)
     p.roll_d = 0;
     p.pitch_d = pi/4;
     p.yaw_d = 0;
     p.ang_acc_d = [0;0.5;0];
end

%% Model Predictive Control
% --- initial condition ---
% Xt = [pc dpc vR wb pf]': [30,1]
if gait == 1
    [p,Xt,Ut] = fcn_bound_ref_traj(p);
else
    [Xt,Ut] = fcn_gen_XdUd(0,[],[1;1;1;1],p);
end

% --- logging ---
tstart = 0;
tend = dt_sim;

[tout,Xout,Uout,Xdout,Udout,Uext,FSMout] = deal([]);

% --- simulation ----
h_waitbar = waitbar(0,'Calculating...');
tic
Ut_ref = []; Ud_ref = [];
Selection = [];

for ii = 1:MAX_ITER
    fprintf('Iteration number: %i \n',ii);
    % --- time vector ---
    t_ = dt_sim * (ii-1) + p.Tmpc * (0:p.predHorizon-1);
    
    % --- FSM ---
    if gait == 1
        [FSM,Xd,Ud,Xt] = fcn_FSM_bound(t_,Xt,p);
    else
        [FSM,Xd,Ud,Xt] = fcn_FSM(t_,Xt,p,ii);
        
    end

    % set up selection matrix (makes problem infeasible)
    % S = 0 if feet not in contant
    S = Ud;
    S(S~=0)=1;
    Selection = [Selection, S];

    % current stance feet
    idx = any(reshape(S,3,4*N));
    stance_feet = ones(3,4*N).*idx;

    % future idx for feet in contact for horizon 
    idx = reshape(idx,4,p.predHorizon);
    num_feet_contact = length(nonzeros(idx(:,1)));

    %% --- MPC ----
   
    if use_Casadi
        % solve QP using qpoases through casadi
        % 1 - single shooting, 0 - multi shooting 
        single_shoot = 1;
        [zval] = sim_MPC(Xt,Xd,Ud,idx,N,p,single_shoot);       
    else
        %form QP using explicit matrices
        [f, G, A, b] = get_QP(Xt,Xd,Ud,idx,N,p);

        % solve QP using quadprog     
        zval = quadprog(G,f,A,b,[],[],[],[]);
    end

    % get the foot forces value for first time step and repeat
    contacts = 1;
    contact_seq = idx(:,1);
    for i=1:4
        if(contact_seq(i)==1)
            Ut((i-1)*3+1:i*3) = zval((contacts-1)*3+1:contacts*3).*[1;1;1];
            contacts = contacts + 1;
        else
            Ut((i-1)*3+1:i*3) = [0;0;0];
        end
    end
   
    % logging
    i_hor = 1;
    Ut_ref = [Ut_ref, Ut];
    Ud_ref = [Ud_ref, Ud(:,i_hor)];     
    %% --- external disturbance ---
    [u_ext,p_ext] = fcn_get_disturbance(tstart,p);
    p.p_ext = p_ext;        % position of external force
    u_ext = 0*u_ext;
    %% --- simulate without any external disturbances ---
    [t,X] = ode45(@(t,X)dynamics_SRB(t,X,Ut,Xd,u_ext,p),[tstart,tend],Xt);
    % 61 times is called
    %% --- update ---
    Xt = X(end,:)';
    tstart = tend;
    tend = tstart + dt_sim;
    %% --- log ---  
    lent = length(t(2:end));
    tout = [tout;t(2:end)];
    Xout = [Xout;X(2:end,:)];
    Uout = [Uout;repmat(Ut',[lent,1])];
    Xdout = [Xdout;repmat(Xd(:,1)',[lent,1])];
    Udout = [Udout;repmat(Ud(:,1)',[lent,1])];
    Uext = [Uext;repmat(u_ext',[lent,1])];
    FSMout = [FSMout;repmat(FSM',[lent,1])];
    if ii == 62

        testamento = 1;
    end
    waitbar(ii/MAX_ITER,h_waitbar,'Calculating...');
end
close(h_waitbar)
fprintf('Calculation Complete!\n')
toc

%% Animation

[t,EA,EAd] = fig_animate(tout,Xout,Uout,Xdout,Udout,Uext,p);
%[t,EA,EAd] = fig_animate_default(tout,Xout,Uout,Xdout,Udout,Uext,p);
%% plot states
fig_plot(tout,Xout,Uout,Xdout,Udout,Uext,p)
if saveMAT
    save('mpc_experiment.mat', 'tout', 'Xout','Uout','Xdout','Udout','Uext','-v7.3');
end


