function [Xd,Ud] = fcn_gen_XdUd(t,Xt,bool_inStance,p)

%% parameters
gait = p.gait;



%% generate reference trajectory
% X = [pc dpc eta wb]
lent = length(t);
Xd = zeros(30,lent);
Ud = zeros(12,lent);
Rground = p.Rground;           % ground slope
for ii = 1:lent
    %%%%%%%%%% any walking gait %%%%%%%%%%%%%
    if gait >= 0        % --- March forward and rotate ---
        acc_d = p.acc_d;
        vel_d = p.vel_d;
        yaw_d = p.yaw_d;
        wb_d = p.wb_d;
        ang_acc_d = p.ang_acc_d;
        %%%%%%%%%% linear motion %%%%%%%%%%%%%
        pc_d = [0;0;p.z0];
        dpc_d = [0;0;0];
        for jj = 1:2
            if t(ii) < (vel_d(jj) / acc_d)
                dpc_d(jj) = acc_d * t(ii);
                pc_d(jj) = 1/2 * acc_d * t(ii)^2;
                
            else
                dpc_d(jj) = vel_d(jj);
                pc_d(jj) = vel_d(jj) * t(ii) - 1/2 * vel_d(jj) * vel_d(jj)/acc_d;%          
            end
        end

        %%%%%%%%%% angular motion default %%%%%%%%%%%%%
%         if isempty(Xt)
%             ea_d = [0;0;0];
%         else
%             ea_d = [0;0;yaw_d];
%         end
%         vR_d = reshape(expm(hatMap(ea_d)),[9,1]);
%         wb_d = [0;0;0];
        
        %%%%%%%%%% angular motion %%%%%%%%%%%%%
        ea_d = [0;0;0];
        
        if isempty(Xt)
            ea_d = [0;0;0];
            wb_d = [0;0;0];
        else
            wb_d(3) = ang_acc_d(3) * t(ii);
            ea_d(3) = 1/2 * ang_acc_d(3) * t(ii)^2;
        end

        if  ea_d(3)>=yaw_d
            wb_d(3) = 0;
            ea_d(3) = yaw_d;
        end
        vR_d = reshape(expm(hatMap(ea_d)),[9,1]);
        pfd = reshape(Rground * p.pf34,[12,1]);
    end
    
    %%%% for pose control %%%%%%%%%
    if(gait==-1)
        roll_d = p.roll_d;
        pitch_d = p.pitch_d;
        yaw_d = p.yaw_d;
        ang_acc_d = p.ang_acc_d;
        ea_d = [0;0;0];
        wb_d = [0;0;0];
        if isempty(Xt)
                ea_d = [0;0;0];
                wb_d = [0;0;0];
            else
%                 ea_d = [roll_d;pitch_d;yaw_d];
                wb_d(2) = ang_acc_d(2) * t(ii);
                ea_d(2) = 1/2 * ang_acc_d(2) * t(ii)^2;
        end
        if  ea_d(2)>=pitch_d
            wb_d(2) = 0;
            ea_d(2) = pitch_d;
        end
        pc_d = [0;0;p.z0];
        dpc_d = [0;0;0];
        vR_d = reshape(expm(hatMap(ea_d)),[9,1]);
        pfd = reshape(Rground * p.pf34,[12,1]);
    end

    Xd(:,ii) = [pc_d;dpc_d;vR_d;wb_d;pfd];
    
    %%%% force for backflip %%%%%%%%%
    if (gait == -3)
        Ud(:,ii) = U_d;
    else
        sum_inStance = sum(bool_inStance(:,ii));
        if sum_inStance == 0    % four legs in swing
            Ud(:,ii) = zeros(12,1);
        else
            Ud([3,6,9,12],ii) = bool_inStance(:,ii)*(p.mass*p.g/sum_inStance);
        end
        
    end
end
testamento = 1;
end

