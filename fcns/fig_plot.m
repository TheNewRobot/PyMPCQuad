function fig_plot(tout,Xout,Uout,Xdout,Udout,Uext,p)
%% Robot path
% figure()
% plot3(Xout(:,1), Xout(:,2), Xout(:,3)); grid on
% %plot(Xdout(:,1), Xdout(:,2)); grid on
% xlabel('x(t)')
% ylabel('y(t)')
% zlabel('z(t)')

%% Z direction foot forces
fig = figure();
subplot(4,1,1)
plot(tout,Uout(:,3),'LineWidth',2); hold on;
plot(tout,Udout(:,3),'LineWidth',2); hold off;
ylabel('f_1 (N)')
legend('MPC','ideal')
hold off

subplot(4,1,2)
plot(tout,Uout(:,6),'LineWidth',2); hold on;
plot(tout,Udout(:,6),'LineWidth',2); hold off;
ylabel('f_2 (N)')
legend('MPC','ideal')
hold off

subplot(4,1,3)
plot(tout,Uout(:,9),'LineWidth',2); hold on;
plot(tout,Udout(:,9),'LineWidth',2); hold off;
ylabel('f_3 (N)')
legend('MPC','ideal')
hold off

subplot(4,1,4)
plot(tout,Uout(:,12),'LineWidth',2); hold on;
plot(tout,Udout(:,12),'LineWidth',2); hold off;
ylabel('f_4 (N)')
legend('MPC','ideal')
hold off

handle=axes(fig,'visible','off');
handle.XLabel.Visible='on';
xlabel(handle,'time (s)');
%% Y direction foot forces
% figure()
% subplot(2,2,1)
% plot(Ud_ref(2,:),'--'); hold on;
% plot(Ut_ref(2,:))
% title('Y direction foot force of front left foot')
% xlabel('time (ms)')
% ylabel('force (N)')
% legend('reference','MPC')
% hold off
% 
% subplot(2,2,2)
% plot(Ud_ref(5,:),'--'); hold on;
% plot(Ut_ref(5,:))
% title('Y direction foot force of front left foot')
% xlabel('time (ms)')
% ylabel('force (N)')
% legend('reference','MPC')
% hold off
% 
% subplot(2,2,3)
% plot(Ud_ref(5,:),'--'); hold on;
% plot(Ut_ref(5,:))
% title('Y direction foot force of front left foot')
% xlabel('time (ms)')
% ylabel('force (N)')
% legend('reference','MPC')
% hold off
% 
% subplot(2,2,4)
% plot(Ud_ref(11,:),'--'); hold on;
% plot(Ut_ref(11,:))
% title('Y direction foot force of front left foot')
% xlabel('time (ms)')
% ylabel('force (N)')
% legend('reference','MPC')
% hold off
% %% X direction foot forces
% figure()
% subplot(2,2,1)
% plot(Ud_ref(1,:),'--'); hold on;
% plot(Ut_ref(1,:))
% title('X direction foot force of front left foot')
% xlabel('time (ms)')
% ylabel('force (N)')
% legend('reference','MPC')
% hold off
% 
% subplot(2,2,2)
% plot(Ud_ref(4,:),'--'); hold on;
% plot(Ut_ref(4,:))
% title('Z direction foot force of front left foot')
% xlabel('time (ms)')
% ylabel('force (N)')
% legend('reference','MPC')
% hold off
% 
% subplot(2,2,3)
% plot(Ud_ref(4,:),'--'); hold on;
% plot(Ut_ref(4,:))
% title('X direction foot force of front left foot')
% xlabel('time (ms)')
% ylabel('force (N)')
% legend('reference','MPC')
% hold off
% 
% subplot(2,2,4)
% plot(Ud_ref(10,:),'--'); hold on;
% plot(Ut_ref(10,:))
% title('X direction foot force of front left foot')
% xlabel('time (ms)')
% ylabel('force (N)')
% legend('reference','MPC')
% hold off

%% state plots
theta = []; thetad = [];
for i = 1:length(Xout)
    R = reshape(Xout(i,7:15),[3,3]);
    Rd = reshape(Xdout(i,7:15),[3,3]);
    theta = [theta; veeMap(logm(R))']; 
    thetad = [thetad; veeMap(logm(Rd))']; 
end

figure()
subplot(4,1,1)
plot(tout,Xout(:,1)); hold on;
plot(tout,Xout(:,2)); hold on;
plot(tout,Xout(:,3)); hold on;
plot(tout,Xdout(:,1)); hold on;
plot(tout,Xdout(:,2)); hold on;
plot(tout,Xdout(:,3)); hold on;
xlabel('time (s)')
ylabel('Position (m)')
legend('p_x','p_y','p_z')
hold off

subplot(4,1,2)
plot(tout,Xout(:,4)); hold on;
plot(tout,Xout(:,5)); hold on;
plot(tout,Xout(:,6)); hold on;
plot(tout,Xdout(:,4)); hold on;
plot(tout,Xdout(:,5)); hold on;
plot(tout,Xdout(:,6)); hold on;
xlabel('time (s)')
ylabel('Velocity (m/s)')
legend('v_x','v_y','v_z')
hold off

subplot(4,1,3)
plot(tout,theta(:,1)); hold on;
plot(tout,theta(:,2)); hold on;
plot(tout,theta(:,3)); hold on;
plot(tout,thetad(:,1)); hold on;
plot(tout,thetad(:,2)); hold on;
plot(tout,thetad(:,3)); hold on;
xlabel('time (s)')
ylabel('Orientation (rad)')
legend('\phi','\theta','\psi')
hold off

subplot(4,1,4)
plot(tout,Xout(:,16)); hold on;
plot(tout,Xout(:,17)); hold on;
plot(tout,Xout(:,18)); hold on;
plot(tout,Xdout(:,16)); hold on;
plot(tout,Xdout(:,17)); hold on;
plot(tout,Xdout(:,18)); hold on;
xlabel('time (s)')
ylabel('velocity (m/s)')
legend('\omega_\phi','\omega_\theta','\omega_\psi')
hold off
end
