function fig_plot(tout,Xout,Uout,Xdout,Udout,Uext,p)
%% Robot path
% figure()
% plot3(Xout(:,1), Xout(:,2), Xout(:,3)); grid on
% %plot(Xdout(:,1), Xdout(:,2)); grid on
% xlabel('x(t)')
% ylabel('y(t)')
% zlabel('z(t)')

%% Z direction foot forces

X1 = tout;
YMatrix1 = [Uout(:,3),Udout(:,3)];
YMatrix2 = [Uout(:,6),Udout(:,6)];
YMatrix3 = [Uout(:,9),Udout(:,9)];
YMatrix4 = [Uout(:,12),Udout(:,12)];


%% state plots
theta = []; thetad = [];
for i = 1:length(Xout)
    R = reshape(Xout(i,7:15),[3,3]);
    Rd = reshape(Xdout(i,7:15),[3,3]);
    theta = [theta; veeMap(logm(R))']; 
    thetad = [thetad; veeMap(logm(Rd))']; 
end

YMatrix5 = [Xout(:,1:2),Xdout(:,1:2)];
YMatrix6 = [Xout(:,4:5),Xdout(:,4:5)];
YMatrix7 = [theta(:,1:3),thetad(:,3)];
YMatrix8 = [Xout(:,16:18),Xdout(:,18)];

%% plot 2 column figure

createfigure(X1, YMatrix1, YMatrix2, YMatrix3, YMatrix4, ...
                    YMatrix5, YMatrix6, YMatrix7, YMatrix8);
end
