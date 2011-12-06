%addpath('E:\work_center\sw_dev\matlab\rnd\bayesian_filtering\ekfukf_1_2\ekfukf');

% Handles to dynamic and measurement model functions,
% and their derivatives
f_func = @ungm_f;
h_func = @ungm_h;

% Number of samples
n = 500;

% Initial state and covariance
x_0 = .1;
P_0 = 1;

% Space for measurements
Y = zeros(1,n);

% Strengths of perturbations
u_n = 1;
v_n = 1;

fprintf('Generating real states and measurements...');
% Generate the true states with process noise
X = zeros(1,n);
X(1) = ungm_f(x_0,1) + gauss_rnd(0,u_n,1);
for i = 2:n
    X(i) = feval(f_func,X(i-1),i) + gauss_rnd(0,u_n,1);
end
    
% Generate the observations with measurement noise
for i = 1:n
    Y(i) = feval(h_func,X(i)) + gauss_rnd(0,v_n,1);
end

Y_r = feval(h_func, X);

fprintf('Done!\n');

% Parameters for dynamic model. Used by smoothers.
params = cell(size(Y,2));
for i = 1:size(Y,2)
   params{i} = i+1; 
end

fprintf('Filtering with UKF2...');

% Initial values and space for non-augmented UKF (UKF1)
M = x_0;
P = P_0;
MM_UKF2 = zeros(size(M,1),size(Y,2));
PP_UKF2 = zeros(size(M,1),size(M,1),size(Y,2));

for k = 1:size(Y,2)
	[M,P,X_s,w] = ukf_predict3(M,P,f_func,u_n,v_n,k);
	[M,P] = ukf_update3(M,P,Y(:,k),h_func,v_n,X_s,w,[]);
	MM_UKF2(:,k) = M;
	PP_UKF2(:,:,k) = P;
end;

fprintf('Done!\n');

plot(1:100,X(1:100),'-kx', 1:100,MM_UKF2(1:100),'--bo')
title('UKF2 filtering result');
xlim([0 100]);
ylim([-20 20]);
legend('Real signal', 'UKF2 filtered estimate');
