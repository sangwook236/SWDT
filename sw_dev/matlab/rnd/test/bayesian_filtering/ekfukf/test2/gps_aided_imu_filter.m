addpath('E:\work_center\sw_dev\matlab\rnd\src\bayesian_filtering\ekfukf_1_2\ekfukf');

%
fprintf('Generating real states and measurements...');

STATE_DIM = 16;
OUTPUT_DIM = 6;

% Strengths of perturbations
Qc = 1;
Q_n = eye(STATE_DIM) * Qc;
Rc = 1;
R_n = eye(OUTPUT_DIM) * Rc;

T = 100;  % [sec]
Ts = 0.1;  % [sec]
L = [ 10 10 10 ];  % [m]
time = [0:Ts:T]';
Ndata = length(time);

[ traj_poses traj_vels traj_accels ] = synthesize_data_using_trigonometric(time, Ts, T, L);
traj_poses = traj_poses + gauss_rnd(zeros(3,1), R_n(1:3,1:3), Ndata)';
traj_vels = traj_vels + gauss_rnd(zeros(3,1), R_n(4:6,4:6), Ndata)';
traj_accels = traj_accels + gauss_rnd(zeros(3,1), Qc * eye(3), Ndata)';
% FIXME [modify] >>
traj_angular_vels = zeros(size(traj_accels)) + gauss_rnd(zeros(3,1), Qc * eye(3), Ndata)';

% Handles to dynamic and measurement model functions,
% and their derivatives
f_func = @imu_f;
h_func = @gps_h;
df_func = @imu_df_dx;
dh_func = @gps_dh_dx;

% Initial state and covariance
x_0 = zeros(STATE_DIM,1);
x_0(7) = 1;
Pc = 1.0e-5;
P_0 = Pc * eye(STATE_DIM);

% Initial valuesand space for augmented UKF (UKF2)
M = x_0;
P = P_0;
MM_UKF2 = zeros(STATE_DIM, Ndata);
PP_UKF2 = zeros(STATE_DIM, STATE_DIM, Ndata);

% Filtering loop for UKF2
param_f = zeros(8,1);
param_f(1) = Ts;
param_h = zeros(2,1);
param_h(1) = Ts;

%alpha = 0.5;
%beta = 2;
%kappa = 3 - STATE_DIM;

for k = 1:Ndata
	k
	param_f(2) = k;
	param_f(3:5) = traj_accels(k,:)';
	param_f(6:8) = traj_angular_vels(k,:)';
	param_h(2) = k;

   [M, P] = ukf_predict2(M, P, f_func, Q_n, param_f);
   %[M, P, X_s, w] = ukf_predict3(M, P, f_func, Q_n, R_n, param_f);
   Y = [ traj_poses(k,:) traj_vels(k,:) ]';
   [M, P] = ukf_update2(M, P, Y, h_func, R_n, param_h);
   %[M, P] = ukf_update3(M, P, Y, h_func, R_n, X_s, w, param_h);
   MM_UKF2(:,k)   = M;
   PP_UKF2(:,:,k) = P;
end

fprintf('Done!\n');

%plot(1:Ndata, X(1:Ndata), '-kx', 1:Ndata, MM_UKF2(1:Ndata), '--bo')
plot(1:Ndata, MM_UKF2(1:Ndata), '--bo')
title('UKF2 filtering result');
xlim([0 Ndata]);
ylim([-20 20]);
