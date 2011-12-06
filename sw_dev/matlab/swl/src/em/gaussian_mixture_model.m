%% --------------------------------------------------------------
mu0_true = 0;
mu1_true = -0.2;
sigma0_true = 1;
sigma1_true = 0.1;
alpha_true = 0.3;
[ mu0_true mu1_true sigma0_true sigma1_true alpha_true ]

N = 1000;

Z = (randi(10000, N, 1) / 10000) <= alpha_true;
N1_true = sum(Z == 0);
N2_true = sum(Z == 1);
X = zeros(N,1);
for ii = 1:N
    if Z(ii) == 1
        X(ii) = mu1_true + sigma1_true * randn(1);
    else
        X(ii) = mu0_true + sigma0_true * randn(1);
    end;
end;

alpha_init = 0.5;
mu0_init = 1;
sigma0_init = 1;
mu1_init = -1;
sigma1_init = 1;

%% --------------------------------------------------------------
% standard(batch) EM using sufficient statistics
alpha_est = alpha_init;
mu0_est = mu0_init;
sigma0_est = sigma0_init;
mu1_est = mu1_init;
sigma1_est = sigma1_init;

gamma = zeros(N,1);
looping = true;
step = 0;
max_step = 1000;
tol = 1e-10;
while looping && step <= max_step
    N0 = 0; M0 = 0; Q0 = 0;
    N1 = 0; M1 = 0; Q1 = 0;

    IDX = 1:N;
    %IDX = randperm(N);
    for kk = 1:N
        ii = IDX(kk);
        
        p0 = normpdf(X(ii), mu0_est, sigma0_est);
        p1 = normpdf(X(ii), mu1_est, sigma1_est);
        gamma(ii) = alpha_est * p1 / ((1 - alpha_est) * p0 + alpha_est * p1);

        % expectation of sufficient statistics
        N0 = N0 + (1 - gamma(ii));
        M0 = M0 + (1 - gamma(ii)) * X(ii);
        Q0 = Q0 + (1 - gamma(ii)) * X(ii)^2;
        N1 = N1 + gamma(ii);
        M1 = M1 + gamma(ii) * X(ii);
        Q1 = Q1 + gamma(ii) * X(ii)^2;
    end;

    mu0_est_old = mu0_est;
    mu1_est_old = mu1_est;
    sigma0_est_old = sigma0_est;
    sigma1_est_old = sigma1_est;
    alpha_est_old = alpha_est;

    mu0_est = M0 / N0;
    mu1_est = M1 / N1;
    sigma0_est = sqrt((Q0 - 2*mu0_est*M0 + mu0_est^2*N0) / N0);
    sigma1_est = sqrt((Q1 - 2*mu1_est*M1 + mu1_est^2*N1) / N1);
    alpha_est = N1 / (N0 + N1);
    
    if abs(mu0_est - mu0_est_old) <= tol && abs(mu1_est - mu1_est_old) <= tol ...
            && abs(sigma0_est - sigma0_est_old) <= tol && abs(sigma1_est - sigma1_est_old) <= tol ...
            && abs(alpha_est - alpha_est_old) <= tol
        looping = false;
    end;
    
    step = step + 1;
end;

mu0_est_bat = mu0_est;
mu1_est_bat = mu1_est;
sigma0_est_bat = sigma0_est;
sigma1_est_bat = sigma1_est;
alpha_est_bat = alpha_est;
[ mu0_est_bat mu1_est_bat sigma0_est_bat sigma1_est_bat alpha_est_bat ]
step

%% --------------------------------------------------------------
% incremental EM using sufficient statistics
alpha_est = alpha_init;
mu0_est = mu0_init;
sigma0_est = sigma0_init;
mu1_est = mu1_init;
sigma1_est = sigma1_init;

% apply standard EM once in order to initialize sufficient statistics
N0 = 0; M0 = 0; Q0 = 0;
N1 = 0; M1 = 0; Q1 = 0;
gamma = zeros(N,1);
IDX = 1:N;
%IDX = randperm(N);
for kk = 1:N
    ii = IDX(kk);
    
    p0 = normpdf(X(ii), mu0_est, sigma0_est);
    p1 = normpdf(X(ii), mu1_est, sigma1_est);
    gamma(ii) = alpha_est * p1 / ((1 - alpha_est) * p0 + alpha_est * p1);

    % expectation of sufficient statistics
    N0 = N0 + (1 - gamma(ii));
    M0 = M0 + (1 - gamma(ii)) * X(ii);
    Q0 = Q0 + (1 - gamma(ii)) * X(ii)^2;
    N1 = N1 + gamma(ii);
    M1 = M1 + gamma(ii) * X(ii);
    Q1 = Q1 + gamma(ii) * X(ii)^2;
end;

mu0_est = M0 / N0;
mu1_est = M1 / N1;
sigma0_est = sqrt((Q0 - 2*mu0_est*M0 + mu0_est^2*N0) / N0);
sigma1_est = sqrt((Q1 - 2*mu1_est*M1 + mu1_est^2*N1) / N1);
alpha_est = N1 / (N0 + N1);

% incremental EM
looping = true;
step = 0;
max_step = 1000;
tol = 1e-10;
while looping && step <= max_step
    mu0_est_old = mu0_est;
    mu1_est_old = mu1_est;
    sigma0_est_old = sigma0_est;
    sigma1_est_old = sigma1_est;
    alpha_est_old = alpha_est;
    
    IDX = 1:N;
    %IDX = randperm(N);
    for kk = 1:N
        ii = IDX(kk);

        % subtract expectation of previous sufficient statistics
        N0 = N0 - (1 - gamma(ii));
        M0 = M0 - (1 - gamma(ii)) * X(ii);
        Q0 = Q0 - (1 - gamma(ii)) * X(ii)^2;
        N1 = N1 - gamma(ii);
        M1 = M1 - gamma(ii) * X(ii);
        Q1 = Q1 - gamma(ii) * X(ii)^2;
        
        p0 = normpdf(X(ii), mu0_est, sigma0_est);
        p1 = normpdf(X(ii), mu1_est, sigma1_est);
        gamma(ii) = alpha_est * p1 / ((1 - alpha_est) * p0 + alpha_est * p1);

        % add expectation of updated sufficient statistics
        N0 = N0 + (1 - gamma(ii));
        M0 = M0 + (1 - gamma(ii)) * X(ii);
        Q0 = Q0 + (1 - gamma(ii)) * X(ii)^2;
        N1 = N1 + gamma(ii);
        M1 = M1 + gamma(ii) * X(ii);
        Q1 = Q1 + gamma(ii) * X(ii)^2;

        % Oops !!!
        % if the number of data is just 1, mu0_est = mu1_est & sigma0_est = sigma1_est = 0 & alpha_est = gamma
        mu0_est = M0 / N0;
        mu1_est = M1 / N1;
        sigma0_est = sqrt(Q0/N0 - (M0/N0)^2);
        sigma1_est = sqrt(Q1/N1 - (M1/N1)^2);
        alpha_est = N1 / (N0 + N1);
    end;

    if abs(mu0_est - mu0_est_old) <= tol && abs(mu1_est - mu1_est_old) <= tol ...
            && abs(sigma0_est - sigma0_est_old) <= tol && abs(sigma1_est - sigma1_est_old) <= tol ...
            && abs(alpha_est - alpha_est_old) <= tol
        looping = false;
    end;
    
    step = step + 1;
end;

mu0_est_inc = mu0_est;
mu1_est_inc = mu1_est;
sigma0_est_inc = sigma0_est;
sigma1_est_inc = sigma1_est;
alpha_est_inc = alpha_est;
[ mu0_est_inc mu1_est_inc sigma0_est_inc sigma1_est_inc alpha_est_inc ]
step

%% --------------------------------------------------------------
% incremental EM using sufficient statistics
% for sequential data (???)
alpha_est = alpha_init;
mu0_est = mu0_init;
sigma0_est = sigma0_init;
mu1_est = mu1_init;
sigma1_est = sigma1_init;

% apply standard EM once in order to initialize sufficient statistics
N0 = 0; M0 = 0; Q0 = 0;
N1 = 0; M1 = 0; Q1 = 0;
gamma = zeros(N,1);
IDX = 1:N;
%IDX = randperm(N);
for kk = 1:N
    ii = IDX(kk);
    
    p0 = normpdf(X(ii), mu0_est, sigma0_est);
    p1 = normpdf(X(ii), mu1_est, sigma1_est);
    gamma(ii) = alpha_est * p1 / ((1 - alpha_est) * p0 + alpha_est * p1);

    % expectation of sufficient statistics
    N0 = N0 + (1 - gamma(ii));
    M0 = M0 + (1 - gamma(ii)) * X(ii);
    Q0 = Q0 + (1 - gamma(ii)) * X(ii)^2;
    N1 = N1 + gamma(ii);
    M1 = M1 + gamma(ii) * X(ii);
    Q1 = Q1 + gamma(ii) * X(ii)^2;
end;

mu0_est = M0 / N0;
mu1_est = M1 / N1;
sigma0_est = sqrt((Q0 - 2*mu0_est*M0 + mu0_est^2*N0) / N0);
sigma1_est = sqrt((Q1 - 2*mu1_est*M1 + mu1_est^2*N1) / N1);
alpha_est = N1 / (N0 + N1);

% incremental EM (for sequential data)
IDX = 1:N;
%IDX = randperm(N);
for kk = 1:N
    ii = IDX(kk);
    
    p0 = normpdf(X(ii), mu0_est, sigma0_est);
    p1 = normpdf(X(ii), mu1_est, sigma1_est);
    gamma(ii) = alpha_est * p1 / ((1 - alpha_est) * p0 + alpha_est * p1);

    % expectation of sufficient statistics
    N0 = N0 + (1 - gamma(ii));
    M0 = M0 + (1 - gamma(ii)) * X(ii);
    Q0 = Q0 + (1 - gamma(ii)) * X(ii)^2;
    N1 = N1 + gamma(ii);
    M1 = M1 + gamma(ii) * X(ii);
    Q1 = Q1 + gamma(ii) * X(ii)^2;

    % Oops !!!
    % if the number of data is just 1, mu0_est = mu1_est & sigma0_est = sigma1_est = 0 & alpha_est = gamma
    mu0_est = M0 / N0;
    mu1_est = M1 / N1;
    sigma0_est = sqrt(Q0/N0 - (M0/N0)^2);
    sigma1_est = sqrt(Q1/N1 - (M1/N1)^2);
    alpha_est = N1 / (N0 + N1);
end;

mu0_est_seq = mu0_est;
mu1_est_seq = mu1_est;
sigma0_est_seq = sigma0_est;
sigma1_est_seq = sigma1_est;
alpha_est_seq = alpha_est;
[ mu0_est_seq mu1_est_seq sigma0_est_seq sigma1_est_seq alpha_est_seq ]
