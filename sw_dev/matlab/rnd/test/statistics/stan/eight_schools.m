%addpath('D:\lib_repo\matlab\rnd\MatlabProcessManager_github');
%addpath('D:\lib_repo\matlab\rnd\MatlabStan_github');

% REF [site] >> https://github.com/brain-lau/MatlabStan/wiki/Getting-Started

%---------------------------------------------------------------------

schools_code = {
   'data {'
   '	int<lower=0> J;  // Number of schools.'
   '	real y[J];  // Estimated treatment effects (school j).'
   '	real<lower=0> sigma[J];  // Std err of effect estimates (school j).'
   '}'
   'parameters {'
   '	real mu;'
   '	real<lower=0> tau;'
   '	real eta[J];'
   '}'
   'transformed parameters {'
   '	real theta[J];'
   '	for (j in 1:J)'
   '    	theta[j] = mu + tau * eta[j];'
   '}'
   'model {'
   '	eta ~ normal(0, 1);'
   '	y ~ normal(theta, sigma);'
   '}'
};

schools_dat = struct('J', 8, ...
                     'y', [28 8 -3 7 -1 1 18 12], ...
                     'sigma', [15 10 16 11 9 11 10 18]);

fit1 = stan('model_code', schools_code, 'data', schools_dat);

print(fit1);

eta1 = fit1.extract('permuted', true).eta;
mean(eta1)

%---------------------------------------------------------------------

fit2 = stan('file', 'eight_schools.stan', 'data', schools_dat, 'iter', 1000, 'chains', 4);

print(fit2);

eta2 = fit2.extract('permuted', true).eta;
mean(eta2)

%---------------------------------------------------------------------

fit3 = stan('fit', fit1, 'data', schools_dat, 'iter', 1000, 'chains', 4);

print(fit3);

eta3 = fit3.extract('permuted', true).eta;
mean(eta3)
