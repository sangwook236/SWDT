data {
	int<lower=0> J;  // Number of schools.
	real y[J];  // Estimated treatment effects (school j).
	real<lower=0> sigma[J];  // Std err of effect estimates (school j).
}
#parameters {
#	real mu;
#	real theta[J];
#	real<lower=0> tau;
#}
parameters {
	real mu;
	real<lower=0> tau;
	real eta[J];
}
transformed parameters {
	real theta[J];
	for (j in 1:J)
    	theta[j] = mu + tau * eta[j];
}
model {
#	theta ~ normal(mu, tau);
	eta ~ normal(0, 1);
	y ~ normal(theta, sigma);
}
