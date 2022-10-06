#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import aesara.tensor as at
import matplotlib.pyplot as plt

# REF [site] >> https://www.pymc.io/welcome.html
def demo():
	# Assume 10 trials and 5 successes out of those trials.
	# Change these numbers to see how the posterior plot changes.
	trials = 10; successes = 5

	# Set up model context.
	with pm.Model() as coin_flip_model:
		# Probability p of success we want to estimate and assign Beta prior.
		p = pm.Beta("p", alpha=1, beta=1)
		
		# Define likelihood.
		obs = pm.Binomial("obs", p=p, n=trials,
			observed=successes,
		)

		# Hit Inference Button.
		idata = pm.sample()

	az.plot_posterior(idata, show=True);

# REF [site] >> https://www.pymc.io/projects/docs/en/latest/learn/core_notebooks/pymc_overview.html
def linear_regression_example():
	# Generating data.

	# Initialize random number generator.
	RANDOM_SEED = 8927
	rng = np.random.default_rng(RANDOM_SEED)
	az.style.use("arviz-darkgrid")

	# True parameter values.
	alpha, sigma = 1, 1
	beta = [1, 2.5]

	# Size of dataset.
	size = 100

	# Predictor variable.
	X1 = np.random.randn(size)
	X2 = np.random.randn(size) * 0.2

	# Simulate outcome variable.
	Y = alpha + beta[0] * X1 + beta[1] * X2 + rng.normal(size=size) * sigma

	fig, axes = plt.subplots(1, 2, sharex=True, figsize=(10, 4))
	axes[0].scatter(X1, Y, alpha=0.6)
	axes[1].scatter(X2, Y, alpha=0.6)
	axes[0].set_ylabel("Y")
	axes[0].set_xlabel("X1")
	axes[1].set_xlabel("X2");

	plt.show()

	# Model Specification.
	print(f"Running on PyMC v{pm.__version__}")

	basic_model = pm.Model()

	with basic_model:
		# Priors for unknown model parameters.
		alpha = pm.Normal("alpha", mu=0, sigma=10)
		beta = pm.Normal("beta", mu=0, sigma=10, shape=2)
		sigma = pm.HalfNormal("sigma", sigma=1)

		# Expected value of outcome.
		mu = alpha + beta[0] * X1 + beta[1] * X2

		# Likelihood (sampling distribution) of observations.
		Y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma, observed=Y)

	with basic_model:
		# Draw 1000 posterior samples.
		idata = pm.sample()

	idata.posterior["alpha"].sel(draw=slice(0, 4))

	with basic_model:
		# Instantiate sampler.
		step = pm.Slice()

		# Draw 5000 posterior samples.
		slice_idata = pm.sample(5000, step=step)

	slice_idata.posterior["alpha"].sel(draw=slice(0, 4))

	# Posterior analysis.
	az.plot_trace(idata, combined=True);
	print(az.summary(idata, round_to=2))

# REF [site] >> https://www.pymc.io/projects/docs/en/latest/learn/core_notebooks/pymc_overview.html
def educational_outcomes_for_hearing_impaired_children_example():
	# The Data.
	test_scores = pd.read_csv(pm.get_data("test_scores.csv"), index_col=0)
	test_scores.head()
	test_scores["score"].hist()

	# Dropping missing values is a very bad idea in general, but we do so here for simplicity.
	X = test_scores.dropna().astype(float)
	y = X.pop("score")

	# Standardize the features.
	X -= X.mean()
	X /= X.std()

	N, D = X.shape

	# The Model.
	D0 = int(D / 2)

	# Model Specification.
	with pm.Model(coords={"predictors": X.columns.values}) as test_score_model:
		# Prior on error SD.
		sigma = pm.HalfNormal("sigma", 25)

		# Global shrinkage prior.
		tau = pm.HalfStudentT("tau", 2, D0 / (D - D0) * sigma / np.sqrt(N))
		# Local shrinkage prior.
		lam = pm.HalfStudentT("lam", 2, dims="predictors")
		c2 = pm.InverseGamma("c2", 1, 0.1)
		z = pm.Normal("z", 0.0, 1.0, dims="predictors")
		# Shrunken coefficients.
		beta = pm.Deterministic(
			"beta", z * tau * lam * at.sqrt(c2 / (c2 + tau**2 * lam**2)), dims="predictors"
		)
		# No shrinkage on intercept.
		beta0 = pm.Normal("beta0", 100, 25.0)

		scores = pm.Normal("scores", beta0 + at.dot(X.values, beta), sigma, observed=y.values)

		pm.model_to_graphviz(test_score_model)

	with test_score_model:
		prior_samples = pm.sample_prior_predictive(100)

	az.plot_dist(
		test_scores["score"].values,
		kind="hist",
		color="C1",
		hist_kwargs=dict(alpha=0.6),
		label="observed",
	)
	az.plot_dist(
		prior_samples.prior_predictive["scores"],
		kind="hist",
		hist_kwargs=dict(alpha=0.6),
		label="simulated",
	)
	plt.xticks(rotation=45)

	# Model Fitting.
	with test_score_model:
		idata = pm.sample(1000, tune=2000, random_seed=42)
		#idata = pm.sample(1000, tune=2000, random_seed=42, target_accept=0.99)

	# Model Checking.
	az.plot_trace(idata, var_names=["tau", "sigma", "c2"])
	az.plot_energy(idata)
	az.plot_forest(idata, var_names=["beta"], combined=True, hdi_prob=0.95, r_hat=True)

# REF [site] >> https://www.pymc.io/projects/docs/en/latest/learn/core_notebooks/pymc_overview.html
def coal_mining_disasters_example():
	# fmt: off.
	disaster_data = pd.Series(
		[4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,
		3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5,
		2, 2, 3, 4, 2, 1, 3, np.nan, 2, 1, 1, 1, 1, 3, 0, 0,
		1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
		0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
		3, 3, 1, np.nan, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,
		0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1]
	)
	# fmt: on.
	years = np.arange(1851, 1962)

	plt.plot(years, disaster_data, "o", markersize=8, alpha=0.4)
	plt.ylabel("Disaster count")
	plt.xlabel("Year")

	plt.show()

	with pm.Model() as disaster_model:
		switchpoint = pm.DiscreteUniform("switchpoint", lower=years.min(), upper=years.max())

		# Priors for pre- and post-switch rates number of disasters.
		early_rate = pm.Exponential("early_rate", 1.0)
		late_rate = pm.Exponential("late_rate", 1.0)

		# Allocate appropriate Poisson rates to years before and after current.
		rate = pm.math.switch(switchpoint >= years, early_rate, late_rate)

		disasters = pm.Poisson("disasters", rate, observed=disaster_data)

	with disaster_model:
		idata = pm.sample(10000)

	axes_arr = az.plot_trace(idata)
	plt.draw()
	for ax in axes_arr.flatten():
		if ax.get_title() == "switchpoint":
			labels = [label.get_text() for label in ax.get_xticklabels()]
			ax.set_xticklabels(labels, rotation=45, ha="right")
			break
	plt.draw()

	plt.figure(figsize=(10, 8))
	plt.plot(years, disaster_data, ".", alpha=0.6)
	plt.ylabel("Number of accidents", fontsize=16)
	plt.xlabel("Year", fontsize=16)

	trace = idata.posterior.stack(draws=("chain", "draw"))

	plt.vlines(trace["switchpoint"].mean(), disaster_data.min(), disaster_data.max(), color="C1")
	average_disasters = np.zeros_like(disaster_data, dtype="float")
	for i, year in enumerate(years):
		idx = year < trace["switchpoint"]
		average_disasters[i] = np.mean(np.where(idx, trace["early_rate"], trace["late_rate"]))

	sp_hpd = az.hdi(idata, var_names=["switchpoint"])["switchpoint"].values
	plt.fill_betweenx(
		y=[disaster_data.min(), disaster_data.max()],
		x1=sp_hpd[0],
		x2=sp_hpd[1],
		alpha=0.5,
		color="C1",
	)
	plt.plot(years, average_disasters, "k--", lw=2);

# REF [site] >> https://www.pymc.io/projects/docs/en/latest/learn/core_notebooks/pymc_overview.html
def arbitrary_deterministics_and_distributions_example():
	# Arbitrary deterministics.
	from aesara.compile.ops import as_op

	@as_op(itypes=[at.lscalar], otypes=[at.lscalar])
	def crazy_modulo3(value):
		if value > 0:
			return value % 3
		else:
			return (-value + 1) % 3

	with pm.Model() as model_deterministic:
		a = pm.Poisson("a", 1)
		b = crazy_modulo3(a)

	# Arbitrary distributions.
	with pm.Model() as model:
		alpha = pm.Uniform('intercept', -100, 100)
		
		# Create custom densities.
		beta = pm.DensityDist('beta', logp=lambda value: -1.5 * at.log(1 + value**2))
		eps = pm.DensityDist('eps', logp=lambda value: -at.log(at.abs_(value)))
		
		# Create likelihood.
		like = pm.Normal('y_est', mu=alpha + beta * X, sigma=eps, observed=Y)

	class BetaRV(at.random.op.RandomVariable):
		name = "beta"
		ndim_supp = 0
		ndims_params = []
		dtype = "floatX"

		@classmethod
		def rng_fn(cls, rng, size):
			raise NotImplementedError("Cannot sample from beta variable")

	beta = BetaRV()

	class Beta(pm.Continuous):
		rv_op = beta

		@classmethod
		def dist(cls, mu=0, **kwargs):
			mu = at.as_tensor_variable(mu)
			return super().dist([mu], **kwargs)

		def logp(self, value):
			mu = self.mu
			return beta_logp(value - mu)

	def beta_logp(value):
		return -1.5 * at.log(1 + (value) ** 2)

	with pm.Model() as model:
		beta = Beta("beta", mu=0)

def main():
	#demo()

	#linear_regression_example()
	#educational_outcomes_for_hearing_impaired_children_example()
	coal_mining_disasters_example()
	#arbitrary_deterministics_and_distributions_example()  # Not yet tested.

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
