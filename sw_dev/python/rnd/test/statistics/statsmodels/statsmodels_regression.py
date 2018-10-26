#!/usr/bin/env python

import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# REF [site] >> https://www.statsmodels.org/stable/regression.html
def simple_regression_example():
	# Load data.
	spector_data = sm.datasets.spector.load()
	spector_data.exog = sm.add_constant(spector_data.exog, prepend=False)

	# Fit and summarize model.

	# OLS: ordinary least squares for i.i.d. errors Sigma = I.
	mod = sm.OLS(spector_data.endog, spector_data.exog)
	res = mod.fit()
	print(res.summary())

	# GLS: generalized least squares for arbitrary covariance Sigma.
	mod = sm.GLS(spector_data.endog, spector_data.exog)
	res = mod.fit()
	print(res.summary())

	# WLS: weighted least squares for heteroskedastic errors diag(Sigma).
	mod = sm.WLS(spector_data.endog, spector_data.exog)
	res = mod.fit()
	print(res.summary())

	# GLSAR: feasible generalized least squares with autocorrelated AR(p) errors Sigma = Sigma(rho).
	mod = sm.GLSAR(spector_data.endog, spector_data.exog)
	res = mod.fit()
	print(res.summary())

# REF [site] >> https://www.statsmodels.org/stable/glm.html
def generalized_linear_model_example():
	# Load data.
	data = sm.datasets.scotland.load()
	data.exog = sm.add_constant(data.exog)

	# Instantiate a gamma family model with the default link function.
	gamma_model = sm.GLM(data.endog, data.exog, family=sm.families.Gamma())
	gamma_results = gamma_model.fit()
	print(gamma_results.summary())

def generalized_estimating_equation_example():
	data = sm.datasets.get_rdataset('epil', package='MASS').data
	fam = sm.families.Poisson()
	ind = sm.cov_struct.Exchangeable()

	mod = smf.gee('y ~ age + trt + base', 'subject', data, cov_struct=ind, family=fam)
	res = mod.fit()
	print(res.summary())

def robust_linear_model():
	data = sm.datasets.stackloss.load()
	data.exog = sm.add_constant(data.exog)

	rlm_model = sm.RLM(data.endog, data.exog, M=sm.robust.norms.HuberT())
	rlm_results = rlm_model.fit()
	print(rlm_results.summary())
	#print(rlm_results.params)

def main():
	#simple_regression_example()

	#generalized_linear_model_example()
	#generalized_estimating_equation_example()
	robust_linear_model()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
