#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import sklearn.linear_model
import matplotlib.pyplot as plt

# REF [site] >> https://github.com/groverpr/Machine-Learning/blob/master/notebooks/09_Quantile_Regression.ipynb
def simple_quantile_regression_example():
	y = np.arange(1, 25, 0.25)

	# Linear relationship with contant variance of residual.
	x1 = y.copy() + np.random.randn(96)

	# Non-contant variance with residuals .
	x2 = y.copy()
	y2 = x2 + np.concatenate((np.random.randn(20) * 0.5,
		np.random.randn(20) * 1, 
		np.random.randn(20) * 4, 
		np.random.randn(20) * 6, 
		np.random.randn(16) * 8), axis=0)

	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

	ax1.plot(x1, y, 'o')
	ax1.set_xlabel('X1')
	ax1.set_ylabel('Y')
	ax1.set_title('Contant variance of residuals')

	ax2.plot(x2, y2, 'o')
	ax2.set_xlabel('X2')
	ax2.set_ylabel('Y')
	ax2.set_title('Non contant variance of residuals')

	fig.tight_layout()
	plt.show()

	#--------------------
	# Linear regression

	lr = sklearn.linear_model.LinearRegression()
	lr.fit(x1.reshape(-1, 1), y.reshape(-1, 1))

	lr2 = sklearn.linear_model.LinearRegression()
	lr2.fit(x2.reshape(-1, 1), y.reshape(-1, 1))

	fig, (ax1,ax2) = plt.subplots(1, 2, figsize = (12, 5.5) )

	ax1.plot(x1, y, 'o')
	ax1.set_xlabel('X1')
	ax1.set_ylabel('Y')
	ax1.set_title('Contant variance of residuals')
	ax1.plot(x1, lr.predict(x1.reshape(-1, 1)))

	ax2.plot(x2, y2, 'o')
	ax2.set_xlabel('X2')
	ax2.set_ylabel('Y')
	ax2.set_title('Non contant variance of residuals')
	ax2.plot(x2, lr2.predict(x2.reshape(-1, 1)))

	fig.tight_layout()
	plt.show()

	#--------------------
	# Quantile regression.

	data = pd.DataFrame(data={'X': x2, 'Y': y2})

	mod = smf.quantreg('Y ~ X', data)
	res = mod.fit(q=0.5)

	def fit_model(q):
		res = mod.fit(q=q)
		return [q, res.params['Intercept'], res.params['X']] + res.conf_int().loc['X'].tolist()
		
	quantiles = (0.05, 0.95)
	models = [fit_model(x) for x in quantiles]
	models = pd.DataFrame(models, columns=['q', 'a', 'b', 'lb', 'ub'])

	ols = smf.ols('Y ~ X', data).fit()
	ols_ci = ols.conf_int().loc['X'].tolist()
	ols = dict(a = ols.params['Intercept'],
		b = ols.params['X'],
		lb = ols_ci[0],
		ub = ols_ci[1])

	print(models)
	print(ols)

	xn = np.arange(data.X.min(), data.X.max(), 2)
	get_y = lambda a, b: a + b * xn

	fig, ax = plt.subplots(figsize=(8, 6))

	for i in range(models.shape[0]):
		yn = get_y(models.a[i], models.b[i])
		ax.plot(xn, yn, linestyle='dotted', color='grey')

	yn = get_y(ols['a'], ols['b'])

	ax.plot(xn, yn, color='red', label='OLS')

	ax.scatter(data.X, data.Y, alpha=0.2)
	legend = ax.legend()
	ax.set_xlabel('X', fontsize=16)
	ax.set_ylabel('Y', fontsize=16)
	ax.set_title('Quantile regression with 0.05 and 0.95 quantiles')

	fig.tight_layout()
	plt.show()

def main():
	simple_quantile_regression_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
