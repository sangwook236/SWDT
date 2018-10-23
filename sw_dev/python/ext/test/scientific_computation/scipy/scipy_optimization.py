#!/usr/bin/env python

# REF [site] >>
#	https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html
#	https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

from scipy.optimize import minimize
from scipy.optimize import least_squares
from scipy.optimize import minimize_scalar
from scipy.optimize import BFGS, SR1
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint, NonlinearConstraint
from scipy.optimize import OptimizeResult
#from scipy.optimize import rosen, rosen_der, rosen_hess
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import LinearOperator
from scipy.special import j1
import matplotlib.pyplot as plt
import numpy as np

def rosen(x):
	"""The Rosenbrock function"""
	return sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

def rosen_der(x):
	xm = x[1:-1]
	xm_m1 = x[:-2]
	xm_p1 = x[2:]
	der = np.zeros_like(x)
	der[1:-1] = 200 * (xm - xm_m1**2) - 400 * (xm_p1 - xm**2) * xm - 2*(1 - xm)
	der[0] = -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
	der[-1] = 200 * (x[-1] - x[-2]**2)
	return der

def rosen_hess(x):
	x = np.asarray(x)
	H = np.diag(-400 * x[:-1], 1) - np.diag(400 * x[:-1], -1)
	diagonal = np.zeros_like(x)
	diagonal[0] = 1200 * x[0]**2 - 400 * x[1]+2
	diagonal[-1] = 200
	diagonal[1:-1] = 202 + 1200 * x[1:-1]**2 - 400 * x[2:]
	H = H + np.diag(diagonal)
	return H

def rosen_hess_p(x, p):
	x = np.asarray(x)
	Hp = np.zeros_like(x)
	Hp[0] = (1200 * x[0]**2 - 400 * x[1] + 2) * p[0] - 400 * x[0] * p[1]
	Hp[1:-1] = -400 * x[:-2] * p[:-2] + (202 + 1200 * x[1:-1]**2 - 400 * x[2:]) * p[1:-1] - 400 * x[1:-1] * p[2:]
	Hp[-1] = -400 * x[-2] * p[-2] + 200 * p[-1]
	return Hp

def simple_unconstrained_minimization_example():
	x0 = [1.3, 0.7, 0.8, 1.9, 1.2]

	res = minimize(rosen, x0, method='Nelder-Mead', tol=1e-6)
	#res = minimize(rosen, x0, method='nelder-mead', options={'xtol': 1e-8, 'disp': True})
	print('Solution =', res.x)

	#res = minimize(rosen, x0, method='BFGS', jac=rosen_der, options={'gtol': 1e-6, 'disp': True})
	res = minimize(rosen, x0, method='BFGS', jac=rosen_der, options={'disp': True})
	print('Solution =', res.x)
	print('Message =', res.message)
	print('Inv(H) =', res.hess_inv)

	res = minimize(rosen, x0, method='Newton-CG', jac=rosen_der, hess=rosen_hess, options={'xtol': 1e-8, 'disp': True})
	print('Solution =', res.x)
	res = minimize(rosen, x0, method='Newton-CG', jac=rosen_der, hessp=rosen_hess_p, options={'xtol': 1e-8, 'disp': True})
	print('Solution =', res.x)

	res = minimize(rosen, x0, method='trust-ncg', jac=rosen_der, hess=rosen_hess, options={'gtol': 1e-8, 'disp': True})
	print('Solution =', res.x)
	res = minimize(rosen, x0, method='trust-ncg', jac=rosen_der, hessp=rosen_hess_p, options={'gtol': 1e-8, 'disp': True})
	print('Solution =', res.x)

	res = minimize(rosen, x0, method='trust-krylov', jac=rosen_der, hess=rosen_hess, options={'gtol': 1e-8, 'disp': True})
	print('Solution =', res.x)
	res = minimize(rosen, x0, method='trust-krylov', jac=rosen_der, hessp=rosen_hess_p, options={'gtol': 1e-8, 'disp': True})
	print('Solution =', res.x)

	res = minimize(rosen, x0, method='trust-exact', jac=rosen_der, hess=rosen_hess, options={'gtol': 1e-8, 'disp': True})
	print('Solution =', res.x)

def cons_f(x):
	return [x[0]**2 + x[1], x[0]**2 - x[1]]

def cons_J(x):
	return [[2 * x[0], 1], [2 * x[0], -1]]

def cons_H(x, v):
	return v[0] * np.array([[2, 0], [0, 0]]) + v[1] * np.array([[2, 0], [0, 0]])

def cons_H_sparse(x, v):
	return v[0] * csc_matrix([[2, 0], [0, 0]]) + v[1] * csc_matrix([[2, 0], [0, 0]])

def cons_H_linear_operator(x, v):
	def matvec(p):
		return np.array([p[0] * 2 * (v[0] + v[1]), 0])
	return LinearOperator((2, 2), matvec=matvec)

def rosen_hess_linop(x):
	def matvec(p):
		return rosen_hess_p(x, p)
	return LinearOperator((2, 2), matvec=matvec)

def simple_constrained_minimization_example():
	# Consider a minimization problem with several constraints.
	fun = lambda x: (x[0] - 1)**2 + (x[1] - 2.5)**2
	cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - 2 * x[1] + 2},
		{'type': 'ineq', 'fun': lambda x: -x[0] - 2 * x[1] + 6},
		{'type': 'ineq', 'fun': lambda x: -x[0] + 2 * x[1] + 2})
	bnds = ((0, None), (0, None))
	res = minimize(fun, (2, 0), method='SLSQP', bounds=bnds, constraints=cons)
	print('Solution =', res.x)

	# Trust-region constrained algorithm.
	bounds = Bounds([0, -0.5], [1.0, 2.0])
	linear_constraint = LinearConstraint([[1, 2], [2, 1]], [-np.inf, 1], [1, 1])
	nonlinear_constraint = NonlinearConstraint(cons_f, -np.inf, 1, jac=cons_J, hess=cons_H)
	#nonlinear_constraint = NonlinearConstraint(cons_f, -np.inf, 1, jac=cons_J, hess=cons_H_sparse)
	#nonlinear_constraint = NonlinearConstraint(cons_f, -np.inf, 1, jac=cons_J, hess=cons_H_linear_operator)
	#nonlinear_constraint = NonlinearConstraint(cons_f, -np.inf, 1, jac=cons_J, hess=BFGS())
	#nonlinear_constraint = NonlinearConstraint(cons_f, -np.inf, 1, jac=cons_J, hess='2-point')
	#nonlinear_constraint = NonlinearConstraint(cons_f, -np.inf, 1, jac='2-point', hess=BFGS())

	x0 = np.array([0.5, 0])
	res = minimize(rosen, x0, method='trust-constr', jac=rosen_der, hess=rosen_hess, constraints=[linear_constraint, nonlinear_constraint], options={'verbose': 1}, bounds=bounds)
	print('Solution =', res.x)
	res = minimize(rosen, x0, method='trust-constr', jac=rosen_der, hess=rosen_hess_linop, constraints=[linear_constraint, nonlinear_constraint], options={'verbose': 1}, bounds=bounds)
	print('Solution =', res.x)
	res = minimize(rosen, x0, method='trust-constr', jac=rosen_der, hessp=rosen_hess_p, constraints=[linear_constraint, nonlinear_constraint], options={'verbose': 1}, bounds=bounds)
	print('Solution =', res.x)
	res = minimize(rosen, x0, method='trust-constr', jac='2-point', hess=SR1(), constraints=[linear_constraint, nonlinear_constraint], options={'verbose': 1}, bounds=bounds)
	print('Solution =', res.x)

def model(x, u):
	return x[0] * (u**2 + x[1] * u) / (u**2 + x[2] * u + x[3])

def fun(x, u, y):
	return model(x, u) - y

def jac(x, u, y):
	J = np.empty((u.size, x.size))
	den = u**2 + x[2] * u + x[3]
	num = u**2 + x[1] * u
	J[:,0] = num / den
	J[:,1] = x[0] * u / den
	J[:,2] = -x[0] * num * u / den**2
	J[:,3] = -x[0] * num / den**2
	return J

def simple_least_squares_example():
	u = np.array([4.0, 2.0, 1.0, 5.0e-1, 2.5e-1, 1.67e-1, 1.25e-1, 1.0e-1, 8.33e-2, 7.14e-2, 6.25e-2])
	y = np.array([1.957e-1, 1.947e-1, 1.735e-1, 1.6e-1, 8.44e-2, 6.27e-2, 4.56e-2, 3.42e-2, 3.23e-2, 2.35e-2, 2.46e-2])
	x0 = np.array([2.5, 3.9, 4.15, 3.9])
	res = least_squares(fun, x0, jac=jac, bounds=(0, 100), args=(u, y), verbose=1)

	u_test = np.linspace(0, 5)
	y_test = model(res.x, u_test)
	plt.plot(u, y, 'o', markersize=4, label='data')
	plt.plot(u_test, y_test, label='fitted model')
	plt.xlabel('u')
	plt.ylabel('y')
	plt.legend(loc='lower right')
	plt.show()

def simple_univariate_minimization_example():
	# Unconstrained minimization.
	f = lambda x: (x - 2) * (x + 1)**2
	res = minimize_scalar(f, method='brent')
	print('Solution =', res.x)

	# Bounded minimization.
	res = minimize_scalar(j1, bounds=(4, 7), method='bounded')
	print('Solution =', res.x)

def custmin1(fun, x0, args=(), maxfev=None, stepsize=0.1, maxiter=100, callback=None, **options):
	bestx = x0
	besty = fun(x0)
	funcalls = 1
	niter = 0
	improved = True
	stop = False

	while improved and not stop and niter < maxiter:
		improved = False
		niter += 1
		for dim in range(np.size(x0)):
			for s in [bestx[dim] - stepsize, bestx[dim] + stepsize]:
				testx = np.copy(bestx)
				testx[dim] = s
				testy = fun(testx, *args)
				funcalls += 1
				if testy < besty:
					besty = testy
					bestx = testx
					improved = True
			if callback is not None:
				callback(bestx)
			if maxfev is not None and funcalls >= maxfev:
				stop = True
				break

	return OptimizeResult(fun=besty, x=bestx, nit=niter, nfev=funcalls, success=(niter > 1))

def custmin2(fun, bracket, args=(), maxfev=None, stepsize=0.1, maxiter=100, callback=None, **options):
	bestx = (bracket[1] + bracket[0]) / 2.0
	besty = fun(bestx)
	funcalls = 1
	niter = 0
	improved = True
	stop = False

	while improved and not stop and niter < maxiter:
		improved = False
		niter += 1
		for testx in [bestx - stepsize, bestx + stepsize]:
			testy = fun(testx, *args)
			funcalls += 1
			if testy < besty:
				besty = testy
				bestx = testx
				improved = True
		if callback is not None:
			callback(bestx)
		if maxfev is not None and funcalls >= maxfev:
			stop = True
			break

	return OptimizeResult(fun=besty, x=bestx, nit=niter, nfev=funcalls, success=(niter > 1))

def custom_minization_example():
	x0 = [1.35, 0.9, 0.8, 1.1, 1.2]
	res = minimize(rosen, x0, method=custmin1, options=dict(stepsize=0.05))
	print('Solution =', res.x)

	def f(x):
		return (x - 2)**2 * (x + 2)**2
	res = minimize_scalar(f, bracket=(-3.5, 0), method=custmin2, options=dict(stepsize = 0.05))
	print('Solution =', res.x)

def main():
	#simple_unconstrained_minimization_example()
	simple_constrained_minimization_example()

	#simple_least_squares_example()
	#simple_univariate_minimization_example()

	#custom_minization_example()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
