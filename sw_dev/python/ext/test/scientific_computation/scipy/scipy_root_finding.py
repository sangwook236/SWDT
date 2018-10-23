#!/usr/bin/env python

# REF [site] >>
#	https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html

from scipy.optimize import root
import matplotlib.pyplot as plt
import numpy as np

def simple_example():
	# Single-variable equation.
	def func(x):
		return x + 2 * np.cos(x)

	sol = root(func, 0.3)
	print('Solution =', sol.x)
	print('Function =', sol.fun)

	# Set of equations.
	def func2(x):
		f = [x[0] * np.cos(x[1]) - 4, x[1]*x[0] - x[1] - 5]
		df = np.array([[np.cos(x[1]), -x[0] * np.sin(x[1])], [x[1], x[0] - 1]])
		return f, df

	sol = root(func2, [1, 1], jac=True, method='lm')
	print('Solution =', sol.x)
	print('Function =', sol.fun)

def large_problem():
# Parameters.
	nx, ny = 75, 75
	hx, hy = 1. / (nx - 1), 1. / (ny - 1)

	P_left, P_right = 0, 0
	P_top, P_bottom = 1, 0

	def residual(P):
		d2x = np.zeros_like(P)
		d2y = np.zeros_like(P)

		d2x[1:-1] = (P[2:] - 2 * P[1:-1] + P[:-2]) / hx / hx
		d2x[0] = (P[1] - 2 * P[0] + P_left) / hx / hx
		d2x[-1] = (P_right - 2 * P[-1] + P[-2]) / hx / hx

		d2y[:,1:-1] = (P[:,2:] - 2 * P[:,1:-1] + P[:,:-2]) / hy / hy
		d2y[:,0] = (P[:,1] - 2 * P[:,0] + P_bottom) / hy / hy
		d2y[:,-1] = (P_top - 2 * P[:,-1] + P[:,-2]) / hy / hy

		return d2x + d2y + 5 * np.cosh(P).mean()**2

	# Solve.
	guess = np.zeros((nx, ny), float)
	sol = root(residual, guess, method='krylov', options={'disp': True})
	#sol = root(residual, guess, method='broyden2', options={'disp': True, 'max_rank': 50})
	#sol = root(residual, guess, method='anderson', options={'disp': True, 'M': 10})
	print('Residual: %g' % np.abs(residual(sol.x)).max())

	# Visualize.
	x, y = np.mgrid[0:1:(nx * 1j), 0:1:(ny * 1j)]
	plt.pcolor(x, y, sol.x)
	plt.colorbar()
	plt.show()

def main():
	#simple_example()
	large_problem()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
