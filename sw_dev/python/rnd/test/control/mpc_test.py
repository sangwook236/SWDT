#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://dynamics-and-control.readthedocs.io/en/latest/2_Control/7_Multivariable_control/Simple%20MPC.html
def simple_mpc_example():
	import numpy as np
	import scipy.signal, scipy.optimize
	import matplotlib.pyplot as plt

	# A linear model of the system: G = 1 / (15 s^2 + 8 s + 1)
	G = scipy.signal.lti([1], [15, 8, 1])

	plt.plot(*G.step())

	# Controller parameters
	M = 10  # Control horizon
	P = 20  # Prediction horizon
	DeltaT = 1  # Sampling rate

	tcontinuous = np.linspace(0, P * DeltaT, 1000)  # Some closely spaced time points
	tpredict = np.arange(0, P * DeltaT, DeltaT)  # Discrete points at prediction horizon

	# Choose a first order setpoint response
	tau_c = 1
	r = 1 - np.exp(-tpredict / tau_c)

	# For an initial guess
	u = np.ones(M)

	# Initital state is zero
	x0 = np.zeros(G.to_ss().A.shape[0])

	def extend(u):
		"""We optimise the first M values of u but we need P values for prediction"""
		return np.concatenate([u, np.repeat(u[-1], P - M)])

	def prediction(u, t=tpredict, x0=x0):
		"""Predict the effect of an input signal"""
		t, y, x = scipy.signal.lsim(G, u, t, X0=x0, interp=False)
		return y

	# One of the reasons for the popularity of MPC is how easy it is to change its behaviour using weights in the objective function.
	if True:
		def objective(u, x0=x0):
			"""Calculate the sum of the square error for the cotnrol problem"""
			y = prediction(extend(u))
			return sum((r - y)**2)
	else:
		# Try using this definition instead of the simple one above and see if you can remove the ringing in the controller output.
		def objective(u, x0=x0):
			y = prediction(extend(u))
			u_mag = np.abs(u)
			constraintpenalty = sum(u_mag[u_mag > 2])
			movepenalty = sum(np.abs(np.diff(u)))
			strongfinish = np.abs(y[-1] - r[-1])
			return sum((r - y)**2) + 0 * constraintpenalty + 0.1 * movepenalty + 0 * strongfinish

	plt.plot(tpredict, prediction(extend(u)))

	print(f"{objective(u)=}.")

	# Figure out a set of moves which will minimise our objective function
	result = scipy.optimize.minimize(objective, u)
	u_opt = result.x
	print(f"{result.fun=}.")

	# Resample the discrete output to continuous time (effectively work out the 0 order hold value)
	u_cont = extend(u_opt)[((tcontinuous - 0.01) // DeltaT).astype(int)]

	# Plot the move plan and the output.
	# Notice that we are getting exactly the output we want at the sampling times.
	# At this point we have effectively recovered the Dahlin controller.

	def plot_output(u_cont, u_opt):
		plt.figure()
		plt.plot(tcontinuous, u_cont)
		plt.xlim([0, DeltaT * (P + 1)])

		plt.figure()
		plt.plot(tcontinuous, prediction(u_cont, tcontinuous), label="Continuous response")
		plt.plot(tpredict, prediction(extend(u_opt)), "-o", label="Optimized response")
		plt.plot(tpredict, r, label="Set point")
		plt.legend()

	plot_output(u_cont, u_opt)

def main():
	# Model predictive control (MPC), receding horizon control (RHC).
	simple_mpc_example()

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
