#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://stackoverflow.com/questions/56636040/optimal-control-using-scipy-optimize
def simple_mpc_example_1():
	import numpy as np
	import sympy as sp
	import scipy.optimize as opt
	import matplotlib.pyplot as plt

	# Optimal control problem
	#	Cost, J = x.transpose() * Q * x + u.transpose() * R * u
	#		x_dot = A * x + B * u
	#		x_min < x < x_max
	#		u_min < x < u_max

	class mpc_opt():
		def __init__(self):
			self.Q = sp.diag(0.5, 1, 0)  # State penalty matrix, Q
			self.R = sp.eye(2)  # Input penalty matrix, R
			self.A = sp.Matrix([[-0.79, -0.3, -0.1], [0.5, 0.82, 1.23], [0.52, -0.3, -0.5]])  # State matrix 
			self.B = sp.Matrix([[-2.04, -0.21], [-1.28, 2.75], [0.29, -1.41]])  # Input matrix

			self.t = np.linspace(0, 1, 30)

			self.omega = 0.2

		# Reference trajectory
		def ref_trajectory(self, i):
			# x = 3 * sin(2 * pi * omega * t)
			x = 3 * np.sin(2 * np.pi * self.omega * self.t[i])
			return sp.Matrix(([[self.t[i]], [x], [0]]))
			#x_ref = sp.Matrix([0, 1, 0])  # Static
			#return x_ref

		def cost_function(self, U, *args):
			t = args
			nx, nu = self.A.shape[-1], self.B.shape[-1]
			x0 = U[0:nx]
			u = U[nx:nx+nu]
			u = u.reshape(len(u), -1)
			x0 = x0.reshape(len(x0), -1)
			x1 = self.A * x0 + self.B * u
			#q = [x1[0], x1[1]]
			#pos = self.end_effec_pose(q)
			traj_ref = self.ref_trajectory(t)
			pos_error = x1 - traj_ref
			cost = pos_error.transpose() * self.Q * pos_error + u.transpose() * self.R * u
			return cost

		def cost_gradient(self, U, *args):
			t = args
			nx, nu = self.A.shape[-1], self.B.shape[-1]
			x0 = U[0:nx]
			u = U[nx:nx + nu]
			u = u.reshape(len(u), -1)
			x0 = x0.reshape(len(x0), -1)
			x1 = self.A * x0 + self.B * u
			traj_ref = self.ref_trajectory(t)
			pos_error = x1 - traj_ref
			temp1 = self.Q * pos_error
			cost_gradient = temp1.col_join(self.R * u)
			return cost_gradient

		def optimise(self, u0, t):
			umin = [-2., -3.]
			umax = [2., 3.]
			xmin = [-10., -9., -8.]
			xmax = [10., 9., 8.]
			bounds = ((xmin[0], xmax[0]), (xmin[1], xmax[1]), (xmin[2], xmax[2]), (umin[0], umax[0]), (umin[1], umax[1]))

			U = opt.minimize(
				self.cost_function, u0, args=(t),
				method="SLSQP", bounds=bounds, jac=self.cost_gradient,
				options={"maxiter": 200, "disp": True},
			)
			U = U.x
			return U

	mpc = mpc_opt()

	x0, u0, = sp.Matrix([[0.1], [0.02], [0.05]]), sp.Matrix([[0.4], [0.2]])
	X, U = sp.zeros(len(x0), len(mpc.t)), sp.zeros(len(u0), len(mpc.t))
	U0 = sp.Matrix([x0, u0])
	nx, nu = mpc.A.shape[-1], mpc.B.shape[-1]
	for i in range(len(mpc.t)):
		print(f"i = {i}.")
		result = mpc.optimise(U0, i)
		x0 = result[0:nx]
		u = result[nx:nx + nu]
		u = u.reshape(len(u), -1)
		x0 = x0.reshape(len(x0), -1)
		U[:, i], X[:, i] = u0, x0
		# x0 = mpc.A * x0 + mpc.B * u
		U0 = result

	X, U = np.asarray(X), np.asarray(U)
	plt.plot(X[0, :], "--r")
	plt.plot(X[1, :], "--b")
	plt.plot(X[2, :], "*r")
	plt.show()

# REF [site] >> https://dynamics-and-control.readthedocs.io/en/latest/2_Control/7_Multivariable_control/Simple%20MPC.html
def simple_mpc_example_2():
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
	# Model predictive control (MPC), receding horizon control (RHC)
	#	REF [project] >> ${SWDT_CPP_HOME}/rnd/test/control_theory.

	simple_mpc_example_1()
	#simple_mpc_example_2()

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
