#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import time
import numpy as np
import control as ct
import control.optimal as obc
import matplotlib.pyplot as plt

# REF [site] >> https://dynamics-and-control.readthedocs.io/en/latest/2_Control/1_Conventional_feedback_control/PID%20controller%20step%20responses.html
def pid_step_response_example():
	def plot_step_response(G):
		t, y = ct.step_response(G, ts)
		# Add some action before time zero so that the initial step is visible
		t = np.concatenate([[-1, 0], t])
		y = np.concatenate([[0, 0], y])
		plt.plot(t, y)

	s = ct.tf([1, 0], 1)
	ts = np.linspace(0, 5)

	#-----
	# PI

	K_C = 1
	tau_I = 1
	Gc = K_C * (1 + 1 / (tau_I * s))

	plot_step_response(Gc)

	#-----
	# PID

	# Because the ideal PID is unrealisable, we can't plot the response of the ideal PID, but we can do it for the realisable ISA PID
	alpha = 0.1
	tau_D = 1
	Gc = K_C * (1 + 1 / (tau_I * s) + 1 * s / (alpha * tau_D * s + 1))

	plot_step_response(Gc)

	#-----
	# PD

	Gc = K_C * (1 + 1 * s / (alpha * tau_D * s + 1))

	plot_step_response(Gc)

# REF [site] >> https://python-control.readthedocs.io/en/0.9.4/mpc_aircraft.html
def mpc_aircraft_example():
	# Model of an aircraft discretized with 0.2s sampling time
	# Source: https://www.mpt3.org/UI/RegulationProblem
	A = [[0.99, 0.01, 0.18, -0.09,   0],
		 [   0, 0.94,    0,  0.29,   0],
		 [   0, 0.14, 0.81,  -0.9,   0],
		 [   0, -0.2,    0,  0.95,   0],
		 [   0, 0.09,    0,     0, 0.9]]
	B = [[ 0.01, -0.02],
		 [-0.14,     0],
		 [ 0.05,  -0.2],
		 [ 0.02,     0],
		 [-0.01, 0]]
	C = [[0, 1, 0, 0, -1],
		 [0, 0, 1, 0,  0],
		 [0, 0, 0, 1,  0],
		 [1, 0, 0, 0,  0]]
	model = ct.ss2io(ct.ss(A, B, C, 0, 0.2))

	# For the simulation we need the full state output
	sys = ct.ss2io(ct.ss(A, B, np.eye(5), 0, 0.2))

	# Compute the steady state values for a particular value of the input
	ud = np.array([0.8, -0.3])
	xd = np.linalg.inv(np.eye(5) - A) @ B @ ud
	yd = C @ xd

	# Computed values will be used as references for the desired steady state which can be added using "reference" filter
	#model.u.with("reference");
	#model.u.reference = us;
	#model.y.with("reference");
	#model.y.reference = ys;

	# Provide constraints on the system signals
	constraints = [obc.input_range_constraint(sys, [-5, -6], [5, 6])]

	# Provide penalties on the system signals
	Q = model.C.transpose() @ np.diag([10, 10, 10, 10]) @ model.C
	R = np.diag([3, 2])
	cost = obc.quadratic_cost(model, Q, R, x0=xd, u0=ud)

	# Online MPC controller object is constructed with a horizon 6
	ctrl = obc.create_mpc_iosystem(model, np.arange(0, 6) * 0.2, cost, constraints)

	# Define an I/O system implementing model predictive control
	loop = ct.feedback(sys, ctrl, 1)
	print(loop)

	#loop = ClosedLoop(ctrl, model);
	#x0 = [0, 0, 0, 0, 0]
	Nsim = 60

	start = time.time()
	tout, xout = ct.input_output_response(loop, np.arange(0, Nsim) * 0.2, 0, 0)
	end = time.time()
	print("Computation time = %g seconds" % (end - start))

	# Plot the results
	#plt.subplot(2, 1, 1)
	for i, y in enumerate(C @ xout):
		plt.plot(tout, y)
		plt.plot(tout, yd[i] * np.ones(tout.shape), "k--")
	plt.title("outputs")

	#plt.subplot(2, 1, 2)
	#plt.plot(t, u);
	#plot(np.range(Nsim), us * ones(1, Nsim), "k--")
	#plt.title("inputs")

	plt.tight_layout()

	# Print the final error
	print(f"Final error = {xd - xout[:,-1]}.")

# REF [site] >> https://python-control.readthedocs.io/en/0.9.4/pvtol-lqr-nested.html
def pvtol_lqr_nested_example():
	import numpy as np  # Grab all of the NumPy functions
	from control.matlab import *  # MATLAB-like functions
	import matplotlib.pyplot as plt  # Grab MATLAB plotting functions

	# The parameters for the system
	m = 4  # Mass of aircraft
	J = 0.0475  # Inertia around pitch axis
	r = 0.25  # Distance to center of force
	g = 9.8  # Gravitational constant
	c = 0.05  # Damping factor (estimated)

	#-----
	# LQR state feedback controller

	# Choosing equilibrium inputs to be u_e = (0, mg), the dynamics of the system dz/dt, and their linearization A about equilibrium point z_e = (0, 0, 0, 0, 0, 0)
	# State space dynamics
	xe = [0, 0, 0, 0, 0, 0]  # Equilibrium point of interest
	ue = [0, m * g]  # Note these are lists, not matrices

	# Dynamics matrix (use matrix type so that * works for multiplication)
	# Note that we write A and B here in full generality in case we want to test different xe and ue.
	A = np.matrix(
		[[ 0,    0,    0,    1,    0,    0],
		 [ 0,    0,    0,    0,    1,    0],
		 [ 0,    0,    0,    0,    0,    1],
		 [ 0, 0, (-ue[0] * sin(xe[2]) - ue[1] * cos(xe[2])) / m, -c / m, 0, 0],
		 [ 0, 0, (ue[0] * cos(xe[2]) - ue[1] * sin(xe[2])) / m, 0, -c / m, 0],
		 [ 0,    0,    0,    0,    0,    0 ]]
	)

	# Input matrix
	B = np.matrix(
		[[0, 0], [0, 0], [0, 0],
		 [cos(xe[2]) / m, -sin(xe[2]) / m],
		 [sin(xe[2]) / m,  cos(xe[2]) / m],
		 [r / J, 0]]
	)

	# Output matrix
	C = np.matrix([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])
	D = np.matrix([[0, 0], [0, 0]])

	# Compute a linear quadratic regulator for the system
	Qx1 = np.diag([1, 1, 1, 1, 1, 1])
	Qu1a = np.diag([1, 1])
	(K, X, E) = lqr(A, B, Qx1, Qu1a)
	K1a = np.matrix(K)

	# Closed loop system
	# Our input to the system will only be (x_d, y_d), so we need to multiply it by this matrix to turn it into z_d
	Xd = np.matrix([[1, 0, 0, 0, 0, 0],
					[0, 1, 0, 0, 0, 0]]).T

	# Closed loop dynamics
	H = ss(A - B * K, B * K * Xd, C, D)

	# Step response for the first input
	x, t = step(H, input=0, output=0, T=linspace(0, 10, 100))
	# Step response for the second input
	y, t = step(H, input=1, output=1, T=linspace(0, 10, 100))

	plt.plot(t, x, "-", t, y, "--")
	plt.plot([0, 10], [1, 1], "k-")
	plt.ylabel("Position")
	plt.xlabel("Time (s)")
	plt.title("Step Response for Inputs")
	plt.legend(("Yx", "Yy"), loc="lower right")
	plt.show()

	#-----
	# Lateral control using inner/outer loop design

	# Transfer functions for dynamics
	Pi = tf([r], [J, 0, 0])  # Inner loop (roll)
	Po = tf([1], [m, c, 0])  # Outer loop (position)

	# For the inner loop, use a lead compensator
	k = 200
	a = 2
	b = 50
	Ci = k * tf([1, a], [1, b])  # Lead compensator
	Li = Pi * Ci

	# The closed loop dynamics of the inner loop, H_i
	Hi = parallel(feedback(Ci, Pi), -m * g * feedback(Ci * Pi, 1))

	# Design the lateral compensator using another lead compenstor
	# Now design the lateral control system
	a = 0.02
	b = 5
	K = 2
	Co = -K * tf([1, 0.3], [1, 10])  # Another lead compensator
	Lo = -m * g * Po * Co

	# The performance of the system can be characterized using the sensitivity function and the complementary sensitivity function
	L = Co * Hi * Po
	S = feedback(1, L)
	T = feedback(L, 1)

	t, y = step(T, T=linspace(0, 10, 100))
	plt.plot(y, t)
	plt.title("Step Response")
	plt.grid()
	plt.xlabel("time (s)")
	plt.ylabel("y(t)")
	plt.show()

	# The frequency response and Nyquist plot for the loop transfer function are computed using the commands
	bode(L)
	plt.show()

	nyquist(L, (0.0001, 1000))
	plt.show()

	gangof4(Hi * Po, Co)

def main():
	# Proportional–integral–derivative (PID) control
	pid_step_response_example()

	# Model predictive control (MPC), receding horizon control (RHC)
	mpc_aircraft_example()

	# Linear-quadratic regulator (LQR)
	pvtol_lqr_nested_example()

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
