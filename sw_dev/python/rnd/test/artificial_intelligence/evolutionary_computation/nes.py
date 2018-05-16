# REF [site] >> https://gist.github.com/karpathy

"""
A bare bones examples of optimizing a black-box function (f) using
Natural Evolution Strategies (NES), where the parameter distribution is a 
gaussian of fixed standard deviation.
"""

import numpy as np
np.random.seed(0)

# The function we want to optimize.
def f(w):
	# Here we would normally:
	# ... 1) create a neural network with weights w
	# ... 2) run the neural network on the environment for some time
	# ... 3) sum up and return the total reward

	# But for the purposes of an example, lets try to minimize
	# the L2 distance to a specific solution vector. So the highest reward
	# we can achieve is 0, when the vector w is exactly equal to solution
	reward = -np.sum(np.square(solution - w))
	return reward

# Hyperparameters.
npop = 50  # Population size.
sigma = 0.1  # Noise standard deviation.
alpha = 0.001  # Learning rate.

# Start the optimization.
solution = np.array([0.5, 0.1, -0.3])
w = np.random.randn(3)  # Our initial guess is random.
for i in range(300):
	# Print current fitness of the most likely parameter setting.
	if i % 20 == 0:
		print('iter %d. w: %s, solution: %s, reward: %f' % (i, str(w), str(solution), f(w)))

	# Initialize memory for a population of w's, and their rewards.
	N = np.random.randn(npop, 3)  # Samples from a normal distribution N(0,1).
	R = np.zeros(npop)
	for j in range(npop):
		w_try = w + sigma*N[j]  # Jitter w using gaussian of sigma 0.1.
		R[j] = f(w_try) # Evaluate the jittered version.

	# Standardize the rewards to have a gaussian distribution.
	A = (R - np.mean(R)) / np.std(R)
	# Perform the parameter update. The matrix multiply below
	# is just an efficient way to sum up all the rows of the noise matrix N,
	# where each row N[j] is weighted by A[j].
	w = w + alpha / (npop * sigma) * np.dot(N.T, A)
