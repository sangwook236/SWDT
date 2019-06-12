#!/usr/bin/env python

# REF [site] >> https://github.com/ray-project/ray

import time
import ray
import tensorflow as tf

ray.init()

@ray.remote
def func():
	time.sleep(1)
	return 1

def simple_example():
	results = ray.get([func.remote() for i in range(4)])
	print('Result =', results)

@ray.remote
def sqr(x):
	time.sleep(1)
	return x * x

def parallelized_map(func, args):
    return list(func.remote(arg) for arg in args)

def parallelized_map_example():
	# Call parallelized_map() on a list.
	result_ids = parallelized_map(sqr, range(1, 6))

	# Get the results.
	results = ray.get(result_ids)
	print('Result =', results)

@ray.remote
def negative(x):
	time.sleep(1)
	return x < 0

def parallelized_filter(func, args):
    return list(arg for arg in args if func.remote(arg))

def parallelized_filter_example():
	# Call parallelized_filter() on a list.
	result_ids = parallelized_filter(negative, range(-5, 5))

	# Get the results.
	results = ray.get(result_ids)
	print('Result =', results)

@ray.remote
class Simulator(object):
	def __init__(self):
		self.sess = tf.Session()
		self.simple_model = tf.constant([1.0])

	def simulate(self):
		return self.sess.run(self.simple_model)

def simple_tensorflow_example():
	# Create two actors.
	simulators = [Simulator.remote() for _ in range(2)]

	# Run two simulations in parallel.
	results = ray.get([s.simulate.remote() for s in simulators])
	print(results)

def main():
	#simple_example()

	parallelized_map_example()
	#parallelized_filter_example()

	#simple_tensorflow_example()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
