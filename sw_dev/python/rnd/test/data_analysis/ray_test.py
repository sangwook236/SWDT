#!/usr/bin/env python

# REF [site] >> https://github.com/ray-project/ray

import time
import ray
import tensorflow as tf

ray.init()

@ray.remote
def f():
	time.sleep(1)
	return 1

def simple_example():
	results = ray.get([f.remote() for i in range(4)])
	print(results)

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
	simple_example()

	simple_tensorflow_example()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
