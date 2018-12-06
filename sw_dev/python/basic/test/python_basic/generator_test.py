#!/usr/bin/env python

# REF [site] >> https://wiki.python.org/moin/Generators

import time

# Build and return a list.
def firstn_1(n):
	num, nums = 0, list()
	while num < n:
		nums.append(num)
		num += 1
	return nums

# Using the generator pattern (an iterable).
class firstn_2(object):
	def __init__(self, n):
		self.n = n
		self.num, self.nums = 0, list()

	def __iter__(self):
		return self

	# Python 3 compatibility.
	def __next__(self):
		return self.next()

	def next(self):
		if self.num < self.n:
			cur, self.num = self.num, self.num + 1
			return cur
		else:
			raise StopIteration()

# A generator that yields items instead of returning a list.
def firstn_3(n):
	num = 0
	while num < n:
		yield num
		num += 1

# Generators are iterators, but you can only iterate over them once.
# It's because they do not store all the values in memory, they generate the values on the fly.
def simple_generator_example():
	start = time.time()
	firstn_1_retval = firstn_1(1000000)
	sum_of_first_n_1 = sum(firstn_1_retval)
	print('Elapsed time = {}'.format(time.time() - start))
	print('type(firstn_1_retval) =', type(firstn_1_retval))
	print('sum_of_first_n_1 =', sum_of_first_n_1)

	start = time.time()
	firstn_2_retval = firstn_2(1000000)
	sum_of_first_n_2 = sum(firstn_2_retval)
	print('Elapsed time = {}'.format(time.time() - start))
	print('type(firstn_2_retval) =', type(firstn_2_retval))
	print('sum_of_first_n_2 =', sum_of_first_n_2)

	start = time.time()
	firstn_3_retval = firstn_3(1000000)
	sum_of_first_n_3 = sum(firstn_3_retval)
	print('Elapsed time = {}'.format(time.time() - start))
	print('type(firstn_3_retval) =', type(firstn_3_retval))
	print('sum_of_first_n_3 =', sum_of_first_n_3)

# yield is a keyword that is used like return, except the function will return a generator.
#	REF [site] >> https://pythontips.com/2013/09/29/the-python-yield-keyword-explained/
def simple_yield_example():
	def createGenerator():
		mylist = range(3)
		for i in mylist:
			yield i * i

	gen = createGenerator()
	print('type(gen) =', type(gen), gen)
	for i in gen:
		print(i)

def main():
	simple_generator_example()
	simple_yield_example()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
