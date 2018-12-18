#!/usr/bin/env python

# REF [site] >> https://docs.python.org/3.4/library/multiprocessing.html

import multiprocessing as mp
import os, time
import numpy as np

def process_info(title):
    print(title)
    print('\tmodule name:', __name__)
    print('\tparent process:', os.getppid())
    print('\tprocess id:', os.getpid())

def hello(name):
	process_info('function hello()')
	print('hello', name)

def simple_process():
	process_info('function process1()')
	p = mp.Process(target=hello, args=('bob',))
	p.start()
	p.join()

def foo(q):
	q.put('hello')

def process_with_context():
	# set_start_method() should not be used more than once in the program.
	#mp.set_start_method('spawn')
	#q = mp.Queue()
	#p = mp.Process(target=foo, args=(q,))
	# Context objects allow one to use multiple start methods in the same program.
	ctx = mp.get_context('spawn')
	q = ctx.Queue()
	p = ctx.Process(target=foo, args=(q,))

	p.start()
	print(q.get())
	p.join()

def sqr(x):
    return x*x

def pool_1():
	with mp.Pool(processes=5) as pool:
		print(pool.map(sqr, [item for item in range(10000)]))

		# In arbitrary order.
		for i in pool.imap_unordered(sqr, range(10)):
			print(i)

def loop_with_sleep(sec):
	time.sleep(sec)
	print('{}: sec={}'.format(os.getpid(), sec))

def pool_2():
	# set_start_method() should not be used more than once in the program.
	#mp.set_start_method('spawn')
	with mp.Pool(processes=5) as pool:
		pool.map(loop_with_sleep, [np.random.randint(6, 11) for _ in range(10)])
		# Async.
		#multiple_results = [pool.apply_async(loop_with_sleep, args=(np.random.randint(6, 11),)) for _ in range(10)]
		#[res.get() for res in multiple_results]

def sqr_with_sleep(x):
	time.sleep(2)
	return x*x

async_result_list = []
def async_callback(result):
	# This is called whenever sqr_with_sleep(i) returns a result.
	# async_result_list is modified only by the main process, not the pool workers.
	async_result_list.append(result)

def pool_async_1():
	with mp.Pool(processes=5) as pool:
		# Evaluate 'os.getpid()' asynchronously.
		res = pool.apply_async(os.getpid, args=())  # Runs in *only* one process.
		print(res.get(timeout=1))

		multiple_results = [pool.apply_async(os.getpid, args=()) for i in range(4)]
		print([res.get(timeout=1) for res in multiple_results])

		res = pool.apply_async(time.sleep, args=(10,))
		try:
			print(res.get(timeout=1))
		except mp.context.TimeoutError as ex:
			print('A multiprocessing.TimeoutError raised:', ex)

def pool_async_2():
	with mp.Pool() as pool:
		for i in range(10):
			pool.apply_async(sqr_with_sleep, args=(i,), callback=async_callback)
		pool.close()
		pool.join()
		print(async_result_list)

def main():
	#simple_process()
	#process_with_context()

	#pool_1()
	pool_2()
	#pool_async_1()
	#pool_async_2()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
