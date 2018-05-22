#!/usr/bin/env python

# REF [site] >> https://docs.python.org/3.4/library/multiprocessing.html

import multiprocessing as mp
import os, time

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

def pool():
	with mp.Pool(processes=5) as pool:
		print(pool.map(sqr, [item for item in range(10000)]))

def sqr_with_sleep(x):
	time.sleep(2)
	return x*x

async_result_list = []
def async_callback(result):
	# This is called whenever sqr_with_sleep(i) returns a result.
	# async_result_list is modified only by the main process, not the pool workers.
	async_result_list.append(result)

def pool_async():
	with mp.Pool() as pool:
		for i in range(10):
			pool.apply_async(sqr_with_sleep, args = (i,), callback = async_callback)
		pool.close()
		pool.join()
		print(async_result_list)

def main():
	#simple_process()
	#process_with_context()

	#pool()
	pool_async()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
