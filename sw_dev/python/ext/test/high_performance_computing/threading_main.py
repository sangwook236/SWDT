#!/usr/bin/env python

# REF [site] >> https://docs.python.org/3/library/threading.html

import time
import threading
from multiprocessing.pool import ThreadPool
from concurrent.futures import ThreadPoolExecutor

def worker_proc():
	print('\t{}: Start worker thread.'.format(threading.get_ident()))
	for i in range(1, 11):
		time.sleep(1)
		print(i)
	print('\t{}: End worker thread.'.format(threading.get_ident()))

def simple_example_1():
	worker_thread = threading.Thread(target=worker_proc)
	worker_thread.start()
	worker_thread.join()

class MyThread(threading.Thread):
	def run(self):
		print("\t{}: Start Thread '{}'.".format(self.ident, self.getName()))
		time.sleep(1)
		print("\t{}: End Thread '{}'.".format(self.ident, self.getName()))

def simple_example_2():
	for x in range(4):
		mythread = MyThread(name='Thread-{}'.format(x + 1))
		mythread.start()
		time.sleep(0.5)

def sqr(x):
	return x * x

def thread_pool_example_1():
	with ThreadPool(processes=5, initializer=None) as pool:
		print(pool.map(sqr, [item for item in range(10000)]))

def thread_pool_example_2():
	with ThreadPoolExecutor(max_workers=3) as executor:
		executor.submit(worker_proc)
		executor.submit(worker_proc)
		executor.submit(worker_proc)

	with ThreadPoolExecutor(max_workers=10) as executor:
		future_gen = executor.map(sqr, [item for item in range(10000)])

		results = [future for future in future_gen]
		print(results)

def main():
	print('{}: Start main thread.'.format(threading.get_ident()))

	#simple_example_1()
	#simple_example_2()

	#thread_pool_example_1()
	thread_pool_example_2()

	print('{}: End main thread.'.format(threading.get_ident()))

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
