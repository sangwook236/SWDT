#!/usr/bin/env python

# REF [site] >> https://docs.python.org/3.4/library/multiprocessing.html

import os, time
from functools import partial
import multiprocessing as mp
from multiprocessing.managers import BaseManager
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

def func1(lock, i):
	lock.acquire()
	try:
		print('hello world', i)
	finally:
		lock.release()

# REF [site] >> https://docs.python.org/3.7/library/multiprocessing.html
def synchronization():
	lock = mp.Lock()

	for num in range(10):
		mp.Process(target=func1, args=(lock, num)).start()

def func2(n, a):
	n.value = 3.1415927
	for i in range(len(a)):
		a[i] = -a[i]

# REF [site] >> https://docs.python.org/3.7/library/multiprocessing.html
def state_sharing_by_shared_memory():
	num = mp.Value('d', 0.0)
	arr = mp.Array('i', range(10))

	p = mp.Process(target=func2, args=(num, arr))
	p.start()
	p.join()

	print(num.value)
	print(arr[:])

def func3(d, l):
	d[1] = '1'
	d['2'] = 2
	d[0.25] = None
	l.reverse()

# REF [site] >> https://docs.python.org/3.7/library/multiprocessing.html
def state_sharing_by_server_process():
	with mp.Manager() as manager:
		d = manager.dict()
		l = manager.list(range(10))

		p = mp.Process(target=func3, args=(d, l))
		p.start()
		p.join()

		print(d)
		print(l)

def init(lock):
	global global_lock
	global_lock = lock

class Adder(object):
	def __init__(self, offset):
		self._offset = offset

	def process(self, num):
		return num + self._offset

def worker0_func(args):
	ii, ff, ss, job = args

	worker_id = 0
	#logger = logging.getLogger('python_mp_logging_test')
	#logger.info('\t{}: Start worker{}(job{}) process.'.format(os.getpid(), worker_id, job))
	print('\t{}: Start worker{}(job{}) process.'.format(os.getpid(), worker_id, job))

	#--------------------
	# Do something.
	for loop in range(3):
		with global_lock:
			print('\t{}: Do something by worker{}(job{}, loop{}): {}, {}, {}.'.format(os.getpid(), worker_id, job, loop, ii, ff, ss))
		time.sleep(np.random.randint(1, 4))

	print('\t{}: End worker{}(job{}) process.'.format(os.getpid(), worker_id, job))

def worker1_func(ii, ff, ss, job):
	worker_id = 1
	#logger = logging.getLogger('python_mp_logging_test')
	#logger.exception('\t{}: Start worker{}(job{}) process.'.format(os.getpid(), worker_id, job))
	print('\t{}: Start worker{}(job{}) process.'.format(os.getpid(), worker_id, job))

	#--------------------
	# Do something.
	for loop in range(3):
		with global_lock:
			print('\t{}: Do something by worker{}(job{}, loop{}): {}, {}, {}.'.format(os.getpid(), worker_id, job, loop, ii, ff, ss))
		time.sleep(np.random.randint(1, 4))

	print('\t{}: End worker{}(job{}) process.'.format(os.getpid(), worker_id, job))

def worker2_func(ii, ff, ss, num_jobs):
	worker_id = 2
	for job in range(num_jobs):
		#logger = logging.getLogger('python_mp_logging_test')
		#logger.exception('\t{}: Start worker{}(job{}) process.'.format(os.getpid(), worker_id, job))
		print('\t{}: Start worker{}(job{}) process.'.format(os.getpid(), worker_id, job))

		#--------------------
		# Do something.
		for loop in range(3):
			with global_lock:
				print('\t{}: Do something by worker{}(job{}, loop{}): {}, {}, {}.'.format(os.getpid(), worker_id, job, loop, ii, ff, ss))
			time.sleep(np.random.randint(1, 4))

		print('\t{}: End worker{}(job{}) process.'.format(os.getpid(), worker_id, job))

def worker3_func(lock, ii, ff, ss, job):
	worker_id = 3
	#logger = logging.getLogger('python_mp_logging_test')
	#logger.exception('\t{}: Start worker{}(job{}) process.'.format(os.getpid(), worker_id, job))
	print('\t{}: Start worker{}(job{}) process.'.format(os.getpid(), worker_id, job))

	#--------------------
	# Do something.
	for loop in range(3):
		with lock:
			print('\t{}: Do something by worker{}(job{}, loop{}): {}, {}, {}.'.format(os.getpid(), worker_id, job, loop, ii, ff, ss))
		time.sleep(np.random.randint(1, 4))

	print('\t{}: End worker{}(job{}) process.'.format(os.getpid(), worker_id, job))

def worker4_func(args):
	adder, ii, ff, ss, job = args

	worker_id = 4
	#logger = logging.getLogger('python_mp_logging_test')
	#logger.exception('\t{}: Start worker{}(job{}) process.'.format(os.getpid(), worker_id, job))
	print('\t{}: Start worker{}(job{}) process.'.format(os.getpid(), worker_id, job))

	#--------------------
	# Do something.
	for loop in range(3):
		with global_lock:
			print('\t{}: Do something by worker{}(job{}, loop{}): {}, {}, {}, {}.'.format(os.getpid(), worker_id, job, loop, ii, ff, ss, adder.process(ii)))
		time.sleep(np.random.randint(1, 4))

	print('\t{}: End worker{}(job{}) process.'.format(os.getpid(), worker_id, job))

def worker5_func(adder, ii, ff, ss, job):
	worker_id = 5
	#logger = logging.getLogger('python_mp_logging_test')
	#logger.exception('\t{}: Start worker{}(job{}) process.'.format(os.getpid(), worker_id, job))
	print('\t{}: Start worker{}(job{}) process.'.format(os.getpid(), worker_id, job))

	#--------------------
	# Do something.
	for loop in range(3):
		with global_lock:
			print('\t{}: Do something by worker{}(job{}, loop{}): {}, {}, {}, {}.'.format(os.getpid(), worker_id, job, loop, ii, ff, ss, adder.process(ii)))
		time.sleep(np.random.randint(1, 4))

	print('\t{}: End worker{}(job{}) process.'.format(os.getpid(), worker_id, job))

def run_worker_processes(lock, num_processes):
	adder = Adder(2)
	"""
	# Passes objects between processes.
	#	REF [site] >> https://stackoverflow.com/questions/3671666/sharing-a-complex-object-between-python-processes
	BaseManager.register('Adder', Adder)
	manager = BaseManager()
	manager.start()

	adder = manager.Adder(2)
	"""

	# set_start_method() should not be used more than once in the program.
	#mp.set_start_method('spawn')

	#timeout = 10
	timeout = None
	with mp.Pool(processes=num_processes, initializer=init, initargs=(lock,)) as pool:
		num_jobs = np.random.randint(2, 8)
		worker0_results = pool.map_async(worker0_func, [(0, 0.0, 'a0', job) for job in range(num_jobs)])
		num_jobs = np.random.randint(2, 8)
		worker1_results = pool.map_async(partial(worker1_func, 1, 1.0, 'b1'), [job for job in range(num_jobs)])
		num_jobs = np.random.randint(2, 8)
		worker2_results = pool.apply_async(worker2_func, args=(2, 2.0, 'c2', num_jobs))
		# Passes a mp.Lock object as an argument.
		#num_jobs = np.random.randint(2, 8)
		#worker3_results = pool.map_async(partial(worker3_func, lock, 3, 3.0, 'd3'), [job for job in range(num_jobs)])  # RuntimeError: Lock objects should only be shared between processes through inheritance.
		# Passes a Class object as an argument.
		num_jobs = np.random.randint(2, 8)
		worker4_results = pool.map_async(worker4_func, [(adder, 4, 4.0, 'e4', job) for job in range(num_jobs)])
		num_jobs = np.random.randint(2, 8)
		worker5_results = pool.map_async(partial(worker5_func, adder, 5, 5.0, 'f5'), [job for job in range(num_jobs)])

		worker0_results.get(timeout)
		worker1_results.get(timeout)
		worker2_results.get(timeout)
		#worker3_results.get(timeout)
		worker4_results.get(timeout)
		worker5_results.get(timeout)

def main():
	#simple_process()
	#process_with_context()

	#pool_1()
	#pool_2()
	#pool_async_1()
	#pool_async_2()

	#synchronization()

	#state_sharing_by_shared_memory()
	#state_sharing_by_server_process()

	#--------------------
	num_processes = 10
	lock = mp.Lock()  # RuntimeError: Lock objects should only be shared between processes through inheritance.
	#lock= mp.Manager().Lock()  # TypeError: can't pickle _thread.lock objects.

	run_worker_processes(lock, num_processes)

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
