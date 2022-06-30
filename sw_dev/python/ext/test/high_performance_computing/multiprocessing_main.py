#!/usr/bin/env python

# REF [site] >> https://docs.python.org/3/library/multiprocessing.html

import os, time, math, random, functools, threading
import multiprocessing as mp
from multiprocessing.managers import BaseManager
from concurrent.futures import ProcessPoolExecutor

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
	return x * x

def pool_1():
	with mp.Pool(processes=5, initializer=None) as pool:
	#with mp.pool.ThreadPool(processes=5, initializer=None) as pool:
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
	with mp.Pool(processes=5, initializer=None) as pool:
	#with mp.pool.ThreadPool(processes=5, initializer=None) as pool:
		pool.map(loop_with_sleep, [random.randint(6, 10) for _ in range(10)])
		# Async.
		#multiple_results = [pool.apply_async(loop_with_sleep, args=(random.randint(6, 10),)) for _ in range(10)]
		#[res.get() for res in multiple_results]

def pool_async_1():
	with mp.Pool(processes=5, initializer=None) as pool:
	#with mp.pool.ThreadPool(processes=5, initializer=None) as pool:
		# Evaluate 'os.getpid()' asynchronously.
		res = pool.apply_async(os.getpid, args=())  # Runs in *only* one process.
		print(res.get(timeout=1))

		multiple_results = [pool.apply_async(os.getpid, args=()) for _ in range(4)]
		print([res.get(timeout=1) for res in multiple_results])

		res = pool.apply_async(time.sleep, args=(10,))
		try:
			print(res.get(timeout=1))
		except mp.context.TimeoutError as ex:
			print('A multiprocessing.TimeoutError raised:', ex)

def sqr_with_sleep(x):
	time.sleep(2)
	return x * x

def pool_async_2():
	async_results = list()
	def async_callback(result):
		# This is called whenever sqr_with_sleep(i) returns a result.
		# async_results is modified only by the main process, not the pool workers.
		async_results.append(result)

	with mp.Pool() as pool:
	#with mp.pool.ThreadPool() as pool:
		for i in range(10):
			#pool.apply_async(sqr_with_sleep, args=(i,))
			pool.apply_async(sqr_with_sleep, args=(i,), callback=async_callback)
		pool.close()  # Necessary. Why?
		pool.join()

	print('Async results =', async_results)

def simple_worker_proc(val, step):
	print('{}: step = {}.'.format(os.getpid(), step))

	time.sleep(random.randrange(3))
	val += step
	time.sleep(random.randrange(3))
	return step, val

def pool_async_3():
	async_results = list()
	def async_callback(result):
		# This is called whenever sqr_with_sleep(i) returns a result.
		# async_results is modified only by the main process, not the pool workers.
		async_results.append(result)

	num_processes = 5
	num_steps = 20

	#lock = mp.Lock()  # RuntimeError: Lock objects should only be shared between processes through inheritance.
	##lock= mp.Manager().Lock()  # TypeError: can't pickle _thread.lock objects.

	#timeout = 10
	timeout = None
	#with mp.Pool(processes=num_processes, initializer=initialize_lock, initargs=(lock,)) as pool:
	with mp.Pool(processes=num_processes) as pool:
		#results = pool.map_async(functools.partial(simple_worker_proc, 37), [step for step in range(num_steps)])
		results = pool.map_async(functools.partial(simple_worker_proc, 37), [step for step in range(num_steps)], callback=async_callback)

		results.get(timeout)

	print('Async results =', async_results)

def func1(lock, i):
	"""
	lock.acquire()
	try:
		print('hello world', i)
	finally:
		lock.release()
	"""
	with lock:
		print('hello world', i)

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

def is_prime(n):
	if n % 2 == 0:
		return False

	sqrt_n = int(math.floor(math.sqrt(n)))
	for i in range(3, sqrt_n + 1, 2):
		if n % i == 0:
			return False
	return True

# REF [site] >> https://docs.python.org/3/library/concurrent.futures.html
def process_pool_executor():
	PRIMES = [
		112272535095293,
		112582705942171,
		112272535095293,
		115280095190773,
		115797848077099,
		1099726899285419
	]

	with ProcessPoolExecutor(max_workers=3) as executor:
		for number, prime in zip(PRIMES, executor.map(is_prime, PRIMES)):
			print('%d is prime: %s' % (number, prime))

def worker_thread_proc(sec):
	print('\tThread {}: Start worker thread.'.format(threading.get_ident()))
	for i in range(1, 11):
		print('\tThread {}: Do something.'.format(threading.get_ident()))
		time.sleep(sec)
	print('\tThread {}: End worker thread.'.format(threading.get_ident()))

def worker_process_proc(sec):
	print('\tProcess {}: Start worker process.'.format(os.getpid()))
	for i in range(1, 4):
		print('\tProcess {}: Do something.'.format(os.getpid()))
		time.sleep(sec)
	print('\tProcess {}: End worker process.'.format(os.getpid()))

def multiprocess_and_multithreading():
	num_processes = 5
	num_threads = 2

	# Run worker threads.
	worker_thread = threading.Thread(target=worker_thread_proc, args=(2,))
	worker_thread.start()

	# Run worker processes.
	#timeout = 10
	timeout = None
	with mp.Pool(processes=num_processes) as pool:
	#with mp.pool.ThreadPool(processes=num_processes) as pool:
		worker_results = pool.map_async(worker_process_proc, [random.randint(1, 3) for _ in range(10)])

		worker_results.get(timeout)

	worker_thread.join()

def initialize_lock(lock):
	global global_lock
	global_lock = lock

class Adder(object):
	def __init__(self, offset):
		self._offset = offset

	def process(self, num):
		return num + self._offset

def worker0_proc(args):
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
		time.sleep(random.randint(1, 3))

	print('\t{}: End worker{}(job{}) process.'.format(os.getpid(), worker_id, job))

def worker1_proc(ii, ff, ss, job):
	worker_id = 1
	#logger = logging.getLogger('python_mp_logging_test')
	#logger.exception('\t{}: Start worker{}(job{}) process.'.format(os.getpid(), worker_id, job))
	print('\t{}: Start worker{}(job{}) process.'.format(os.getpid(), worker_id, job))

	#--------------------
	# Do something.
	for loop in range(3):
		with global_lock:
			print('\t{}: Do something by worker{}(job{}, loop{}): {}, {}, {}.'.format(os.getpid(), worker_id, job, loop, ii, ff, ss))
		time.sleep(random.randint(1, 3))

	print('\t{}: End worker{}(job{}) process.'.format(os.getpid(), worker_id, job))

def worker2_proc(ii, ff, ss, num_jobs):
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
			time.sleep(random.randint(1, 3))

		print('\t{}: End worker{}(job{}) process.'.format(os.getpid(), worker_id, job))

def worker3_proc(lock, ii, ff, ss, job):
	worker_id = 3
	#logger = logging.getLogger('python_mp_logging_test')
	#logger.exception('\t{}: Start worker{}(job{}) process.'.format(os.getpid(), worker_id, job))
	print('\t{}: Start worker{}(job{}) process.'.format(os.getpid(), worker_id, job))

	#--------------------
	# Do something.
	for loop in range(3):
		with lock:
			print('\t{}: Do something by worker{}(job{}, loop{}): {}, {}, {}.'.format(os.getpid(), worker_id, job, loop, ii, ff, ss))
		time.sleep(random.randint(1, 3))

	print('\t{}: End worker{}(job{}) process.'.format(os.getpid(), worker_id, job))

def worker4_proc(args):
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
		time.sleep(random.randint(1, 3))

	print('\t{}: End worker{}(job{}) process.'.format(os.getpid(), worker_id, job))

def worker5_proc(adder, ii, ff, ss, job):
	worker_id = 5
	#logger = logging.getLogger('python_mp_logging_test')
	#logger.exception('\t{}: Start worker{}(job{}) process.'.format(os.getpid(), worker_id, job))
	print('\t{}: Start worker{}(job{}) process.'.format(os.getpid(), worker_id, job))

	#--------------------
	# Do something.
	for loop in range(3):
		with global_lock:
			print('\t{}: Do something by worker{}(job{}, loop{}): {}, {}, {}, {}.'.format(os.getpid(), worker_id, job, loop, ii, ff, ss, adder.process(ii)))
		time.sleep(random.randint(1, 3))

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
	with mp.Pool(processes=num_processes, initializer=initialize_lock, initargs=(lock,)) as pool:
	#with mp.pool.ThreadPool(processes=num_processes, initializer=initialize_lock, initargs=(lock,)) as pool:
		num_jobs = random.randint(2, 7)
		worker0_results = pool.map_async(worker0_proc, [(0, 0.0, 'a0', job) for job in range(num_jobs)])
		num_jobs = random.randint(2, 7)
		worker1_results = pool.map_async(functools.partial(worker1_proc, 1, 1.0, 'b1'), [job for job in range(num_jobs)])
		num_jobs = random.randint(2, 7)
		worker2_results = pool.apply_async(worker2_proc, args=(2, 2.0, 'c2', num_jobs))
		# Passes a mp.Lock object as an argument.
		#num_jobs = random.randint(2, 7)
		#worker3_results = pool.map_async(functools.partial(worker3_proc, lock, 3, 3.0, 'd3'), [job for job in range(num_jobs)])  # RuntimeError: Lock objects should only be shared between processes through inheritance.
		# Passes a Class object as an argument.
		num_jobs = random.randint(2, 7)
		worker4_results = pool.map_async(worker4_proc, [(adder, 4, 4.0, 'e4', job) for job in range(num_jobs)])
		num_jobs = random.randint(2, 7)
		worker5_results = pool.map_async(functools.partial(worker5_proc, adder, 5, 5.0, 'f5'), [job for job in range(num_jobs)])

		worker0_results.get(timeout)
		worker1_results.get(timeout)
		worker2_results.get(timeout)
		#worker3_results.get(timeout)
		worker4_results.get(timeout)
		worker5_results.get(timeout)

def main():
	print('{}: Start main process.'.format(os.getpid()))

	#simple_process()
	#process_with_context()

	#pool_1()
	#pool_2()
	#pool_async_1()  # apply_async().
	#pool_async_2()  # apply_async().
	pool_async_3()  # map_async().

	#synchronization()

	#state_sharing_by_shared_memory()
	#state_sharing_by_server_process()

	#--------------------
	#process_pool_executor()
	
	#--------------------
	#multiprocess_and_multithreading()

	#--------------------
	num_processes = 10
	lock = mp.Lock()  # RuntimeError: Lock objects should only be shared between processes through inheritance.
	#lock= mp.Manager().Lock()  # TypeError: can't pickle _thread.lock objects.

	#run_worker_processes(lock, num_processes)

	print('{}: End main process.'.format(os.getpid()))

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
