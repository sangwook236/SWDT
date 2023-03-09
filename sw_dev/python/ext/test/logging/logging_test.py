#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys, os, argparse, logging, logging.handlers

# Define a global variable, print.
#print = print

def get_logger(log_filepath, log_level, is_rotating=True):
	if is_rotating:
		file_handler = logging.handlers.RotatingFileHandler(log_filepath, maxBytes=10000000, backupCount=10)
	else:
		file_handler = logging.FileHandler(log_filepath)
	stream_handler = logging.StreamHandler()

	formatter = logging.Formatter('[%(levelname)s][%(filename)s:%(lineno)s][%(asctime)s] %(message)s')
	#formatter = logging.Formatter('[%(levelname)s][%(asctime)s] %(message)s')
	file_handler.setFormatter(formatter)
	stream_handler.setFormatter(formatter)

	#name = 'A.B.C'
	name = None
	logger = logging.getLogger(name if name else __name__)
	logger.setLevel(log_level)  # {NOTSET=0, DEBUG=10, INFO=20, WARNING=WARN=30, ERROR=40, CRITICAL=FATAL=50}.
	logger.addHandler(file_handler) 
	logger.addHandler(stream_handler) 

	return logger

def output_logs(logger):
	#logger.log(logging.WARNING, '[logger] Warning.')
	logger.debug('[logger] Debug.')
	logger.info('[logger] Info.')
	logger.warning('[logger] Warning.')
	logger.error('[logger] Error.')
	logger.critical('[logger] Critical.')
	try:
		raise Exception('Test exception')
	except Exception as ex:
		#logger.exception('[logger] Exception.')
		logger.exception(ex)

	#print = logger.info  # Locally applied.

	print('[sys.stdout] Print')
	print('[sys.stderr] Print', file=sys.stderr)  # TypeError: _log() got an unexpected keyword argument 'file'.
	#print('[sys.stdout] Print', 'Print2')  # TypeError: not all arguments converted during string formatting.
	#print('[sys.stderr] Print', 'Print2', file=sys.stderr)  # TypeError: not all arguments converted during string formatting.

def root_logger_example(log_level):
	#format = '%(asctime)-15s: %(message)s'
	format = '%(name)s: %(levelname)-10s: %(asctime)-30s: %(message)s'

	# Basic configuration for the logging system by creating a StreamHandler with a default Formatter and adding it to the root logger.
	# The functions debug(), info(), warning(), error() and critical() will call basicConfig() automatically if no handlers are defined for the root logger.
	logging.basicConfig(format=format)
	#logging.basicConfig(filename='root_logger.log', filemode='w', level=log_level)
	"""
	logging.basicConfig(
		level=log_level,
		format=format,
		handlers=[
			logging.StreamHandler(),
			logging.FileHandler('root_logger.log')
		]
	)
	"""

	#logging.log(logging.WARNING, '[root logger] Warning.')  # Logs a message with level 'level' on the root logger.
	logging.debug('[root logger] Debug.')
	logging.info('[root logger] Info.')
	logging.warning('[root logger] Warning.')
	logging.error('[root logger] Error.')
	logging.critical('[root logger] Critical.')
	try:
		raise Exception('Test exception')
	except Exception as ex:
		#logging.exception('[root logger] Exception.')
		logging.exception(ex)

	#-----
	root_logger = logging.getLogger(name=None)
	root_logger.setLevel(log_level)

	output_logs(root_logger)

def simple_logging_example(log_level):
	format = '%(name)s: %(levelname)-10s: %(asctime)-30s: %(message)s'
	logging.basicConfig(format=format)

	logger = logging.getLogger(__name__)

	print('-----')
	output_logs(logger)

	#--------------------
	format = '%(asctime)-15s %(clientip)s %(user)-8s: %(message)s'
	logging.basicConfig(format=format)

	logger = logging.getLogger('tcpserver')

	print('-----')
	d = {'clientip': '192.168.0.1', 'user': 'fbloggs'}
	logger.warning('Protocol problem: %s', 'connection reset', extra=d)

	#--------------------
	filter = logging.Filter(name='A.B')

	loggerA = logging.getLogger('A')
	loggerA.setLevel(log_level)
	loggerA.addFilter(filter)

	loggerAB = logging.getLogger('A.B')
	loggerAB.setLevel(log_level)
	loggerAB.addFilter(filter)

	loggerABC = logging.getLogger('A.B.C')
	loggerABC.setLevel(log_level)
	loggerABC.addFilter(filter)

	loggerAX = logging.getLogger('A.X')
	loggerAX.setLevel(log_level)
	loggerAX.addFilter(filter)

	print('-----')
	output_logs(loggerA)
	output_logs(loggerAB)
	output_logs(loggerABC)
	output_logs(loggerAX)

	#--------------------
	log_filepath = './simple_logging.log'
	#handler = logging.StreamHandler()
	#handler = logging.FileHandler(log_filepath)
	handler = logging.handlers.RotatingFileHandler(log_filepath, maxBytes=5000, backupCount=10)
	formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
	handler.setFormatter(formatter)

	logger = logging.getLogger('simple_logging_logger')
	logger.addHandler(handler) 
	logger.setLevel(log_level)

	for _ in range(1000):
		print('-----')
		output_logs(logger)

def log_setting_function(log_level):
	format = '%(asctime)-15s: %(message)s'
	logging.basicConfig(format=format)

	logger = logging.getLogger('simple_logging_in_multiple_functions_logger')
	logger.setLevel(log_level)

def log_function():
	logger = logging.getLogger('simple_logging_in_multiple_functions_logger')

	output_logs(logger)

def simple_logging_in_multiple_functions_example(log_level):
	logger = logging.getLogger('simple_logging_in_multiple_functions_logger')

	print('-----')
	output_logs(logger)

	log_setting_function(log_level)

	log_function()

	print('-----')
	output_logs(logger)

# REF [site] >> https://stackoverflow.com/questions/19425736/how-to-redirect-stdout-and-stderr-to-logger-in-python
def redirect_stdout_and_stderr_to_logger(log_level):
	class MyStreamLogger1(object):
		def __init__(self, log):
			self.log = log

		def write(self, buf):
			if buf != '\n':
				self.log(buf)

		def flush(self):
			pass

	class MyStreamLogger2(object):
		def __init__(self, logger, level):
			self.logger = logger
			self.level = level
			self.linebuf = ''

		def write(self, buf):
			for line in buf.rstrip().splitlines():
				self.logger.log(self.level, line.rstrip())

		def flush(self):
			pass

	handler = logging.StreamHandler()
	formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
	handler.setFormatter(formatter)

	logger = logging.getLogger(__name__)
	logger.addHandler(handler) 
	logger.setLevel(log_level)

	if False:
		sys.stdout = MyStreamLogger1(logger.info)
		sys.stderr = MyStreamLogger1(logger.error)
	else:
		# Outputs better and consistently.
		sys.stdout = MyStreamLogger2(logger, logging.INFO)
		sys.stderr = MyStreamLogger2(logger, logging.ERROR)

	print('[sys.stdout] log message line #1.\nlog message line #2.\nlog message line #3.')
	print('[sys.stderr] log message line #1.\nlog message line #2.\nlog message line #3.', file=sys.stderr)
	print('[sys.stdout] log message line #1.', 'log message line #2.', 'log message line #3.')
	print('[sys.stderr] log message line #1.', 'log message line #2.', 'log message line #3.', file=sys.stderr)
	print('[sys.stdout] log message line #1.', 'log message line #2.', 'log message line #3.', sep='-----', end='=====')
	print('[sys.stderr] log message line #1.', 'log message line #2.', 'log message line #3.', sep='-----', end='=====', file=sys.stderr)

	#output_logs(logger)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-l', '--loglevel', help='Log level', type=str, required=False, default='debug')  # {NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL}. [0, 50].

	args = parser.parse_args()

	if isinstance(args.loglevel, str):
		log_level = getattr(logging, args.loglevel.upper(), logging.DEBUG)
		if not isinstance(log_level, int):
			raise ValueError('Invalid log level: {}'.format(log_level))
	else:
		raise ValueError('Invalid log level, {}'.format(args.loglevel))

	log_dir_path = './log'
	if not os.path.isdir(log_dir_path):
		os.mkdir(log_dir_path)

	#--------------------
	#logging.basicConfig(level=logging.WARN)
	#logger = logging.getLogger(__name__)

	#--------------------
	if False:
		# Define a logger.
		log_filename = os.path.basename(os.path.normpath(__file__))
		log_filepath = os.path.join(log_dir_path, '{}.log'.format(log_filename if log_filename else 'swdt'))
		logger = get_logger(log_filepath, log_level, is_rotating=True)

		output_logs(logger)

	#--------------------
	if False:
		# Redirection of print().
		#	Refer to redirect_stdout_and_stderr_to_logger().

		#fd_stdout = open(os.path.join(log_dir_path, 'stdout.log'), 'a', encoding='utf-8')
		#sys.stdout = fd_stdout
		#fd_stderr = open(os.path.join(log_dir_path, 'stderr.log'), 'a', encoding='utf-8')
		#sys.stderr = fd_stderr

		# NOTE [error] >> If logging.StreamHandler is set, recursion error will occur.
		#sys.stdout.write = logger.info
		#sys.stderr.write = logger.error

		global print
		#print = logger.info
		print = lambda *objects, sep=' ', end='\n', file=sys.stdout, flush=False: logger.error(*objects) if file == sys.stderr else logger.info(*objects)

	#--------------------
	print('Log level: {}.'.format(log_level))

	#root_logger_example(log_level)
	#simple_logging_example(log_level)
	#simple_logging_in_multiple_functions_example(log_level)

	redirect_stdout_and_stderr_to_logger(log_level)

#--------------------------------------------------------------------

# Usage:
#	python logging_main.py --loglevel=DEBUG
#	python logging_main.py --loglevel=debug

if '__main__' == __name__:
	main()
