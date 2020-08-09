#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys, os, argparse, logging, logging.handlers

# Define a global variable, print.
print = print

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

	logger = logging.getLogger(name if name else __name__)
	logger.setLevel(log_level)  # {NOTSET=0, DEBUG=10, INFO=20, WARNING=WARN=30, ERROR=40, CRITICAL=FATAL=50}.
	logger.addHandler(file_handler) 
	logger.addHandler(stream_handler) 

	return logger

def output_logs(logger):
	#logger.log(logging.WARNING, '[Warning] Warning.')
	logger.debug('[logger] Debug.')
	logger.info('[logger] Info.')
	logger.warning('[logger] Warning.')
	logger.error('[logger] Error.')
	logger.critical('[logger] Critical.')
	try:
		raise Exception('Test exception')
	except Exception as ex:
		logger.exception('[logger] Exception.')

	#print = logger.info  # Locally applied.

	print('[sys.stdout] Print')
	print('[sys.stderr] Print', file=sys.stderr)  # TypeError: _log() got an unexpected keyword argument 'file'.
	#print('[sys.stdout] Print', 'Print2')  # TypeError: not all arguments converted during string formatting.
	#print('[sys.stderr] Print', 'Print2', file=sys.stderr)  # TypeError: not all arguments converted during string formatting.

def simple_logging(log_level):
	#format = '%(asctime)-15s %(clientip)s %(user)-8s: %(message)s'
	format = '%(asctime)-15s: %(message)s'
	logging.basicConfig(format=format)
	#logging.basicConfig(filename='python_logging.log', filemode='w', level=log_level)

	logger = logging.getLogger(__name__)
	logger.setLevel(log_level)

	output_logs(logger)

def simple_file_logging(log_level):
	handler = logging.handlers.RotatingFileHandler('./python_logging.log', maxBytes=5000, backupCount=10)
	formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
	handler.setFormatter(formatter)

	logger = logging.getLogger('python_logging_test')
	logger.addHandler(handler) 
	logger.setLevel(log_level)

	for _ in range(1000):
		output_logs(logger)

def log_setting_function(log_level):
	format = '%(asctime)-15s: %(message)s'
	logging.basicConfig(format=format)

	logger = logging.getLogger('python_logging_test')
	logger.setLevel(log_level)

def log_function():
	logger = logging.getLogger('python_logging_test')

	output_logs(logger)

def simple_logging_in_multiple_functions(log_level):
	logger = logging.getLogger('python_logging_test')

	output_logs(logger)

	log_setting_function(log_level)

	log_function()

	output_logs(logger)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-l', '--loglevel', help='Log level')  # {NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL}. [0, 50]

	args = parser.parse_args()

	if args.loglevel is not None:
		log_level = getattr(logging, args.loglevel.upper(), None)
		if not isinstance(log_level, int):
			raise ValueError('Invalid log level: {}'.format(log_level))
	else:
		#log_level = logging.WARNING
		log_level = logging.DEBUG

	log_dir_path = './log'
	if not os.path.isdir(log_dir_path):
		os.mkdir(log_dir_path)

	# Define a logger.
	log_filename = os.path.basename(os.path.normpath(__file__))
	log_filepath = os.path.join(log_dir_path, '{}.log'.format(log_filename if log_filename else 'swdt'))
	logger = get_logger(log_filepath, log_level, is_rotating=True)

	# Redirection of print().
	if True:
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

	output_logs(logger)

	#simple_logging(log_level)
	#simple_file_logging(log_level)
	simple_logging_in_multiple_functions(log_level)

#--------------------------------------------------------------------

# Usage:
#	python logging_main.py --loglevel=DEBUG
#	python logging_main.py --loglevel=debug

if '__main__' == __name__:
	main()
