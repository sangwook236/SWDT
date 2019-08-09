#!/usr/bin/env python

import logging
from logging.handlers import RotatingFileHandler
import argparse

def output_logs(logger):
	#logger.log(logging.WARNING, '[Warning] Warning.')
	logger.debug('[Debug] Debug.')
	logger.info('[Info] Info.')
	logger.warning('[Warning] Warning.')
	logger.error('[Error] Error.')
	logger.critical('[Critical] Critical.')
	logger.exception('[Exception] Exception.')

def simple_logging(log_level):
	#format = '%(asctime)-15s %(clientip)s %(user)-8s: %(message)s'
	format = '%(asctime)-15s: %(message)s'
	logging.basicConfig(format=format)
	#logging.basicConfig(filename='python_logging.log', filemode='w', level=log_level)

	logger = logging.getLogger(__name__)
	logger.setLevel(log_level)

	output_logs(logger)

def simple_file_logging(log_level):
	handler = RotatingFileHandler('./python_logging.log', maxBytes=5000, backupCount=10)
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
		log_level = logging.WARNING
	print('Log level:', log_level)

	#simple_logging(log_level)
	#simple_file_logging(log_level)
	simple_logging_in_multiple_functions(log_level)

#%%------------------------------------------------------------------

# Usage:
#	python logging_main.py --loglevel=DEBUG
#	python logging_main.py --loglevel=debug

if '__main__' == __name__:
	main()
