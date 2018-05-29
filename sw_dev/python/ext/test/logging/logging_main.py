#!/usr/bin/env python

import logging
from logging.handlers import RotatingFileHandler

def simple_logging():
	#format = '%(asctime)-15s %(clientip)s %(user)-8s %(message)s'
	format = '%(asctime)-15s %(message)s'
	logging.basicConfig(format=format)

	logger = logging.getLogger(__name__)
	logger.setLevel(logging.INFO)

	logger.log(logging.WARNING, '[Warning] Warning.')
	logger.debug('[Debug] Debug.')
	logger.info('[Info] Info.')
	logger.warning('[Warning] Warning.')
	logger.error('[Error] Error.')
	logger.critical('[Critical] Critical.')
	logger.exception('[Exception] Exception.')

def simple_file_logging():
	handler = RotatingFileHandler('./python_logging.log', maxBytes=5000, backupCount=10)
	formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
	handler.setFormatter(formatter)

	logger = logging.getLogger('myapp')
	logger.addHandler(handler) 
	logger.setLevel(logging.WARNING)

	for _ in range(1000):
		logger.log(logging.WARNING, '[Warning] Warning.')
		logger.debug('[Debug] Debug.')
		logger.info('[Info] Info.')
		logger.warning('[Warning] Warning.')
		logger.error('[Error] Error.')
		logger.critical('[Critical] Critical.')
		logger.exception('[Exception] Exception.')

def main():
	#simple_logging()
	simple_file_logging()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
