#!/usr/bin/env python

import logging

def main():
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

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
