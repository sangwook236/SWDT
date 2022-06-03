#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys, traceback, logging

# REF [site] >> https://docs.python.org/3/library/exceptions.html
def basic_exception_handling():
	def raise_exception():
		raise RuntimeError('Error message', 'abc', 123, (1, 'a'))
		#raise ValueError('Error message')
		#raise Exception('Error message')

	try:
		raise_exception()
	except RuntimeError as ex:
	#except Exception as ex:
		print('{} raised: {}, {}, {}.'.format(ex.__class__.__name__, ex, ex.args, str(ex)))

		#----------
		# REF [site] >> https://docs.python.org/3/library/sys.html
		exc = sys.exc_info()  # (type, value(exception object), traceback).
		print('{} raised: {}, {}, {}.'.format(exc[0].__name__, exc[0], exc[1], exc[2]))
		print('\t>>> {}, {}, {}.'.format(type(exc[0]), type(exc[1]), type(exc[2])))

		# REF [site] >> https://docs.python.org/3/library/traceback.html
		print('-----1')
		traceback.print_tb(exc[2], limit=None, file=sys.stdout)
		print('-----2')
		traceback.print_exception(*exc, limit=None, file=sys.stdout, chain=True)
		print('-----3')
		traceback.print_exc(limit=None, file=sys.stdout, chain=True)  # A shorthand for print_exception(*sys.exc_info(), limit, file, chain).
		print('-----4')
		#traceback.print_last(limit=None, file=sys.stdout, chain=True)  # A shorthand for print_exception(sys.last_type, sys.last_value, sys.last_traceback, limit, file, chain).
		print('-----5')
		traceback.print_stack(f=None, limit=None, file=sys.stdout)
		print('-----6')

		#----------
		logging.exception(ex)  # Logs a message with level ERROR on the root logger.
		print('-----7')
	
		#----------
		#raise ex
		raise

def main():
	basic_exception_handling()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
