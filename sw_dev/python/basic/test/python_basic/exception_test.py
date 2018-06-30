#!/usr/bin/env python

# REF [site] >> https://docs.python.org/3/library/exceptions.html

import struct
import sys, traceback

def main():
	raise NotImplementedError('abc', 123, (1, 'a'))
	#raise ValueError('ValueError raised')
	#raise Exception('Fatal error')

#%%------------------------------------------------------------------

if '__main__' == __name__:
	try:
		main()
	except NotImplementedError as ex:
		print('******************************* 1')
		print('NotImplementedError raised: {}, {}, {}'.format(ex, ex.args, str(ex)))
	except ValueError as ex:
		print('******************************* 2')
		#tb = sys.exc_info()[2]
		#raise Exception().with_traceback(tb)
		#----------
		raise
	#except Exception as ex:
	#	print('******************************* 3')
	#	print('Exception raised:', ex)
	except:
		print('******************************* 4')
		#----------
		#ex = sys.exc_info()  # (type, exception object, traceback).
		##print('{} raised: {}.'.format(ex[0], ex[1]))
		#print('{} raised: {}.'.format(ex[0].__name__, ex[1]))
		#traceback.print_tb(ex[2], limit=None, file=sys.stdout)
		#----------
		#traceback.print_exception(*sys.exc_info(), limit=None, file=sys.stdout)
		#----------
		traceback.print_exc(limit=None, file=sys.stdout)
