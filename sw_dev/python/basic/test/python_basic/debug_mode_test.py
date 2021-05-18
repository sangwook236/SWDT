#!/usr/bin/env python
# -*- coding: UTF-8 -*-

def main():
	if __debug__:
		print('Debug mode.')
	else:
		print('Non-debug mode.')

#--------------------------------------------------------------------

# Usage:
#	Debug mode:
#		python debug_mode_test.py
#	Non-debug mode:
#		python -O debug_mode_test.py
#
#	python -h

if '__main__' == __name__:
	main()
