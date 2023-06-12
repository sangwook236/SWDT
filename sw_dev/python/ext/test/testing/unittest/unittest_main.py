#!/usr/bin/env python

# REF [site] >> https://docs.python.org/3/library/unittest.html
#	python -m unittest -h
#	python -m unittest
#	python -m unittest discover
#	python -m unittest module_name
#	python -m unittest -v module_name
#	python -m unittest module_name module_name.class_name module_name.class_name.func_name module_name.func_name
#	python -m unittest discover -s dir_path -p "*_test.py"

import unittest
from add_test import AddTestCase
from subtract_test import SubtractTestCase

# Method 3.
def suite():
	suite = unittest.TestSuite()
	#suite.addTest(AddTestCase('runTest'))
	suite.addTest(AddTestCase())
	suite.addTest(SubtractTestCase('test_subtract'))
	return suite

#--------------------------------------------------------------------

# Usage:
#	Method 1:
#		python -m unittest discover -p "*_test.py"
#		python -m unittest discover -p "*_test.py" -v
#			Test cases in test files that match '*_test.py' run.
#			The main routine below does not run.
#		python -m unittest unittest_main
#			Test cases run which are imported by 'import' statements or are defined in this file.
#			The main routine below does not run.
#	Method 2 & 3:
#		python unittest_main.py
#			The main routine below runs.

if '__main__' == __name__:
	# Method 2.
	#	Test cases run which are imported by 'import' statements or are defined in this file.
	#print('[UnitTest] Test cases run using unittest.main().')
	#unittest.main(verbosity=1)

	# Method 3.
	#	Test cases added in suite() run.
	print('[UnitTest] Test cases run using unittest.TextTestRunner.')
	runner = unittest.TextTestRunner()
	runner.run(suite())
