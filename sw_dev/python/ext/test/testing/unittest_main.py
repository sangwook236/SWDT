#!/usr/bin/env python

# REF [site] >> https://docs.python.org/3/library/unittest.html

import unittest
from add_test import AddTestCase
from subtract_test import SubtractTestCase

def suite():
	suite = unittest.TestSuite()
	suite.addTest(AddTestCase())
	suite.addTest(SubtractTestCase())
	return suite

#%%------------------------------------------------------------------

# Usage:
#	python -m unittest unittest_main
#	python -m unittest unittest_main.py
#
#	python -m unittest discover -s project_directory -p "*_test.py"

if '__main__' == __name__:
	unittest.main()
