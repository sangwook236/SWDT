#!/usr/bin/env python

# REF [site] >> https://docs.python.org/3/library/unittest.html

import unittest
from add_testcase import AddTestCase
from subtract_testcase import SubtractTestCase

def suite():
	suite = unittest.TestSuite()
	suite.addTest(AddTestCase())
	suite.addTest(SubtractTestCase())
	return suite

# Usage:
#	python -m unittest arithmetic_testsuite
#	python -m unittest arithmetic_testsuite.py

if '__main__' == __name__:
	unittest.main()
