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

# Usage:
#	python -m unittest arithmetic_test
#	python -m unittest arithmetic_test.py

if '__main__' == __name__:
	unittest.main()
