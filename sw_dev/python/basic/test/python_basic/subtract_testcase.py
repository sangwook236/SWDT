#!/usr/bin/env python

# REF [site] >> https://docs.python.org/3/library/unittest.html

import unittest
from subtract import subtract

class SubtractTestCase(unittest.TestCase):
	def setUp(self):
		pass

	def tearDown(self):
		pass

	def test_subtract(self):
		self.assertEqual(subtract(1, 2), -1)
		self.assertEqual(subtract(2, 1), 1)

# Usage:
#	python -m unittest subtract_testcase
#	python -m unittest subtract_testcase.py
#	python -m unittest add_testcase subtract_testcase
#	python -m unittest add_testcase.py subtract_testcase.py

if '__main__' == __name__:
	unittest.main()
