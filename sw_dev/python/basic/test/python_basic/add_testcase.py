#!/usr/bin/env python

# REF [site] >> https://docs.python.org/3/library/unittest.html

import unittest
from add import add

class AddTestCase(unittest.TestCase):
	def setUp(self):
		pass

	def tearDown(self):
		pass

	def test_add(self):
		self.assertEqual(add(1, 2), 3)
		self.assertEqual(add(-1, -2), -3)

# Usage:
#	python -m unittest add_testcase
#	python -m unittest add_testcase.py
#	python -m unittest add_testcase subtract_testcase
#	python -m unittest add_testcase.py subtract_testcase.py

if '__main__' == __name__:
	unittest.main()
