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

#%%------------------------------------------------------------------

# Usage:
#	python -m unittest add_test
#	python -m unittest add_test.py
#	python -m unittest add_test subtract_test
#	python -m unittest add_test.py subtract_test.py

if '__main__' == __name__:
	unittest.main()
