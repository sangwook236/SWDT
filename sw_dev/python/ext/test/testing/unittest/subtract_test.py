#!/usr/bin/env python

import unittest
from subtract import subtract

class SubtractTestCase(unittest.TestCase):
	def setUp(self):
		pass

	def tearDown(self):
		pass

	def test_subtract(self):
		self.assertEqual(subtract(1, 2), -1)
		self.assertEqual(subtract(1, 2), 0)
		self.assertEqual(subtract(2, 1), 1)

#%%------------------------------------------------------------------

# Usage:
#	python subtract_test.py

if '__main__' == __name__:
	unittest.main(verbosity=1)
