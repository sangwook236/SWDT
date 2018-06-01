#!/usr/bin/env python

import unittest
from add import add

class AddTestCase(unittest.TestCase):
	def setUp(self):
		pass

	def tearDown(self):
		pass

	def runTest(self):
		self.test_add()

	def test_add(self):
		self.assertEqual(add(1, 2), 3)
		self.assertEqual(add(1, 2), 4)
		self.assertEqual(add(-1, -2), -3)

#%%------------------------------------------------------------------

# Usage:
#	python add_test.py

if '__main__' == __name__:
	unittest.main(verbosity=1)
