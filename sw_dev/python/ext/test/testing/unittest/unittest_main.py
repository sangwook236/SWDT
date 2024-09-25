#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import unittest
from add_test import AddTestCase
from subtract_test import SubtractTestCase

#--------------------------------------------------------------------

# Usage:
#	https://docs.python.org/3/library/unittest.html
#
#	python -m unittest --help
#
#	python -m unittest
#	python -m unittest -v
#	python -m unittest -q
#	python -m unittest <module_name> <module_name>.<class_name> <module_name>.<class_name>.<function_name> <module_name>.<function_name>
#	
#	For test modules:
#		python -m unittest <module_name>
#	For test classes:
#		python -m unittest <module_name>.<class_name>
#	For test methods:
#		python -m unittest <module_name>.<function_name>
#		python -m unittest <module_name>.<class_name>.<function_name>
#	For test files:
#		python -m unittest <path/to/test_file.py>
#
#	Test discovery:
#		For test discovery all test modules must be importable from the top level directory of the project
#
#		python -m unittest discover --help
#
#		python -m unittest discover
#		python -m unittest discover -s <path/to/test_dir> -p <pattern> -k <test_name_patterns> -t <path/to/top_level_dir>
#			python -m unittest discover -s '.' -p 'test*.py'
#
#		Run tests in a specific directory:
#			python -m unittest discover -s <path/to/test_dir>
#				python -m unittest discover -s "./tests"
#		Run tests in files which match a given pattern:
#			python -m unittest discover -p <pattern>
#				python -m unittest discover -p "*_test.py" (ending with '_test')
#		Run tests which match a given substring:
#			python -m unittest discover -k <test_name_patterns>
#				python -m unittest discover -k "MyMath"
#				python -m unittest discover -k "test_"
#
#	Coverage:
#		https://coverage.readthedocs.io/en/latest/index.html
#
#		Install:
#			pip install coverage
#
#		Run your tests under coverage:
#			coverage run -m <module_name> <arg1> <arg2> ... <argN>
#
#		Run your test suite and gather data:
#			coverage run -m unittest discover
#		Report on the results:
#			coverage report -m
#		Get annotated HTML listings detailing missed lines:
#			coverage html

if "__main__" == __name__:
	# Usage:
	#	python unittest_main.py

	if True:
		print("[UnitTest] Test cases run using unittest.main().")
		#unittest.main()
		unittest.main(verbosity=1)
	else:
		print("[UnitTest] Test cases run using unittest.TextTestRunner.")
		suite = unittest.TestSuite()
		if True:
			suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(AddTestCase))
			suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(SubtractTestCase))
		else:
			suite.addTest(AddTestCase())  # NOTE [info] >> only when 'AddTestCase' object has the attribute 'runTest'
			#suite.addTest(AddTestCase("runTest"))
			#suite.addTest(AddTestCase("test_add"))
			#suite.addTest(AddTestCase.test_add)  # Error
			#suite.addTest(SubtractTestCase())  # AttributeError: 'SubtractTestCase' object has no attribute 'runTest'. Did you mean: 'subTest'?
			suite.addTest(SubtractTestCase("test_subtract"))
		unittest.TextTestRunner().run(suite)
