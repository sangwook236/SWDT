#!/usr/bin/env python

# REF [site] >> https://docs.pytest.org/en/latest/

import pytest

def main():
	pass

#%%------------------------------------------------------------------

# REF [site] >> https://docs.pytest.org/en/latest/usage.html
# All test functions must have their names starting with 'test_'.
# All test classes must have their names starting with 'Test_'.

# Usage:
#	pytest
#	python -m pytest
#		Runs tests in files starting with 'test_' or ending with '_test'.
#	pytest dir_path
#	python -m pytest dir_path
#	pytest python_module.py
#	python -m pytest python_module.py
#		Runs tests in the specified python module which does not need to have its name starting with 'test_' or ending with '_test'.
#	pytest python_module.py::test_func
#	python -m pytest python_module.py::test_func
#	pytest -k "MyClass and not method"
#	python -m pytest -k "MyClass and not method"
#		Will run TestMyClass.test_something but not TestMyClass.test_method_simple.

if '__main__' == __name__:
	main()
