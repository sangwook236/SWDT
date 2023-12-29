#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import random, io, re
import cProfile, pstats

def adder(iterations):
	sum = 0
	for _ in range(iterations):
		sum += random.randrange(100)
	return sum

# REF [site] >> https://docs.python.org/3/library/profile.html
def simple_example():
	stats_filepath = 'restats'

	if stats_filepath:
		#cProfile.run('re.compile("foo|bar")', stats_filepath)
		#cProfile.run('re.compile("foo|bar")', stats_filepath, sort='time')
		cProfile.run('adder(iterations=1000000)', stats_filepath, sort='time')
	else:
		# Directly using the Profile class allows formatting profile results without writing the profile data to a file:
		pr = cProfile.Profile()
		pr.enable()

		# Do something.
		#re.compile('foo|bar')
		adder(iterations=1000000)

		pr.disable()

	#--------------------
	if stats_filepath:
		ps = pstats.Stats(stats_filepath)
	else:
		s = io.StringIO()
		ps = pstats.Stats(pr, stream=s)

	ps.sort_stats()
	#sortby = pstats.SortKey.CUMULATIVE
	#ps.sort_stats(sortby)

	print('-------------------------------------------------- ps.strip_dirs()')
	ps.strip_dirs()
	print('-------------------------------------------------- ps.print_stats()')
	ps.print_stats()
	print('-------------------------------------------------- ps.print_callers()')
	ps.print_callers()
	print('-------------------------------------------------- ps.print_callees()')
	ps.print_callees()

	if not stats_filepath:
		print('-------------------------------------------------- s.getvalue()')
		print(s.getvalue())

def main():
	simple_example()

#--------------------------------------------------------------------

# Usage:
#	python profiling_test.py
#
#	In terminal:
#		python -m cProfile profiling_on_existing_application.py
#
#		python -m cProfile -o ./test.prof test.py
#		snakeviz ./test.prof
#
#		vprof -c cpmh test.py
#		vprof -c cpmh "test.py --arg1 --arg2"
#
#	Refer to ./memory_test.py

if '__main__' == __name__:
	main()
