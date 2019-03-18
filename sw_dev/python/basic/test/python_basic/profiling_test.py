#!/usr/bin/env python

# REF [site] >> https://docs.python.org/3/library/profile.html

import cProfile, pstats, io
import re

def main():
	use_stats_file = False

	if use_stats_file:
		stats_filepath = 'restats'
		cProfile.run('re.compile("foo|bar")', stats_filepath)
	else:
		pr = cProfile.Profile()
		pr.enable()

		# Do something.
		re.compile('foo|bar')

		pr.disable()
	
	#--------------------

	if use_stats_file:
		ps = pstats.Stats(stats_filepath)
	else:
		s = io.StringIO()
		ps = pstats.Stats(pr, stream=s)

	ps.sort_stats()
	#sortby = pstats.SortKey.CUMULATIVE
	#ps.sort_stats(sortby)

	ps.strip_dirs()
	ps.print_stats()
	ps.print_callers()
	ps.print_callees()

	if not use_stats_file:
		print(s.getvalue())

#%%------------------------------------------------------------------

# Usage:
#	1) python profiling_test.py
#	2) python -m cProfile profiling_on_existing_application.py

if '__main__' == __name__:
	main()
