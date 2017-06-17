import os

#%%------------------------------------------------------------------
# Assert.

#assert(2 + 2 == 5, "Error: addition.")  # Error: not working.
assert 2 + 2 == 5, "Error: addition."

if __debug__:
	if not 2 + 2 == 5:
		raise AssertionError
		#raise AssertionError, "Error: addition."  # Error: invalid syntax.

#%%------------------------------------------------------------------
# Exception.

if not os.path.exists(prediction_dir_path):
	try:
		os.makedirs(prediction_dir_path)
	except OSError as ex:
		if ex.errno != os.errno.EEXIST:
			raise
