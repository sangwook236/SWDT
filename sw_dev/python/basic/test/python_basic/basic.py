import os, sys, platform

#%%------------------------------------------------------------------
# Platform.

os.name

sys.platform

platform.platform()
platform.system()
platform.machine()

platform.uname()
platform.release()
platform.version()

platform.dist()
platform.linux_distribution()
platform.mac_ver()

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

#%%------------------------------------------------------------------
# Lambda expression.

def make_incrementor(n):
	return lambda x: x + n

increment_func = make_incrementor(5)
print(increment_func(1))
print(increment_func(3))

a = [1, 2, 3]
b = [4, 5, 6]
c = [7, 8, 9]
a_plus_b_plus_c = list(map(lambda x, y, z: x + y + z, a, b, c))
print('a + b + c =', a_plus_b_plus_c)

#%%------------------------------------------------------------------
# Map, filter, reduce.

items = [1, 2, 3, 4, 5]
squared = map(lambda x: x**2, items)
print('Type of squared =', squared)
print('squared =', list(squared))

def mul(x):
    return x * x
def add(x):
    return x + x

funcs = [mul, add]
for i in range(1, 6):
    value = list(map(lambda x: x(i), funcs))
    print(value)

#--------------------
number_list = range(-5, 5)
less_than_zero = filter(lambda x: x < 0, number_list)
print('Type of less_than_zero =', less_than_zero)
print('less_than_zero =', list(less_than_zero))

#--------------------
from functools import reduce

items = [3, 4, 5, 6, 7]
summation = reduce((lambda x, y: x + y), items)
print('summation =', summation)
product = reduce((lambda x, y: x * y), items)
print('product =', product)
